import os
import random
import logging
import threading
from threading import Thread
from datetime import datetime
from pathlib import Path
from bson import ObjectId
from typing import Optional
import numpy as np

from .base import EngineStrategy

from pylib.mongo_db import EnumStatus
from pylib.dto.database import Simulation, SimulationConfig, Generation
from pylib import plot_network

from lib.util.build_input_sim_cooja import create_files
from lib.util.population import PopulationSnapshot, select_next_population

# NSGA utils
from lib.nsga import fast_nondominated_sort
from lib.nsga import generate_reference_points, niching_selection
from lib.genetic_operators.selection import tournament_selection_2, compute_individual_ranks
# Problem Adapter
from lib.problem.adapter import ProblemAdapter, Chromosome
from lib.problem.resolve import build_adapter

logger = logging.getLogger(__name__)

class NSGA3LoopStrategy(EngineStrategy):
    """
    NSGA-III loop strategy integrated with SimLab.

    - Generates an initial population (fixed 2D mote positions).
    - Queues simulations in MongoDB (Change Streams trigger execution on the master node).
    - Upon receiving the results (DONE status), calculates the objectives, 
        performs NSGA-III selection with reference point niching,
        creates the next generation (offspring via SBX + Polynomial Mutation),
        and repeats until `number_of_generations` is reached.
    """
    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        self._watch_thread: Thread | None = None
        self._stop_flag: bool = False

        # --- experiment parameters ---
        params = experiment.get("parameters", {}) or {}
        algorithm_config = params.get("algorithm", {}) or {}
        problem_config = params.get("problem", {}) or {}
        simulation_config = params.get("simulation", {}) or {}
        
        src_repo_opts = experiment.get("source_repository_options", {}) or {}
        
        self.source_repository_options: dict[str, ObjectId] = {
            str(k): ObjectId(v) if isinstance(v, (str, bytes)) else v
            for k, v in src_repo_opts.items()
        }
        
        # --- simulation and algorithm parameters ---
        self._sim_duration: int = int(simulation_config.get("duration", 120))
        self._pop_size: int = int(algorithm_config.get("population_size", 20))
        self._max_gen: int = int(algorithm_config.get("number_of_generations", 5))
                        
        self._prob_cx = float(algorithm_config.get("prob_cx", 0.8))
        self._prob_mt = float(algorithm_config.get("prob_mt", 0.2))
                        
        self._problem_adapter: ProblemAdapter = build_adapter(problem_config, algorithm_config)
                
        # prepare objective infos
        cfg = experiment.get("transform_config", {}) or {}
        obj = cfg.get("objectives", []) or []
        self._objective_keys: list[str] = [o["name"] for o in obj]
        self._objective_goals: list[int] = [1 if o["goal"]=='min' else -1 for o in obj]
        if len(self._objective_keys) != len(self._objective_goals):
            raise ValueError(
                f"objective_keys ({len(self._objective_keys)}) and "
                f"objective_goals ({len(self._objective_goals)}) length mismatch"
            )
            
        # --- nsga3 niching ---
        self._divisions: int = int(algorithm_config.get("divisions", 10))      
        self._ref_points = generate_reference_points(len(self._objective_keys), self._divisions)
        
        # --- loop state ---
        self._exp_id: ObjectId | None = None
        self._gen_index: int = 0
        self._gen_id: ObjectId | None = None
        self._evaluated_count: int = 0

        # --- nsga3 workflow ---
        self._current_population: list[Chromosome] = []        
        self._parents = PopulationSnapshot()                 
        self._map_genome_sim: dict[Chromosome, str] = {}  
        self._map_sim_objectives: dict[str, list[float]] = {}


# ------------------------------
# Interface EngineStrategy
# ------------------------------
    # START implementation
    def start(self):
        """
        Initializes the population and creates Generation 1 with `population_size` simulations.
        """
        self._exp_id = ObjectId(self.experiment["_id"]) if isinstance(self.experiment.get("_id"), (str, bytes)) else self.experiment.get("_id")
        if not isinstance(self._exp_id, ObjectId):
            self._exp_id = ObjectId(str(self.experiment.get("_id")))
        self._gen_index = 0

        # Initial Population P_0
        self._current_population = self._problem_adapter.random_individual_generator(self._pop_size)

        # Enqueue Simulations to first (compute F_0=f(P_0))
        self._generation_enqueue()

        # Starts watcher for receive results from this generation
        self._start_watcher()

    # EVENT_SIMULATION_DONE implementation
    def event_simulation_done(self, result_doc: dict):
        """ 
        On event Simulation Result Done
        Args: result_doc (dict): Simulation dictionary
        """
        if self._stop_flag or self._gen_id is None:
            return

        logger.info("EVENT SIMULATION RESULT DONE")
        sim = result_doc.get("fullDocument")

        # Ensures that this result is from the current generation
        if str(sim.get("generation_id")) != str(self._gen_id):
            return

        sim_id = str(sim.get("_id"))
        logger.info(f"sim_id={sim_id}")

        obj = self._extract_objectives_to_minimization(sim)
        if obj is None:
            logger.info(f"objectives not found in {sim_id}")
            return

        self._map_sim_objectives[sim_id] = obj
        
        self._evaluated_count+=1

        # check generation is complete
        logger.info(f"{self._evaluated_count} of {len(self._current_population)}")
        if self._evaluated_count >= len(self._current_population):
            self._on_generation_completed()

    # STOP implementation
    def stop(self):
        self._stop_flag = True
        t = self._watch_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._watch_thread = None

# ------------------------------
# Watcher (Change Stream)
# ------------------------------
    def _start_watcher(self):
        # encerra watcher anterior, se houver
        self._stop_flag = False
        if self._watch_thread and self._watch_thread.is_alive():
            self._stop_flag = True
            self._watch_thread.join(timeout=1.0)
            self._stop_flag = False

        def _run():
            # This method already opens the Change Stream with a fixed pipeline. (status == DONE)
            def _callback(result_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_simulation_done(result_doc)
                except Exception:
                    logger.exception(f"[NSGA-III] Watcher Callback Error.")

            logger.info("[NSGA-III] Starting Simulations watcher (DONE).")
            self.mongo.simulation_repo.watch_status_done(_callback)

        self._watch_thread = Thread(target=_run, daemon=True)
        self._watch_thread.start()

# ------------------------------
# Encodes genome into simulation elements
# ------------------------------
    def _build_simulation(self, 
            genome: Chromosome,
            exp_oid: ObjectId,
            gen_oid: ObjectId,
            gen_index: int,
            ind_idx: int
            )-> ObjectId:
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Encodes genome into simulation elements
        simulationElements = self._problem_adapter.encode_simulation_input(genome)
        
        config: SimulationConfig = {
                "name": f"nsga3-g{gen_index}-{ind_idx}",
                "duration": self._sim_duration,
                "radiusOfReach": self._problem_adapter.radius_of_reach,
                "radiusOfInter": self._problem_adapter.radius_of_inter,
                "region": self._problem_adapter.bounds,
                "simulationElements": simulationElements
            }

        files_ids = create_files(config, self.mongo.fs_handler)
        image_tmp_path = tmp_dir / f"topology-{exp_oid}-{gen_oid}-{ind_idx}.png"
        plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
        topology_picture_id = self.mongo.fs_handler.upload_file(
            str(image_tmp_path),
            f"topology-{exp_oid}-{gen_oid}-{ind_idx}"
        )
        if os.path.exists(image_tmp_path):
            os.remove(image_tmp_path)
             
        genome, src_id = genome.get_source_by_mac_protocol(self.source_repository_options)
             
        sim_doc: Simulation = {
            "id": ind_idx,
            "experiment_id": exp_oid,
            "generation_id": gen_oid,
            "status": EnumStatus.WAITING,
            "start_time": None,
            "end_time": None,
            "parameters": config,
            "pos_file_id": files_ids.get("pos_file_id", ""),
            "csc_file_id": files_ids.get("csc_file_id", ""),
            "source_repository_id": src_id,
            "topology_picture_id": topology_picture_id,
            "log_cooja_id": "",
            "runtime_log_id": "",
            "csv_log_id": ""
        }
        sim_oid = self.mongo.simulation_repo.insert(sim_doc)
        return sim_oid

# ------------------------------
# Generation / Queuing
# ------------------------------
    def _generation_enqueue(self) -> None:
        """
        Cria a `Generation` e insere `population_size` simulações no MongoDB.
        """
        assert self._exp_id is not None
        exp_oid = self._exp_id
        gen_index = self._gen_index
        population = self._current_population

        gen_doc: Generation = {
            "index": gen_index,
            "experiment_id": exp_oid,
            "status": EnumStatus.BUILDING,
            "start_time": datetime.now(),
            "end_time": None,
            "simulations_ids": []
        }
        gen_oid: ObjectId = self.mongo.generation_repo.insert(gen_doc)
        self._gen_id = gen_oid

        self._evaluated_count = 0
        simulation_ids: list[ObjectId] = []
        
        for i, genome in enumerate(population):
            if genome in self._map_genome_sim.keys():
                self._evaluated_count+=1
                logger.info(f"genome computed: count={self._evaluated_count}")
                continue
            sim_oid = self._build_simulation(genome, exp_oid, gen_oid, gen_index, i)
            logger.info(f"sim_oid={sim_oid}")
            self._map_genome_sim[genome] = str(sim_oid)
            simulation_ids.append(sim_oid)

        # update generation
        self.mongo.generation_repo.update(gen_oid, {
            "simulations_ids": [str(_id) for _id in simulation_ids],
            "status": EnumStatus.WAITING
        })
        self.mongo.generation_repo.mark_waiting(gen_oid)

        if gen_index == 1:
            self.mongo.experiment_repo.update(str(exp_oid), {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now(),
                "generations_ids": [str(gen_oid)]
            })

        logger.info(f"[NSGA-III] Generation {gen_index} enqueued with {len(population)} Simulations.")
        
        self.mongo.experiment_repo.add_generation(exp_oid, gen_oid)


    def _objectives_to_original(self, vec: list[float]) -> dict[str, float]:
        return {
            k: float(v * s)
            for k, v, s in zip(self._objective_keys, vec, self._objective_goals)
        }
    
# ---------------------------------------
# Conclusion of generation and evolution
# ---------------------------------------
    def _on_generation_completed(self):
        assert self._gen_id is not None
        logger.info(f"[NSGA-III] Generation {self._gen_index} completed.")

        # close current generation
        self.mongo.generation_repo.mark_done(self._gen_id)
        
        self._evolution()


# ---------------------------------------
# NSGA-III Evolution
# ---------------------------------------
    def _evolution(self) -> None:
        """
        Two-phase (μ+λ) loop:
        Phase P_1: parents (P_0) finished -> produce offspring P_1 and enqueue -> return.
        Phase P_{t+1}: offspring (P_t) finished -> environmental selection on R_t = P_t U P_{t-1} -> spawn P_{t+1}.
        """        
        # ---------------- First PHASE P_1 ----------------
        if self._parents.dont_have_objectives():
            # Store P_0 (genomes + objectives) = newly assessed population
            self._parents.set(
                genomes= self._current_population,
                genome_to_sim_id= self._map_genome_sim,
                sim_to_objectives= self._map_sim_objectives
            )

            # Generate P_1 from P_0
            offspring = self._run_genetic_algorithm()
            
            # current population is P_1 and next generation
            self._gen_index += 1
            self._current_population = offspring
            self._generation_enqueue()
            
            logger.info("[NSGA-III] Enqueued P_{t+1}; waiting results.")
            return
        
        # stop condition?
        if self._gen_index >= self._max_gen:
            try:
                all_objectives: list[list[float]] = []
                pareto_items: list[dict] = []

                for genome, sim_id in self._map_genome_sim.items():
                    objetives = self._map_sim_objectives[sim_id]
                    pareto_items.append({
                        "simulation_id": sim_id,
                        "chromosome": genome.to_dict(),
                        "objectives": self._objectives_to_original(objetives),
                    })
                    all_objectives.append(objetives)
                        
                pareto_fronts = fast_nondominated_sort(all_objectives)
                first_pareto_front = [pareto_items[idx] for idx in pareto_fronts[0]]

            except Exception:
                logger.exception("[NSGA-III] Could not compute final Pareto front.")
                first_pareto_front = []

            self._finalize_experiment(pareto_front=first_pareto_front)
            return        
        
        # ------- PHASE P_{t+1}: offspring done -> environmental selection on union -------       
        p_t_snapshot = PopulationSnapshot()
        p_t_snapshot.set(
            genomes= self._current_population,
            genome_to_sim_id= self._map_genome_sim,
            sim_to_objectives= self._map_sim_objectives
        )
        
        # union R_t = P_t ∪ P_{t-1}
        p_previous = np.array(self._parents.get_objectives(), dtype=float)
        p_current = np.array(p_t_snapshot.get_objectives(), dtype=float)
        
        if p_previous.ndim != 2 or p_previous.shape[0] == 0:
            error_msg = "Invalid objective matrix P_{t-1}; aborting."
            logger.exception(f"[NSGA-III] {error_msg}")
            self._finalize_experiment(system_msg=error_msg)
            return
        if p_current.ndim != 2 or p_current.shape[0] == 0:
            error_msg = "Invalid objective matrix P_t; aborting."
            logger.exception(f"[NSGA-III] {error_msg}")
            self._finalize_experiment(system_msg=error_msg)
            return

        # ------- PHASE Pareto front and environmental selection -------
        # concatenate
        R_F_list = [list(row) for row in p_previous.tolist()] + [list(row) for row in p_current.tolist()]
            
        # fast non-dominated sort on union
        fronts = fast_nondominated_sort(R_F_list)
        if not fronts:            
            error_msg = "No fronts on union; aborting."
            logger.exception(f"[NSGA-III] {error_msg}")
            self._finalize_experiment(system_msg=error_msg)
            return

        # environmental selection
        selected_idx: list[int] = []
        for front in fronts:
            if len(selected_idx) + len(front) <= self._pop_size:
                selected_idx.extend(front)
            else:
                remaining = self._pop_size - len(selected_idx)
                if remaining > 0:
                    partial = niching_selection(front, R_F_list, self._ref_points, remaining)
                    selected_idx.extend(partial)
                break

        # build next parents Q=E(R_t,α_t)
        self._parents = select_next_population(
            selected_idxs= selected_idx,
            pop_size= self._pop_size,
            P= self._parents,
            Q= p_t_snapshot,
            genome_to_sim_id= self._map_genome_sim,
            sim_to_objectives= self._map_sim_objectives
        )
        
        # ------- PHASE Q: parents done -> generate offspring and enqueue -------
        # produce P_{t+1} from GA(Q,H) (variation on parents)            
        offspring = self._run_genetic_algorithm()

        # enqueue P_{t+1} as next "generation" to be evaluated
        self._gen_index += 1
        self._current_population = offspring
        self._generation_enqueue()        
        
        logger.info("[NSGA-III] Offspring enqueued; waiting for P_t results to perform environmental selection.")

        return
    
# ---------------------------------------
# Run Genetic Algorithm
# ---------------------------------------   
    def _run_genetic_algorithm(self) -> list[list[float]]:
        rng = random.Random()
        parents = self._parents.get_genomes()
        objectives = self._parents.get_objectives()
        children: list[Chromosome] = []        
        fronts: list[list[int]] = fast_nondominated_sort(objectives)
        individual_ranks: dict[int, int] = compute_individual_ranks(fronts)
        while len(children) < self._pop_size:
            # Selection
            parent1: Chromosome = tournament_selection_2(parents, individual_ranks)
            parent2: Chromosome = tournament_selection_2(parents, individual_ranks)
            # Crossover 
            if rng.random() < self._prob_cx:
                c1, c2 = self._problem_adapter.crossover([parent1, parent2])
            else:
                c1, c2 = parent1, parent2
            # Mutation
            if rng.random() < self._prob_mt:
                c1 = self._problem_adapter.mutate(c1)
            children.append(c1)
            if rng.random() < self._prob_mt:
                c2 = self._problem_adapter.mutate(c2)
            children.append(c2)
        return children[:self._pop_size]

# ---------------------------------------
# Extract Objectives
# ---------------------------------------  
    def _extract_objectives_to_minimization(
        self,
        result_doc: dict
    ) -> list[float] | None:
        """
        Extracts objective vector (minimization) from DONE simulation doc.
        """
        obj = result_doc.get("objectives")

        if not isinstance(obj, dict):
            return None

        if not self._objective_keys:
            logger.error("[NSGA-III] objective_keys not configured.")
            return None

        if len(self._objective_keys) != len(self._objective_goals):
            raise ValueError("objective_keys and objective_goals size mismatch")

        vec: list[float] = []

        try:
            for k, s in zip(self._objective_keys, self._objective_goals):
                if k not in obj:
                    raise KeyError(f"Missing objective key: {k}")

                v = float(obj[k])
                if not np.isfinite(v):
                    raise ValueError(f"Non-finite objective {k}={obj[k]}")

                vec.append(s * v)

            return vec

        except Exception as e:
            logger.exception(
                "[NSGA-III] Error extracting objectives. sim_id=%s obj=%s error=%s",
                result_doc.get("_id"),
                obj,
                e,
            )
            return None

# ---------------------------------------
# Finalize Experiment
# ---------------------------------------  
    def _finalize_experiment(self, 
        system_msg: Optional[str] = None, 
        pareto_front: Optional[list[dict]] = None
    ) -> None:
        assert self._exp_id is not None
        if pareto_front is not None:
            logger.info(f"[NSGA-III] Experiment {self._exp_id} completed.")
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now(),
                "system_message": system_msg if system_msg is not None else f"Experiment {self._exp_id} completed.",
                "pareto_front": pareto_front
            })
        else:
            logger.error(f"[NSGA-III] Experiment {self._exp_id} finished.")
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.ERROR,
                "system_message": system_msg if system_msg is not None else f"Experiment {self._exp_id} finished.",
                "end_time": datetime.now()
            })
        # finish watcher
        self.stop()
