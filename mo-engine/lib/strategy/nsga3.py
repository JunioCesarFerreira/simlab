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
        
        # Simulation and algorithm parameters
        self._sim_duration: int = int(simulation_config.get("duration", 120))
        self._pop_size: int = int(algorithm_config.get("population_size", 20))
        self._max_gen: int = int(algorithm_config.get("number_of_generations", 5))
                        
        self._prob_cx = float(algorithm_config.get("prob_cx", 0.8))
        self._prob_mt = float(algorithm_config.get("prob_mt", 0.2))
        
        params_of_ga_operators: dict[str, float] = {}
        if "eta_cx" in algorithm_config:
            params_of_ga_operators["eta_cx"] = float(algorithm_config.get("eta_cx"))
        if "eta_mt" in algorithm_config:
            params_of_ga_operators["eta_mt"] = float(algorithm_config.get("eta_mt"))
        if "per_gene_prob" in algorithm_config:
            params_of_ga_operators["per_gene_prob"] = float(algorithm_config.get("per_gene_prob")) 
                
        self._problem_adapter: ProblemAdapter = build_adapter(problem_config, params_of_ga_operators)
                
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
            
        # nsga3 niching
        self._divisions: int = int(algorithm_config.get("divisions", 10))      
        self._ref_points = generate_reference_points(len(self._objective_keys), self._divisions)
        
        # --- loop state ---
        self._exp_id: ObjectId | None = None
        self._gen_index: int = 0
        self._gen_id: ObjectId | None = None

        # current population as a list of individuals (each individual is a vector [x0,y0,x1,y1,...])
        self._current_population: list[Chromosome] = []
        # maps simulation_id(str) -> index of the individual in the population by generation
        self._current_sim_oid_to_idx: dict[str, int] = {}
        # results collected from current generation: idx -> list[float] objectives (minimization)
        self._current_idx_to_objectives: dict[int, list[float]] = {}
        
        self._parents = PopulationSnapshot()                   


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

        # Enqueue Simulations to first (compute ε_0)
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
        idx = self._current_sim_oid_to_idx.get(sim_id)
        if idx is None:
            return

        obj = self._extract_objectives_to_minimization(sim)
        if obj is None:
            logger.info(f"objectives not found in {sim_id}")
            return

        self._current_idx_to_objectives[idx] = obj

        # check generation is complete
        logger.info(f"{len(self._current_idx_to_objectives)} of {len(self._current_population)}")
        if len(self._current_idx_to_objectives) >= len(self._current_population):
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

        simulation_ids: list[ObjectId] = []
        self._current_sim_oid_to_idx.clear()
        self._current_idx_to_objectives.clear()

        for i, genome in enumerate(population):
            sim_oid = self._build_simulation(genome, exp_oid, gen_oid, gen_index, i)
            logger.info(f"sim_oid={sim_oid}")
            simulation_ids.append(sim_oid)
            self._current_sim_oid_to_idx[str(sim_oid)] = i

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
        """
        Two-phase (μ+λ) loop:
        Phase P: parents (P_t) finished -> produce offspring Q_t and enqueue -> return.
        Phase Q: offspring (Q_t) finished -> environmental selection on R_t = P_t U Q_t -> spawn P_{t+1}.
        """
        assert self._gen_id is not None
        logger.info(f"[NSGA-III] Generation {self._gen_index} completed.")

        # close current generation
        self.mongo.generation_repo.mark_done(self._gen_id)

        # build objective matrix for the just-finished batch (aligned with self.current_population)
        try:
            indices = list(range(len(self._current_population)))
            objectives = [self._current_idx_to_objectives[i] for i in indices]
        except Exception:
            logger.exception("[NSGA-III] Missing objectives for some individuals; aborting.")
            self._finalize_experiment()
            return
        
        # First PHASE P
        if self._parents.dont_have_objectives():
            # Store P_t (genomes + objectives) = newly assessed population
            self._parents.set(
                genomes= self._current_population,
                sim_oid_to_idx= self._current_sim_oid_to_idx,
                objectives= objectives
            )

            # Gera Q_t ranqueando P_t e enfileira para avaliação
            offspring = self._run_genetic_algorithm(self._parents.get_objectives())
            
            self._gen_index += 1
            self._current_population = offspring
            self._generation_enqueue()
            
            logger.info("[NSGA-III] Enqueued Q_t; waiting results.")
            return
        
        # ---------------- PHASE Q: offspring done -> environmental selection on union ----------------
        # union R_t = P_t ∪ Q_t
        P_F = np.array(self._parents.get_objectives(), dtype=float)
        Q_F = np.array(objectives, dtype=float)
        if Q_F.ndim != 2 or Q_F.shape[0] == 0:
            logger.error("[NSGA-III] Invalid objective matrix; aborting.")
            self._finalize_experiment()
            return
        
        q_snapshot = PopulationSnapshot()
        q_snapshot.set(
            genomes= self._current_population,
            sim_oid_to_idx= self._current_sim_oid_to_idx,
            objectives= objectives
        )

        # concatenate
        R_F_list = [list(row) for row in P_F.tolist()] + [list(row) for row in Q_F.tolist()]
            
        # fast non-dominated sort on union
        fronts = fast_nondominated_sort(R_F_list)
        if not fronts:
            logger.info("[NSGA-III] No fronts on union; aborting.")
            self._finalize_experiment()
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

        # build next parents P_{t+1}
        self._parents = select_next_population(
            selected_idxs= selected_idx,
            pop_size= self._pop_size,
            P= self._parents,
            Q= q_snapshot
        )

        # stop condition?
        if self._gen_index >= self._max_gen:
            try:
                parents_objectives = self._parents.get_objectives()
                fronts_final = fast_nondominated_sort(parents_objectives)

                pareto_front: list[dict] = []

                if fronts_final:
                    for idx in fronts_final[0]:
                        pareto_front.append({
                            "simulation_id": self._parents.simulation(idx),
                            "chromosome": self._parents.genome(idx).to_dict(),
                            "objectives": self._objectives_to_original(
                                parents_objectives[idx]
                            ),
                        })

            except Exception:
                logger.exception("[NSGA-III] Could not compute final Pareto front.")
                pareto_front = []

            self._finalize_experiment(pareto_front)
            return
        
        # ---------------- PHASE P: parents done -> generate offspring and enqueue ----------------
        # produce Q_t from P_t (variation on parents)            
        offspring = self._run_genetic_algorithm(self._parents.get_objectives())

        # enqueue Q_t as next "generation" to be evaluated
        # (note: gen_index here is just a sequence counter of batches evaluated)
        self._gen_index += 1
        self._current_population = offspring
        self._generation_enqueue()
        
        
        logger.info("[NSGA-III] Offspring enqueued; waiting for Q_t results to perform environmental selection.")

        return

# ---------------------------------------
# Run Genetic Algorithm
# ---------------------------------------   
    def _run_genetic_algorithm(self, objectives: list[list[float]]) -> list[list[float]]:
        rng = random.Random()
        parents = self._parents.get_genomes()
        children: list[Chromosome] = []        
        fronts: list[list[int]] = fast_nondominated_sort(objectives)
        individual_ranks: dict[int, int] = self._compute_individual_ranks(fronts)
        while len(children) < self._pop_size:
            # Selection
            parent1: Chromosome = self._tournament_selection(parents, individual_ranks)
            parent2: Chromosome = self._tournament_selection(parents, individual_ranks)
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


    def _tournament_selection(self,
        population: list[Chromosome], 
        individual_ranks: dict[int, int]
        ) -> Chromosome:
            i1, i2 = random.sample(range(len(population)), 2)
            rank1: int = individual_ranks[i1]
            rank2: int = individual_ranks[i2]
            if rank1 < rank2:
                return population[i1]
            elif rank2 < rank1:
                return population[i2]
            else:
                return population[random.choice([i1, i2])]


    def _compute_individual_ranks(self, fronts: list[list[int]]) -> dict[int, int]:
        individual_ranks: dict[int, int] = {}
        for rank, front in enumerate(fronts):
            for idx in front:
                individual_ranks[idx] = rank
        return individual_ranks


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


    def _finalize_experiment(self, pareto_front: Optional[list[dict]] = None):
        assert self._exp_id is not None
        if pareto_front is not None:
            logger.info(f"[NSGA-III] Experiment {self._exp_id} completed.")
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now(),
                "pareto_front": pareto_front
            })
        else:
            logger.error(f"[NSGA-III] Experiment {self._exp_id} finished.")
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.ERROR,
                "end_time": datetime.now()
            })
        # finish watcher
        self.stop()
