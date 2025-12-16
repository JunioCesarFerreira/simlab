import os
import random
import logging
import threading
from threading import Thread
from datetime import datetime
from pathlib import Path
from bson import ObjectId
from dataclasses import dataclass
from typing import Optional
import numpy as np

from strategy.base import EngineStrategy
from pylib.mongo_db import EnumStatus
from pylib.dto.experiment import Simulation, SimulationConfig, Generation
from pylib import plot_network

from lib.random_network_methods import network_gen
from lib.build_input_sim_cooja import create_files

# NSGA utils
from .util.nsga import fast_nondominated_sort
from .util.nsga import generate_reference_points, niching_selection
# Problem Adapter
from .problem.adapter import ProblemAdapter, Chromosome
from .problem.resolve import build_adapter

Objectives = list[float]

logger = logging.getLogger(__name__)


@dataclass
class NSGA3Parameters:
    """
    Container for NSGA-III settings.

    Note:
    - reference_points can be precomputed and stored in the experiment document,
      or derived from (M, p) at runtime.
    """
    pop_size: int
    max_generations: int
    reference_points: list[list[float]]


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
        self.population_size: int = int(params.get("population_size", 20))
        self.max_generations: int = int(params.get("number_of_generations", 5))
        self.sim_duration: int = int(params.get("duration", 120))
                
        # nsga3 niching
        self.divisions: int = int(params.get("divisions", 10))      
        self.ref_points = generate_reference_points(len(self.objective_keys), self.divisions)
        
        self.prob_cx = float(params.get("prob_cx", 0.8))
        self.prob_mt = float(params.get("prob_mt", 0.2))
        
        ga_params: dict[str, float] = {}
        if "eta_cx" in params:
            ga_params["eta_cx"] = float(params.get("eta_cx"))
        if "eta_mt" in params:
            ga_params["eta_mt"] = float(params.get("eta_mt"))
        if "per_gene_prob" in params:
            ga_params["per_gene_prob"] = float(params.get("per_gene_prob")) 
        
        problem = params.get("problem", {}) or {}
        
        self.problem_adapter: ProblemAdapter = build_adapter(problem, ga_params)
        
        # --- loop state ---
        self._exp_id: ObjectId | None = None
        self._gen_index: int = 0
        self._gen_id: ObjectId | None = None

        # current population as a list of individuals (each individual is a vector [x0,y0,x1,y1,...])
        self.current_population: list[Chromosome] = []
        # maps simulation_id(str) -> index of the individual in the population
        self.sim_id_to_index: dict[str, int] = {}
        # results collected from current generation: idx -> list[float] objectives (minimization)
        self.objectives_buffer: dict[int, Objectives] = {}
        # dictionary for reuse evaluations values
        self._obj_by_sim_id: dict[str, Objectives] = {}
        
        # prepare objective keys
        cfg = experiment.get("transform_config", {}) or {}
        obj = cfg.get("objectives", []) or []
        self.objective_keys = [o["name"] for o in obj if "name" in o]
        
        self._awaiting_offspring: bool = False  # False => waiting P; True => waiting Q
        self._parents_population: list[Chromosome] = []   # wait P_t (genomas)
        self._parents_objectives: list[Objectives] = []   # wait F(P_t)
               


# ------------------------------
# API do EngineStrategy
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

        # Initial Population
        self.current_population = [self.problem_adapter.random_individual_generator(self.population_size)]

        # Enqueue Simulations to first Generation
        self._generation_enqueue(self.current_population, self._gen_index)

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
        idx = self.sim_id_to_index.get(sim_id)
        if idx is None:
            return

        obj = self._extract_objectives(sim)
        if obj is None:
            logger.info(f"objectives not found in {sim_id}")
            return

        self.objectives_buffer[idx] = obj
        self._obj_by_sim_id[sim_id] = obj

        # check generation is complete
        logger.info(f"{len(self.objectives_buffer)} >= {len(self.current_population)}")
        if len(self.objectives_buffer) >= len(self.current_population):
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
                except Exception as e:
                    logger.info(f"[NSGA-III] Watcher Callback Error: {e}")

            logger.info("[NSGA-III] Starting Simulations watcher (DONE).")
            self.mongo.simulation_repo.watch_status_done(_callback)

        self._watch_thread = Thread(target=_run, daemon=True)
        self._watch_thread.start()

# ------------------------------
# Generation / Queuing
# ------------------------------
    def _generation_enqueue(self, population: list[Chromosome], gen_index: int)->ObjectId:
        """
        Cria a `Generation` e insere `population_size` simulações no MongoDB.
        """
        assert self._exp_id is not None
        exp_oid = self._exp_id

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

        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        simulation_ids: list[ObjectId] = []
        self.sim_id_to_index.clear()
        self.objectives_buffer.clear()

        for i, genome in enumerate(population):
            config: SimulationConfig = {
                "name": f"nsga3-g{gen_index}-{i}",
                "duration": self.duration,
                "radiusOfReach": self.problem_adapter.radius_of_reach,
                "radiusOfInter": self.problem_adapter.radius_of_inter,
                "region": self.problem_adapter.bounds,
                "simulationElements": self.problem_adapter.encode_simulation_input(genome)
            }

            files_ids = create_files(config, self.mongo.fs_handler)
            image_tmp_path = tmp_dir / f"topology-{exp_oid}-{gen_oid}-{i}.png"
            plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
            topology_picture_id = self.mongo.fs_handler.upload_file(
                str(image_tmp_path),
                f"topology-{exp_oid}-{gen_oid}-{i}"
            )
            if os.path.exists(image_tmp_path):
                os.remove(image_tmp_path)

            sim_doc: Simulation = {
                "id": i,
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
            logger.info(f"sim_oid={sim_oid}")
            simulation_ids.append(sim_oid)
            self.sim_id_to_index[str(sim_oid)] = i

        # update generation
        self.mongo.generation_repo.update(gen_oid, {
            "simulations_ids": [str(_id) for _id in simulation_ids],
            "status": EnumStatus.WAITING
        })
        self.mongo.generation_repo.mark_waiting(gen_oid)

        if self._gen_index == 1:
            self.mongo.experiment_repo.update(str(exp_oid), {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now(),
                "generations_ids": [str(gen_oid)]
            })

        logger.info(f"[NSGA-III] Generation {gen_index} enqueued with {len(population)} Simulations.")
        
        self.mongo.experiment_repo.add_generation(exp_oid, gen_oid)
        
        return gen_oid

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
            indices = list(range(len(self.current_population)))
            objectives = [self.objectives_buffer[i] for i in indices]
        except Exception:
            logger.error("[NSGA-III] Missing objectives for some individuals; aborting.")
            self._finalize_experiment()
            return
        
        # First PHASE P
        if not self._parents_objectives:
            # Armazena P_t (genomas + objetivos) = população recém avaliada
            self._parents_population  = [ind[:] for ind in self.current_population]
            self._parents_objectives  = [row[:] for row in objectives]

            # Gera Q_t ranqueando P_t e enfileira para avaliação
            offspring = self._run_genetic_algorithm(self._parents_objectives)
            self._gen_index += 1
            self._generation_enqueue(offspring, self._gen_index)
            self.current_population = offspring
            logger.info("[NSGA-III] Enqueued Q_t; waiting results.")
            return
        
        # ---------------- PHASE Q: offspring done -> environmental selection on union ----------------
        # union R_t = P_t ∪ Q_t
        P_genomes = self._parents_population
        P_F = np.array(self._parents_objectives, dtype=float)
        Q_genomes = self.current_population
        Q_F = np.array(objectives, dtype=float)
        if Q_F.ndim != 2 or Q_F.shape[0] == 0:
            logger.error("[NSGA-III] Invalid objective matrix; aborting.")
            self._finalize_experiment()
            return

        # concatenate
        R_genomes = P_genomes + Q_genomes
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
            if len(selected_idx) + len(front) <= self.population_size:
                selected_idx.extend(front)
            else:
                remaining = self.population_size - len(selected_idx)
                if remaining > 0:
                    partial = niching_selection(front, R_F_list, self.ref_points, remaining)
                    selected_idx.extend(partial)
                break

        next_population = [R_genomes[i] for i in selected_idx[:self.population_size]]
        next_objectives = [R_F_list[i] for i in selected_idx[:self.population_size]]

        # stop condition?
        if self._gen_index >= self.max_generations:
            final_objs = []
            try:
                # objetivos de R_t selecionados
                final_objs = [R_F_list[i] for i in selected_idx[:self.population_size]]
                fronts_final = fast_nondominated_sort(final_objs)
                pareto_front = [tuple(final_objs[i]) for i in fronts_final[0]] if fronts_final else []
                pareto_front.sort()
            except Exception as e:
                logger.info(f"[NSGA-III] Could not compute final Pareto: {e}")
                pareto_front = []

            self._finalize_experiment(pareto_front)
            return

        self._parents_population = [ind[:] for ind in next_population]
        self._parents_objectives = [list(row) for row in next_objectives]
        
        # ---------------- PHASE P: parents done -> generate offspring and enqueue ----------------
        # produce Q_t from P_t (variation on parents)            
        offspring = self._run_genetic_algorithm(self._parents_objectives)

        # enqueue Q_t as next "generation" to be evaluated
        # (note: gen_index here is just a sequence counter of batches evaluated)
        self._gen_index += 1
        self._generation_enqueue(offspring, self._gen_index)
        
        self.current_population = offspring
        
        logger.info("[NSGA-III] Offspring enqueued; waiting for Q_t results to perform environmental selection.")

        return

# ---------------------------------------
# Run Genetic Algorithm
# ---------------------------------------   
    def _run_genetic_algorithm(self, objectives: list[list[float]]) -> list[list[float]]:
        rng = random.Random()
        parents = self._parents_population
        children: list[Chromosome] = []        
        fronts: list[list[int]] = fast_nondominated_sort(objectives)
        individual_ranks: dict[int, int] = self._compute_individual_ranks(fronts)
        while len(children) < self.population_size:
            # Selection
            parent1: Chromosome = self._tournament_selection(parents, individual_ranks)
            parent2: Chromosome = self._tournament_selection(parents, individual_ranks)
            # Crossover 
            if rng.random() < self.prob_cx:
                c1, c2 = self.problem_adapter.crossover(parent1, parent2)
            else:
                c1, c2 = parent1, parent2
            # Mutation
            if rng.random() < self.prob_mt:
                c1 = self.problem_adapter.mutate(c1)
            children.append(c1)
            if rng.random() < self.prob_mt:
                c2 = self.problem_adapter.mutate(c2)
            children.append(c2)
        return children[:self.population_size]


    def _tournament_selection(
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


    def _extract_objectives(self, result_doc: dict) -> Objectives | None:
        """
        Extracts objective vector (minimization) from DONE simulation doc.
        Priority:
        1) 'objectives' as list[float]
        2) 'objectives' as dict -> follow self.objective_keys
        3) 'metrics'    as dict -> follow self.objective_keys
        """
        obj = result_doc.get("objectives")
        # list already in order
        if isinstance(obj, (list, tuple)) and all(isinstance(v, (int, float)) for v in obj):
            return [float(v) for v in obj]

        # try dict by keys (from transform_config)
        if isinstance(obj, dict) and self.objective_keys:
            try:
                return [float(obj[k]) for k in self.objective_keys]
            except Exception:
                pass

        metrics = result_doc.get("metrics") or {}
        if isinstance(metrics, dict) and self.objective_keys:
            try:
                return [float(metrics[k]) for k in self.objective_keys]
            except Exception:
                pass

        logger.info("[NSGA-III] Warning: objectives not found. Set 'objective_keys' via TransformConfig.")
        return None


    def _finalize_experiment(self, pareto_front: Optional[list[tuple[float, ...]]] = None):
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
