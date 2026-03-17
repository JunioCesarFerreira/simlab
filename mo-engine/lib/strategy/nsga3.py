import os
import time
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
from pylib.dto.database import Batch, Simulation, SimulationConfig, Generation
from pylib import plot_network

from lib.util.build_input_sim_cooja import create_files

# NSGA utils
from lib.nsga import fast_nondominated_sort
from lib.nsga import generate_reference_points, niching_selection
from lib.genetic_operators.selection import tournament_selection, compute_individual_ranks
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
        self._sim_rand_seeds: list[int] = [int(x) for x in simulation_config.get("random_seeds", [123456])]
        self._pop_size: int = int(algorithm_config.get("population_size", 20))
        self._max_gen: int = int(algorithm_config.get("number_of_generations", 5))
        self._prob_cx = float(algorithm_config.get("prob_cx", 0.8))
        self._prob_mt = float(algorithm_config.get("prob_mt", 0.2))
    
        ga_random_seed: int = int(algorithm_config.get("random_seed", 42))
        self._ga_rng = random.Random(ga_random_seed)
                        
        self._problem_adapter: ProblemAdapter = build_adapter(
            problem_config, 
            algorithm_config, 
            self._ga_rng
            )
                
        # prepare network metric conversor
        self._metric_conv_config = experiment.get("data_conversion_config", {}) or {}
        
        # prepare objective keys and goals
        obj = params.get("objectives", []) or []
        self._objective_keys: list[str] = [o["metric_name"] for o in obj]
        self._objective_goals: list[int] = [1 if o["goal"]=='min' else -1 for o in obj]
        if len(self._objective_keys) != len(self._objective_goals):
            raise ValueError(
                f"objective_keys ({len(self._objective_keys)}) and "
                f"objective_goals ({len(self._objective_goals)}) length mismatch"
            )
        logger.info(f"Objective keys: {self._objective_keys} with goals: {self._objective_goals}")
            
        # --- nsga3 niching ---
        self._divisions: int = int(algorithm_config.get("divisions", 10))      
        self._ref_points = generate_reference_points(len(self._objective_keys), self._divisions)
        
        # --- loop state ---
        self._exp_id: ObjectId | None = None
        self._gen_index: int = 0
        self._batch_id: ObjectId | None = None
        self._lock = threading.Lock()

        # --- nsga3 workflow ---
        self._current_population: list[Chromosome] = [] # P_t
        self._parents: list[Chromosome] = [] # P_{t-1}
        self._map_genome_sim: dict[Chromosome, list[ObjectId]] = {}
        self._map_genome_objectives: dict[Chromosome, list[float]] = {}
        
        
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
        self._batch_enqueue()

        # Starts Change Stream watcher and polling fallback
        self._start_watcher()
        self._start_batch_poll()

    # EVENT_SIMULATION_DONE implementation
    def event_batch_done(self, result_doc: dict):
        """
        On event Simulation Batch Result Done (called from Change Stream thread).
        Args: result_doc (dict): raw Change Stream event containing fullDocument.
        """
        batch = result_doc.get("fullDocument")
        if not batch:
            return
        with self._lock:
            self._handle_batch_done(ObjectId(batch.get("_id")))

    def _handle_batch_done(self, batch_oid: ObjectId):
        """Core handler — must be called while holding self._lock."""
        if self._stop_flag or self._batch_id is None:
            return

        # Ensures that this result is from the current batch
        if batch_oid != self._batch_id:
            return

        logger.info("EVENT SIMULATION BATCH RESULT DONE batch_id=%s", self._batch_id)
        
        map_sim_metrics = self.mongo.batch_repo.get_simulations_metrics_map(
            batch_id=self._batch_id, 
            metrics=self._objective_keys
            )
        
        n_obj = len(self._objective_keys)
        worst_objectives = [float("inf")] * n_obj  # worst in minimization space for any goal

        for ind in self._current_population:
            if self._map_genome_objectives.get(ind) is not None:
                logger.info("Objectives already calculated for genome %s; skipping.", ind.get_hash())
                continue

            sims = self._map_genome_sim.get(ind, [])
            if not sims:
                logger.warning("No simulations mapped for genome %s; assigning worst objectives.", ind.get_hash())
                self._map_genome_objectives[ind] = worst_objectives
                continue

            accumulated: list[float] = [0.0] * n_obj
            valid_count = 0
            for sim_id in sims:
                sim_metrics = map_sim_metrics.get(str(sim_id))
                if sim_metrics is None:
                    logger.warning("Missing metrics for simulation %s; skipping.", sim_id)
                    continue
                obj_vector = self._extract_objectives_to_minimization(sim_metrics)
                if obj_vector is None:
                    logger.warning("Could not extract objective vector for simulation %s; skipping.", sim_id)
                    continue
                accumulated = [a + v for a, v in zip(accumulated, obj_vector)]
                valid_count += 1

            if valid_count == 0:
                logger.warning(
                    "All %d simulation(s) failed for genome %s; assigning worst objectives.",
                    len(sims), ind.get_hash()
                )
                self._map_genome_objectives[ind] = worst_objectives
            else:
                if valid_count < len(sims):
                    logger.warning(
                        "Genome %s: %d/%d simulation(s) failed; computing average over valid results.",
                        ind.get_hash(), len(sims) - valid_count, len(sims)
                    )
                self._map_genome_objectives[ind] = [a / valid_count for a in accumulated]
                        
        self._evolution()

    # STOP implementation
    def stop(self):
        self._stop_flag = True
        t = self._watch_thread
        if t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._watch_thread = None

# ------------------------------
# Watcher (Change Stream) + Polling fallback
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
                    self.event_batch_done(result_doc)
                except Exception:
                    logger.exception(f"[NSGA-III] Watcher Callback Error.")

            logger.info("[NSGA-III] Starting Batch watcher (DONE).")
            self.mongo.batch_repo.watch_status_done(_callback)

        self._watch_thread = Thread(target=_run, daemon=True, name="nsga3-watcher")
        self._watch_thread.start()

    def _start_batch_poll(self) -> None:
        """
        Polling fallback that checks the current batch status every 3600 s.
        Fires _handle_batch_done if the batch is DONE but the Change Stream
        event was missed (e.g. after a reconnection gap).
        """
        poll_interval = int(os.getenv("BATCH_POLL_INTERVAL", "3600"))  # default: 1 hour

        def _poll():
            while not self._stop_flag:
                time.sleep(poll_interval)
                if self._stop_flag or self._batch_id is None:
                    continue
                try:
                    batch = self.mongo.batch_repo.get(str(self._batch_id))
                    if batch and batch.get("status") == EnumStatus.DONE:
                        logger.warning(
                            "[NSGA-III] Batch %s DONE detected by polling fallback (Change Stream may have missed the event).",
                            self._batch_id
                        )
                        with self._lock:
                            self._handle_batch_done(ObjectId(batch["_id"]))
                except Exception:
                    logger.exception("[NSGA-III] Polling fallback error.")

        Thread(target=_poll, daemon=True, name="nsga3-poll").start()

    def _upload_topology_async(
        self,
        exp_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        config_snapshot: dict,
        sim_ids: list[ObjectId],
    ) -> None:
        """
        Generates and uploads the topology image for one individual in a background
        thread, then back-fills topology_picture_id on each simulation document.
        Running outside the lock keeps _batch_enqueue fast.
        """
        def _do():
            try:
                topo_id = self._plot_topology(exp_oid, gen_index, ind_idx, config_snapshot)
                for sim_id in sim_ids:
                    self.mongo.simulation_repo.update_topology_picture(sim_id, topo_id)
            except Exception:
                logger.exception(
                    "[NSGA-III] Topology upload failed for gen=%s ind=%s", gen_index, ind_idx
                )

        Thread(target=_do, daemon=True, name=f"topo-g{gen_index}-{ind_idx}").start()

# ------------------------------
# Generation / Queuing
# ------------------------------
    def _batch_enqueue(self) -> None:
        assert self._exp_id is not None
        exp_oid = self._exp_id
        gen_index = self._gen_index
        population = self._current_population
        
        batch_simulation_ids: list[ObjectId] = [] # collect simulation ids for this generation batch
        
        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456 # default seed for topology plotting (if no seeds provided)
        
        for i, genome in enumerate(population):
            # convert genome to simulation config
            config = self._convert_genome_to_sim_config(
                genome=genome,
                gen_index=gen_index,
                ind_idx=i,
                seed=first_seed
            )

            # check if genome was already evaluated (duplicate in population or previous generation)
            if genome in self._map_genome_sim:
                logger.info("Genome %s already evaluated; skipping simulation creation.", genome.get_hash())
                continue

            # build and insert simulations per seed (topology_picture_id filled in async)
            simulation_ids_for_genome: list[ObjectId] = []
            for seed in self._sim_rand_seeds:
                config["randomSeed"] = seed
                sim_oid = self._insert_simulation_db(genome, exp_oid, config, None)
                simulation_ids_for_genome.append(sim_oid)
                logger.info("SIM_OID=%s SEED=%s genome=%s", sim_oid, seed, genome.get_hash())

            self._map_genome_sim[genome] = simulation_ids_for_genome
            batch_simulation_ids.extend(simulation_ids_for_genome)

            # upload topology image in the background (non-blocking)
            self._upload_topology_async(exp_oid, gen_index, i, dict(config), simulation_ids_for_genome)
        
        # create batch for this generation
        batch_doc: Batch = {
            "index": gen_index,
            "status": EnumStatus.WAITING,
            "start_time": datetime.now(),
            "end_time": None,
            "simulations_ids": batch_simulation_ids
        }
        self._batch_id: ObjectId = self.mongo.batch_repo.insert(batch_doc)

        # update experiment
        if gen_index == 0:
            self.mongo.experiment_repo.update(str(exp_oid), {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now()
            })
        else:
            self._generation_to_db() # register previous generation with objectives and topology pictures in experiment generations list
            
        self._gen_index += 1 # next generation index
            
        logger.info(f"[NSGA-III] Generation {gen_index} enqueued with {len(population)} Individuals.")
                
    
    def _generation_to_db(self) -> Generation:
        exp_oid = self._exp_id
        gen_idx = self._gen_index

        generation: Generation = {
            "index": gen_idx - 1,
            "population": []
        }

        for genome in self._parents:
            sim_ids = self._map_genome_sim.get(genome, [])

            # Reuse the topology_picture_id already stored in the first simulation document.
            # This avoids a duplicate render+upload that would occur if we called _plot_topology here.
            topology_picture_id = None
            if sim_ids:
                sim_doc = self.mongo.simulation_repo.get(str(sim_ids[0]))
                if sim_doc:
                    topology_picture_id = sim_doc.get("topology_picture_id")

            generation["population"].append({
                "id": genome.get_hash(),
                "chromosome": genome.to_dict(),
                "objectives": self._map_genome_objectives[genome],
                "topology_picture_id": topology_picture_id,
                "simulations_ids": sim_ids,
            })

        self.mongo.experiment_repo.add_generation(exp_oid, generation)


    def _objectives_to_original(self, vec: list[float]) -> dict[str, float]:
        return {
            k: float(v * s)
            for k, v, s in zip(self._objective_keys, vec, self._objective_goals)
        }       


    def _extract_objectives_to_minimization(self, sim_metrics_map: dict[str, float]) -> Optional[list[float]]:
        obj_vector: list[float] = []
        for key, goal in zip(self._objective_keys, self._objective_goals):
            metric_value = sim_metrics_map.get(key)
            if metric_value is None:
                logger.warning(f"Missing metric '{key}' for objectives; cannot compute objective vector.")
                return None
            # Convert to minimization format if needed
            obj_value = metric_value * goal  # if goal is -1 (max), this will invert it to be minimized
            obj_vector.append(obj_value)
        return obj_vector

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
        if self._parents == []:
            # Store P_0 (genomes + objectives) = newly assessed population
            self._parents = self._current_population.copy()

            # Generate P_1 from P_0
            offspring = self._run_genetic_algorithm()
            
            # current population is P_1 and next generation
            self._current_population = offspring
            self._batch_enqueue()
            
            logger.info("[NSGA-III] Enqueued P_{t+1}; waiting results.")
            return
        
        # stop condition?
        if self._gen_index > self._max_gen:
            try:
                self._generation_to_db()
                all_objectives: list[list[float]] = []
                pareto_items: list[dict] = []

                for genome, objetives in self._map_genome_objectives.items():
                    pareto_items.append({
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
        parents_objectives = [self._map_genome_objectives.get(genome) for genome in self._parents]
        current_objectives = [self._map_genome_objectives.get(genome) for genome in self._current_population]
        # union R_t = P_t ∪ P_{t-1}
        p_previous = np.array(parents_objectives, dtype=float)
        p_current = np.array(current_objectives, dtype=float)
        
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
                    partial = niching_selection(front, R_F_list, self._ref_points, remaining, self._ga_rng)
                    selected_idx.extend(partial)
                break

        # build next parents Q=E(R_t,α_t)
        self._parents = [self._current_population[idx - len(p_previous)] 
                         if idx >= len(p_previous) else self._parents[idx] 
                         for idx in selected_idx]
        
        # ------- PHASE Q: parents done -> generate offspring and enqueue -------
        # produce P_{t+1} from GA(Q,H) (variation on parents)            
        offspring = self._run_genetic_algorithm()

        # enqueue P_{t+1} as next "generation" to be evaluated
        self._current_population = offspring
        self._batch_enqueue()        
        
        logger.info("[NSGA-III] Offspring enqueued; waiting for P_t results to perform environmental selection.")

        return
    
# ---------------------------------------
# Run Genetic Algorithm
# ---------------------------------------   
    def _run_genetic_algorithm(self) -> list[list[float]]:
        parents = self._parents
        worst = [float("inf")] * len(self._objective_keys)
        parents_objectives = [self._map_genome_objectives.get(genome, worst) for genome in parents]
        
        logger.debug("objectives: %s", parents_objectives)

        children: list[Chromosome] = []
        seen: set[Chromosome] = set()

        fronts: list[list[int]] = fast_nondominated_sort(parents_objectives)

        logger.debug("fronts: %s", fronts)
        
        individual_ranks: dict[int, int] = compute_individual_ranks(fronts)
                    
        max_attempts = self._pop_size * 10
        attempts = 0
        
        while len(children) < self._pop_size and attempts < max_attempts:
            attempts += 1
            # Selection
            parent1: Chromosome = tournament_selection(parents, individual_ranks, self._ga_rng)
            parent2: Chromosome = tournament_selection(parents, individual_ranks, self._ga_rng)
            # Crossover 
            if self._ga_rng.random() < self._prob_cx:
                c1, c2 = self._problem_adapter.crossover([parent1, parent2])
            else:
                c1, c2 = parent1, parent2
            # Mutation
            if self._ga_rng.random() < self._prob_mt:
                c1 = self._problem_adapter.mutate(c1)
                
            if c1 not in seen:
                children.append(c1)
                seen.add(c1)

            if len(children) >= self._pop_size:
                break

            if self._ga_rng.random() < self._prob_mt:
                c2 = self._problem_adapter.mutate(c2)

            if c2 not in seen:
                children.append(c2)
                seen.add(c2)
        return children[:self._pop_size]


# ------------------------------
# Encodes genome into simulation elements
# ------------------------------
    def _convert_genome_to_sim_config(self, 
            genome: Chromosome,
            gen_index: int,
            ind_idx: int,
            seed: int
            )-> SimulationConfig:
        
        # Encodes genome into simulation elements
        simulationElements = self._problem_adapter.encode_simulation_input(genome)
        
        config: SimulationConfig = {
                "name": f"nsga3-g{gen_index}-{ind_idx}-{seed}",
                "duration": self._sim_duration,
                "randomSeed": seed,
                "radiusOfReach": self._problem_adapter.radius_of_reach,
                "radiusOfInter": self._problem_adapter.radius_of_inter,
                "region": self._problem_adapter.bounds,
                "simulationElements": simulationElements
            }
        return config


    def _insert_simulation_db(self, 
            genome: Chromosome,
            exp_oid: ObjectId,
            config: SimulationConfig,
            topology_picture_id: ObjectId
            )-> ObjectId:
        files_ids = create_files(config, self.mongo.fs_handler)                     
        _, src_id = genome.get_source_by_mac_protocol(self.source_repository_options)
             
        sim_doc: Simulation = {
            "experiment_id": exp_oid,
            "individual_id": genome.get_hash(),
            "status": EnumStatus.WAITING,
            "random_seed": config.get("randomSeed", 0),
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
        return self.mongo.simulation_repo.insert(sim_doc)


    def _plot_topology(self,
            exp_oid: ObjectId,
            gen_index: int,
            ind_idx: int,
            config: SimulationConfig
            )->ObjectId:
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        image_tmp_path = tmp_dir / f"topology-{exp_oid}-{gen_index}-{ind_idx}.png"
        
        plot_network.plot_network_save_from_sim(str(image_tmp_path), config)
        
        topology_picture_id = self.mongo.fs_handler.upload_file(
            str(image_tmp_path),
            f"topology-{exp_oid}-{gen_index}-{ind_idx}"
        )
        
        if os.path.exists(image_tmp_path):
            os.remove(image_tmp_path)
            
        return topology_picture_id

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
