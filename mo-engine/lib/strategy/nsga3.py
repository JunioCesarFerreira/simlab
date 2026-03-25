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

from pylib.db import EnumStatus
from pylib.db.models import Generation, Individual, Simulation
from pylib.config.simulator import SimulationConfig
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
    - Upon receiving the results (DONE status on the Generation), calculates the objectives,
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

        self._metric_conv_config = experiment.get("data_conversion_config", {}) or {}

        obj = params.get("objectives", []) or []
        self._objective_keys: list[str] = [o["metric_name"] for o in obj]
        self._objective_goals: list[int] = [1 if o["goal"] == 'min' else -1 for o in obj]
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
        self._generation_id: ObjectId | None = None
        self._lock = threading.Lock()

        # --- nsga3 workflow ---
        self._current_population: list[Chromosome] = []   # P_t
        self._parents: list[Chromosome] = []              # P_{t-1}
        self._inserted_genomes: set[str] = set()          # hashes already inserted to DB
        self._map_genome_objectives: dict[Chromosome, list[float]] = {}


# ------------------------------
# Interface EngineStrategy
# ------------------------------
    def start(self):
        """
        Initializes the population and creates Generation 0 with `population_size` simulations.
        """
        self._exp_id = ObjectId(self.experiment["_id"]) if isinstance(self.experiment.get("_id"), (str, bytes)) else self.experiment.get("_id")
        if not isinstance(self._exp_id, ObjectId):
            self._exp_id = ObjectId(str(self.experiment.get("_id")))
        self._gen_index = 0

        self._current_population = self._problem_adapter.random_individual_generator(self._pop_size)

        self._generation_enqueue()
        self._start_watcher()
        self._start_generation_poll()

    def event_batch_done(self, result_doc: dict):
        """
        On event Generation terminal (called from Change Stream thread).
        """
        gen = result_doc.get("fullDocument")
        if not gen:
            return
        with self._lock:
            self._handle_generation_done(ObjectId(gen.get("_id")))

    def _handle_generation_done(self, gen_oid: ObjectId):
        """Core handler — must be called while holding self._lock."""
        if self._stop_flag or self._generation_id is None:
            return

        if gen_oid != self._generation_id:
            return

        logger.info("EVENT GENERATION TERMINAL gen_id=%s", self._generation_id)

        map_ind_metrics = self.mongo.generation_repo.get_simulations_metrics_by_individual(
            generation_id=self._generation_id,
            metrics=self._objective_keys
        )

        n_obj = len(self._objective_keys)
        worst_objectives = [float("inf")] * n_obj

        for ind in self._current_population:
            if self._map_genome_objectives.get(ind) is not None:
                logger.info("Objectives already calculated for genome %s; skipping.", ind.get_hash())
                continue

            ind_metrics = map_ind_metrics.get(ind.get_hash())
            if ind_metrics is None:
                logger.warning("No metrics for genome %s; assigning worst objectives.", ind.get_hash())
                self._map_genome_objectives[ind] = worst_objectives
                continue

            obj_vector = self._extract_objectives_to_minimization(ind_metrics)
            if obj_vector is None:
                logger.warning("Could not extract objective vector for genome %s; assigning worst.", ind.get_hash())
                self._map_genome_objectives[ind] = worst_objectives
            else:
                self._map_genome_objectives[ind] = obj_vector

        # Persist objectives to individual documents before evolving
        self._update_individual_objectives()

        self._evolution()

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
        self._stop_flag = False
        if self._watch_thread and self._watch_thread.is_alive():
            self._stop_flag = True
            self._watch_thread.join(timeout=1.0)
            self._stop_flag = False

        def _run():
            def _callback(result_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_batch_done(result_doc)
                except Exception:
                    logger.exception("[NSGA-III] Watcher Callback Error.")

            logger.info("[NSGA-III] Starting Generation watcher (DONE or ERROR).")
            self.mongo.generation_repo.watch_status_terminal(_callback)

        self._watch_thread = Thread(target=_run, daemon=True, name="nsga3-watcher")
        self._watch_thread.start()

    def _start_generation_poll(self) -> None:
        """
        Polling fallback that checks the current generation status every poll_interval seconds.
        Fires _handle_generation_done if the generation is terminal but the Change Stream
        event was missed (e.g. after a reconnection gap).
        """
        poll_interval = int(os.getenv("BATCH_POLL_INTERVAL", "3600"))

        def _poll():
            while not self._stop_flag:
                time.sleep(poll_interval)
                if self._stop_flag or self._generation_id is None:
                    continue
                try:
                    gen = self.mongo.generation_repo.get(str(self._generation_id))
                    if gen and gen.get("status") in (EnumStatus.DONE, EnumStatus.ERROR):
                        logger.warning(
                            "[NSGA-III] Generation %s %s detected by polling fallback.",
                            self._generation_id, gen.get("status")
                        )
                        with self._lock:
                            self._handle_generation_done(ObjectId(gen["_id"]))
                except Exception:
                    logger.exception("[NSGA-III] Polling fallback error.")

        Thread(target=_poll, daemon=True, name="nsga3-poll").start()

    def _upload_topology_async(
        self,
        exp_oid: ObjectId,
        gen_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        individual_id: str,
        config_snapshot: dict,
    ) -> None:
        """
        Generates and uploads the topology image for one individual in a background thread,
        then updates topology_picture_id on the Individual document.
        """
        def _do():
            try:
                topo_id = self._plot_topology(exp_oid, gen_index, ind_idx, config_snapshot)
                self.mongo.individual_repo.update_topology_picture(individual_id, gen_oid, topo_id)
            except Exception:
                logger.exception(
                    "[NSGA-III] Topology upload failed for gen=%s ind=%s", gen_index, ind_idx
                )

        Thread(target=_do, daemon=True, name=f"topo-g{gen_index}-{ind_idx}").start()


# ------------------------------
# Generation / Queuing
# ------------------------------
    def _generation_enqueue(self) -> None:
        assert self._exp_id is not None
        exp_oid = self._exp_id
        gen_index = self._gen_index
        population = self._current_population

        # Pre-generate generation ObjectId so simulations are inserted with generation_id set,
        # avoiding the race where master-node picks up a simulation before the generation exists.
        gen_oid = ObjectId()

        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456

        for i, genome in enumerate(population):
            genome_hash = genome.get_hash()

            # Skip genomes already inserted to DB in this session
            if genome_hash in self._inserted_genomes:
                logger.info("Genome %s already inserted; skipping.", genome_hash)
                continue

            config = self._convert_genome_to_sim_config(
                genome=genome,
                gen_index=gen_index,
                ind_idx=i,
                seed=first_seed
            )

            # Insert Individual document (objectives filled in after evaluation)
            ind_doc: Individual = {
                "experiment_id": exp_oid,
                "generation_id": gen_oid,
                "individual_id": genome_hash,
                "chromosome": genome.to_dict(),
                "objectives": [],
                "topology_picture_id": None,
            }
            self.mongo.individual_repo.insert(ind_doc)
            self._inserted_genomes.add(genome_hash)

            # Insert one simulation per seed
            for seed in self._sim_rand_seeds:
                config["randomSeed"] = seed
                self._insert_simulation_db(genome_hash, exp_oid, gen_oid, config)
                logger.info("SIM inserted SEED=%s genome=%s", seed, genome_hash)

            # Upload topology image in the background (non-blocking)
            self._upload_topology_async(exp_oid, gen_oid, gen_index, i, genome_hash, dict(config))

        # Create Generation document (WAITING → triggers master-node watcher)
        gen_doc: Generation = {
            "_id": gen_oid,
            "experiment_id": exp_oid,
            "index": gen_index,
            "status": EnumStatus.WAITING,
            "start_time": datetime.now(),
            "end_time": None,
        }
        self._generation_id = self.mongo.generation_repo.insert(gen_doc)

        if gen_index == 0:
            self.mongo.experiment_repo.update(str(exp_oid), {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now()
            })

        self._gen_index += 1

        logger.info("[NSGA-III] Generation %d enqueued with %d individuals.", gen_index, len(population))

    def _update_individual_objectives(self) -> None:
        """
        Persists computed objectives to Individual documents in the DB.
        Called right after objectives are computed, before evolution.
        """
        if self._generation_id is None:
            return
        for genome in self._current_population:
            objectives = self._map_genome_objectives.get(genome)
            if objectives is not None:
                self.mongo.individual_repo.update_objectives(
                    genome.get_hash(),
                    self._generation_id,
                    objectives
                )

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
                logger.warning(f"Missing metric '{key}'; cannot compute objective vector.")
                return None
            obj_vector.append(metric_value * goal)
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
            self._parents = self._current_population.copy()
            offspring = self._run_genetic_algorithm()
            self._current_population = offspring
            self._generation_enqueue()
            logger.info("[NSGA-III] Enqueued P_{t+1}; waiting results.")
            return

        # Stop condition?
        if self._gen_index > self._max_gen:
            try:
                all_objectives: list[list[float]] = []
                pareto_items: list[dict] = []

                for genome, objectives in self._map_genome_objectives.items():
                    pareto_items.append({
                        "chromosome": genome.to_dict(),
                        "objectives": self._objectives_to_original(objectives),
                    })
                    all_objectives.append(objectives)

                pareto_fronts = fast_nondominated_sort(all_objectives)
                first_pareto_front = [pareto_items[idx] for idx in pareto_fronts[0]]

            except Exception:
                logger.exception("[NSGA-III] Could not compute final Pareto front.")
                first_pareto_front = []

            self._finalize_experiment(pareto_front=first_pareto_front)
            return

        # ------- PHASE P_{t+1}: environmental selection on union R_t = P_t ∪ P_{t-1} -------
        parents_objectives = [self._map_genome_objectives.get(genome) for genome in self._parents]
        current_objectives = [self._map_genome_objectives.get(genome) for genome in self._current_population]

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

        R_F_list = [list(row) for row in p_previous.tolist()] + [list(row) for row in p_current.tolist()]

        fronts = fast_nondominated_sort(R_F_list)
        if not fronts:
            error_msg = "No fronts on union; aborting."
            logger.exception(f"[NSGA-III] {error_msg}")
            self._finalize_experiment(system_msg=error_msg)
            return

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

        self._parents = [self._current_population[idx - len(p_previous)]
                         if idx >= len(p_previous) else self._parents[idx]
                         for idx in selected_idx]

        offspring = self._run_genetic_algorithm()
        self._current_population = offspring
        self._generation_enqueue()

        logger.info("[NSGA-III] Offspring enqueued; waiting for P_t results.")
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
        individual_ranks: dict[int, int] = compute_individual_ranks(fronts)

        max_attempts = self._pop_size * 10
        attempts = 0

        while len(children) < self._pop_size and attempts < max_attempts:
            attempts += 1
            parent1: Chromosome = tournament_selection(parents, individual_ranks, self._ga_rng)
            parent2: Chromosome = tournament_selection(parents, individual_ranks, self._ga_rng)
            if self._ga_rng.random() < self._prob_cx:
                c1, c2 = self._problem_adapter.crossover([parent1, parent2])
            else:
                c1, c2 = parent1, parent2
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
            ) -> SimulationConfig:
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
            genome_hash: str,
            exp_oid: ObjectId,
            gen_oid: ObjectId,
            config: SimulationConfig,
            ) -> ObjectId:
        files_ids = create_files(config, self.mongo.fs_handler)
        # Resolve source repository for this genome
        _, src_id = self._genome_hash_to_source(genome_hash)

        sim_doc: Simulation = {
            "experiment_id": exp_oid,
            "generation_id": gen_oid,
            "individual_id": genome_hash,
            "status": EnumStatus.WAITING,
            "random_seed": config.get("randomSeed", 0),
            "start_time": None,
            "end_time": None,
            "parameters": config,
            "pos_file_id": files_ids.get("pos_file_id", ""),
            "csc_file_id": files_ids.get("csc_file_id", ""),
            "source_repository_id": src_id,
            "log_cooja_id": "",
            "runtime_log_id": "",
            "csv_log_id": "",
            "network_metrics": {},
        }
        return self.mongo.simulation_repo.insert(sim_doc)

    def _genome_hash_to_source(self, genome_hash: str):
        """
        Resolves source_repository_id for a given genome hash.
        Finds the genome in current or parent population, then calls get_source_by_mac_protocol.
        """
        for genome in list(self._current_population) + list(self._parents):
            if genome.get_hash() == genome_hash:
                return genome.get_source_by_mac_protocol(self.source_repository_options)
        raise ValueError(f"Genome hash {genome_hash} not found in populations")

    def _plot_topology(self,
            exp_oid: ObjectId,
            gen_index: int,
            ind_idx: int,
            config: SimulationConfig
            ) -> ObjectId:
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
            logger.error(f"[NSGA-III] Experiment {self._exp_id} finished with error.")
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.ERROR,
                "system_message": system_msg if system_msg is not None else f"Experiment {self._exp_id} finished.",
                "end_time": datetime.now()
            })
        self.stop()
