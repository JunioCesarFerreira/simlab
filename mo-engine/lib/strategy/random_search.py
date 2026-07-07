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

from .base import EngineStrategy
from .simulation_seeds import resolve_simulation_seeds

from pylib.db import EnumStatus
from pylib.db.models import Generation, Individual, Simulation
from pylib.config.simulator import SimulationConfig
from pylib import plot_network

from lib.util.build_input_sim_cooja import create_files

from lib.nsga import fast_nondominated_sort
from lib.problem.adapter import ProblemAdapter, Chromosome
from lib.problem.resolve import build_adapter


logger = logging.getLogger(__name__)

# Finite sentinel for individuals with no metrics / unextractable objectives.
# Kept above the frontend PENALTY_THRESHOLD (1e8) and the P1 penalty base (1e9),
# and finite (not float("inf")) so objectives stay JSON-serializable and niching
# distance math never produces NaN.
WORST_OBJECTIVE = 1e12


class RandomSearchStrategy(EngineStrategy):
    """
    Random search strategy.

    Baseline algorithm-agnostic generator: at each generation, samples
    ``population_size`` candidate solutions uniformly from the problem's
    decision space ``X`` via :py:meth:`ProblemAdapter.random_individual_generator`.
    No selection, no crossover, no mutation.

    Genome cache and penalty objectives behave identically to NSGA-III, so
    chromosomes already evaluated in earlier generations (or earlier runs
    of the same experiment) are not re-simulated, and infeasible candidates
    are penalised in the same gradient-of-deficit scheme.

    Purpose:
        Provide a minimal, algorithm-agnostic reference that highlights the
        separation between optimization logic and experiment orchestration,
        as described in Section 3.4 / Table 2 of the SimLab paper.
    """

    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        self._watch_thread: Thread | None = None
        self._sim_watch_thread: Thread | None = None
        self._stop_flag: bool = False

        # --- experiment parameters ---
        params = experiment.get("parameters", {}) or {}
        algorithm_config = params.get("algorithm", {}) or {}
        problem_config = params.get("problem", {}) or {}
        simulation_config = params.get("simulation", {}) or {}
        self._problem_name: str = str(problem_config.get("name", ""))

        src_repo_opts = experiment.get("source_repository_options", {}) or {}
        self.source_repository_options: dict[str, ObjectId] = {
            str(k): ObjectId(v) if isinstance(v, (str, bytes)) else v
            for k, v in src_repo_opts.items()
        }

        # Synthetic mode: when enabled, CSC files and source-repo lookups are skipped.
        syn_cfg: dict = simulation_config.get("synthetic", {}) or {}
        self._is_synthetic: bool = bool(syn_cfg.get("enabled", False))
        if self._is_synthetic:
            logger.info("[RandomSearch] Synthetic mode enabled — skipping CSC/source-repo for simulations.")

        # --- simulation and algorithm parameters ---
        self._sim_duration: int = int(simulation_config.get("duration", 120))
        rs_random_seed: int = int(algorithm_config.get("random_seed", 42))
        self._rng = random.Random(rs_random_seed)
        self._sim_rand_seeds: list[int] = resolve_simulation_seeds(
            simulation_config, self._rng, default_count=1
        )
        self._aggregator: "str | dict" = simulation_config.get("aggregator", "mean")
        self._pop_size: int = int(algorithm_config.get("population_size", 20))
        self._max_gen: int = int(algorithm_config.get("number_of_generations", 5))

        # Random search has no genetic operators, but the adapter still needs a
        # rng (e.g. for random_individual_generator); pass an empty config so it
        # falls back to safe defaults.
        self._problem_adapter: ProblemAdapter = build_adapter(
            problem_config,
            algorithm_config,
            self._rng,
        )

        self._metric_conv_config = experiment.get("data_conversion_config", {}) or {}

        obj = params.get("objectives", []) or []
        self._objective_keys: list[str] = [o["metric_name"] for o in obj]
        self._objective_goals: list[int] = [1 if o["goal"] == "min" else -1 for o in obj]
        if len(self._objective_keys) != len(self._objective_goals):
            raise ValueError(
                f"objective_keys ({len(self._objective_keys)}) and "
                f"objective_goals ({len(self._objective_goals)}) length mismatch"
            )
        logger.info(
            "[RandomSearch] Objectives: %s with goals: %s",
            self._objective_keys, self._objective_goals,
        )

        # --- loop state ---
        self._exp_id: ObjectId | None = None
        self._gen_index: int = 0
        self._generation_id: ObjectId | None = None
        self._lock = threading.Lock()
        self._sim_done_count: int = 0
        self._count_sims_inserted: int = 0
        self._current_population: list[Chromosome] = []
        self._inserted_genomes: set[str] = set()
        self._map_genome_objectives: dict[Chromosome, list[float]] = {}
        self._genome_objectives_cache: dict[str, list[float]] = {}

    # ------------------------------
    # EngineStrategy interface
    # ------------------------------
    def start(self):
        self._exp_id = (
            ObjectId(self.experiment["_id"])
            if isinstance(self.experiment.get("_id"), (str, bytes))
            else self.experiment.get("_id")
        )
        if not isinstance(self._exp_id, ObjectId):
            self._exp_id = ObjectId(str(self.experiment.get("_id")))

        self._gen_index = 0
        self._load_genome_cache_from_db()

        existing_generations = self.mongo.generation_repo.find_by_experiment(self._exp_id)
        if existing_generations:
            last = existing_generations[-1]
            self._generation_id = last["_id"]
            self._gen_index = int(last["index"])
            status = last.get("status")
            logger.info(
                "[RandomSearch] Resuming experiment %s from generation %s (index=%d, status=%s).",
                self._exp_id, self._generation_id, self._gen_index, status,
            )
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.RUNNING,
            })
            self._start_watcher()
            self._start_generation_poll()
            if status == EnumStatus.DONE:
                # Replay terminal generation so the loop continues.
                with self._lock:
                    self._handle_generation_done(self._generation_id)
            return

        self._current_population = self._problem_adapter.random_individual_generator(self._pop_size)
        self._generation_enqueue()
        self._start_watcher()
        self._start_generation_poll()
        # Catch-up: a fast (e.g. synthetic) generation may already be terminal
        # before the change-stream watcher subscribed. Reconcile once at startup.
        self._reconcile_current_generation()

    def event_simulation_done(self, sim_doc: dict):
        sim = sim_doc.get("fullDocument") or {}
        if sim.get("generation_id") != self._generation_id:
            return
        with self._lock:
            self._sim_done_count += 1
            logger.info(
                "[RandomSearch] Simulation terminal (%s): %d/%d for generation %s",
                sim.get("status"), self._sim_done_count,
                self._count_sims_inserted, self._generation_id,
            )

    def event_generation_done(self, gen_doc: dict):
        gen = gen_doc.get("fullDocument")
        if not gen:
            return
        with self._lock:
            self._handle_generation_done(ObjectId(gen.get("_id")))

    def stop(self):
        self._stop_flag = True
        for t in (self._watch_thread, self._sim_watch_thread):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=1.0)
        self._watch_thread = None
        self._sim_watch_thread = None

    # ------------------------------
    # Genome cache
    # ------------------------------
    def _load_genome_cache_from_db(self) -> None:
        assert self._exp_id is not None
        entries = self.mongo.genome_cache_repo.get_all_by_experiment(self._exp_id)
        for entry in entries:
            h = entry["genome_hash"]
            self._inserted_genomes.add(h)
            if entry.get("objectives") is not None:
                self._genome_objectives_cache[h] = entry["objectives"]
        if entries:
            logger.info(
                "[RandomSearch] Genome cache loaded: %d registered, %d with objectives.",
                len(entries), len(self._genome_objectives_cache),
            )

    # ------------------------------
    # Watchers
    # ------------------------------
    def _start_watcher(self):
        self._stop_flag = False

        for t in (self._watch_thread, self._sim_watch_thread):
            if t and t.is_alive():
                self._stop_flag = True
                t.join(timeout=1.0)
                self._stop_flag = False

        def _run_generation_watcher():
            def _callback(gen_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_generation_done(gen_doc)
                except Exception:
                    logger.exception("[RandomSearch] Generation watcher callback error.")

            logger.info("[RandomSearch] Starting Generation watcher (DONE or ERROR).")
            self.mongo.generation_repo.watch_status_terminal(_callback)

        def _run_simulation_watcher():
            def _callback(sim_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_simulation_done(sim_doc)
                except Exception:
                    logger.exception("[RandomSearch] Simulation watcher callback error.")

            logger.info("[RandomSearch] Starting Simulation watcher (DONE or ERROR).")
            self.mongo.simulation_repo.watch_status_terminal(_callback)

        self._watch_thread = Thread(
            target=_run_generation_watcher, daemon=True, name="randsearch-gen-watcher",
        )
        self._sim_watch_thread = Thread(
            target=_run_simulation_watcher, daemon=True, name="randsearch-sim-watcher",
        )
        self._watch_thread.start()
        self._sim_watch_thread.start()

    def _reconcile_current_generation(self) -> None:
        """Catch-up check for the current generation.

        Fast generations (e.g. synthetic mode, or fully genome-cached ones) can
        reach a terminal state before the change-stream watcher subscribes,
        which would otherwise miss the event. This reads the current status once
        and processes it if already terminal. Idempotent under the lock.
        """
        if self._generation_id is None:
            return
        try:
            gen = self.mongo.generation_repo.get(str(self._generation_id))
        except Exception:
            logger.exception("[RandomSearch] Reconcile: failed to read generation status.")
            return
        if gen and gen.get("status") in (EnumStatus.DONE, EnumStatus.ERROR):
            logger.warning(
                "[RandomSearch] Generation %s already %s at startup; processing (watcher race).",
                self._generation_id, gen.get("status"),
            )
            with self._lock:
                self._handle_generation_done(ObjectId(gen["_id"]))

    def _start_generation_poll(self) -> None:
        poll_interval = int(os.getenv("BATCH_POLL_INTERVAL", "30"))

        def _poll():
            while not self._stop_flag:
                time.sleep(poll_interval)
                if self._stop_flag or self._generation_id is None:
                    continue
                try:
                    gen = self.mongo.generation_repo.get(str(self._generation_id))
                    if gen and gen.get("status") in (EnumStatus.DONE, EnumStatus.ERROR):
                        logger.warning(
                            "[RandomSearch] Generation %s %s detected by polling fallback.",
                            self._generation_id, gen.get("status"),
                        )
                        with self._lock:
                            self._handle_generation_done(ObjectId(gen["_id"]))
                except Exception:
                    logger.exception("[RandomSearch] Polling fallback error.")

        Thread(target=_poll, daemon=True, name="randsearch-poll").start()

    # ------------------------------
    # Topology upload helper
    # ------------------------------
    def _upload_topology_async(
        self,
        exp_oid: ObjectId,
        gen_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        individual_id: str,
        config_snapshot: dict,
    ) -> None:
        def _do():
            try:
                topo_id = self._plot_topology(exp_oid, gen_index, ind_idx, config_snapshot)
                self.mongo.individual_repo.update_topology_picture(
                    individual_id, gen_oid, topo_id,
                )
            except Exception:
                logger.exception(
                    "[RandomSearch] Topology upload failed for gen=%s ind=%s",
                    gen_index, ind_idx,
                )

        Thread(target=_do, daemon=True, name=f"topo-rs-g{gen_index}-{ind_idx}").start()

    # ------------------------------
    # Generation / Queuing
    # ------------------------------
    def _generation_enqueue(self) -> None:
        assert self._exp_id is not None
        exp_oid = self._exp_id
        gen_index = self._gen_index
        population = self._current_population
        self._sim_done_count = 0

        gen_oid = ObjectId()

        # Insert the generation document (WAITING) BEFORE any simulation. Fast
        # workers (e.g. synthetic mode) can complete every simulation in
        # milliseconds; if the generation did not exist yet, master-node's
        # generation mark_done() would update a missing document (no-op) and the
        # generation would hang forever. Creating it first guarantees the close
        # signal lands on an existing document.
        gen_doc: Generation = {
            "_id": gen_oid,
            "experiment_id": exp_oid,
            "index": gen_index,
            "status": EnumStatus.WAITING,
            "start_time": datetime.now(),
            "end_time": None,
        }
        # Increment before inserting so the change-stream callback always sees
        # the already-updated index.
        self._gen_index += 1
        self._generation_id = self.mongo.generation_repo.insert(gen_doc)

        if gen_index == 0:
            self.mongo.experiment_repo.update(str(exp_oid), {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now(),
            })

        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456
        sims_inserted = 0
        seen_generation_hashes: set[str] = set()
        self._count_sims_inserted = 0

        for i, genome in enumerate(population):
            genome_hash = genome.get_hash()
            if genome_hash in seen_generation_hashes:
                logger.info(
                    "[RandomSearch] Duplicate genome %s in generation %d; skipping.",
                    genome_hash, gen_index,
                )
                continue
            seen_generation_hashes.add(genome_hash)

            # --- Case A: objectives already cached across runs ---
            if genome_hash in self._genome_objectives_cache:
                cached_obj = self._genome_objectives_cache[genome_hash]
                self._map_genome_objectives[genome] = cached_obj
                ind_doc: Individual = {
                    "experiment_id": exp_oid,
                    "generation_id": gen_oid,
                    "individual_id": genome_hash,
                    "chromosome": genome.to_dict(),
                    "objectives": self._objectives_list_to_original(cached_obj),
                    "topology_picture_id": None,
                }
                self.mongo.individual_repo.insert(ind_doc)
                config_topo = self._convert_genome_to_sim_config(
                    genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed,
                )
                self._upload_topology_async(
                    exp_oid, gen_oid, gen_index, i, genome_hash, dict(config_topo),
                )
                logger.info("[RandomSearch] Genome %s cached; skipping simulation.", genome_hash)
                continue

            # --- Case B: same genome already inserted this session ---
            if genome_hash in self._inserted_genomes:
                logger.info("[RandomSearch] Genome %s already inserted this session; skipping.", genome_hash)
                continue

            # --- Case C: infeasible genome — apply penalty ---
            penalty = self._problem_adapter.penalty_objectives(genome, len(self._objective_keys))
            if penalty is not None:
                self._map_genome_objectives[genome] = penalty
                ind_doc: Individual = {
                    "experiment_id": exp_oid,
                    "generation_id": gen_oid,
                    "individual_id": genome_hash,
                    "chromosome": genome.to_dict(),
                    "objectives": self._objectives_list_to_original(penalty),
                    "topology_picture_id": None,
                }
                self.mongo.individual_repo.insert(ind_doc)
                self._inserted_genomes.add(genome_hash)
                self.mongo.genome_cache_repo.insert(exp_oid, genome_hash, genome.to_dict())
                self.mongo.genome_cache_repo.set_objectives(self._exp_id, genome_hash, penalty)
                self._genome_objectives_cache[genome_hash] = penalty
                config_topo = self._convert_genome_to_sim_config(
                    genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed,
                )
                self._upload_topology_async(
                    exp_oid, gen_oid, gen_index, i, genome_hash, dict(config_topo),
                )
                logger.info(
                    "[RandomSearch] Genome %s infeasible (penalty=%.2e); skipping simulation.",
                    genome_hash, penalty[0],
                )
                continue

            # --- Case D: new genome — register and enqueue simulations ---
            config = self._convert_genome_to_sim_config(
                genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed,
            )
            self._count_sims_inserted += len(self._sim_rand_seeds)

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
            self.mongo.genome_cache_repo.insert(exp_oid, genome_hash, genome.to_dict())

            for seed in self._sim_rand_seeds:
                config["randomSeed"] = seed
                self._insert_simulation_db(genome_hash, exp_oid, gen_oid, config)
                sims_inserted += 1
                logger.info("[RandomSearch] SIM inserted SEED=%s genome=%s", seed, genome_hash)

            self._upload_topology_async(exp_oid, gen_oid, gen_index, i, genome_hash, dict(config))

        all_cached = sims_inserted == 0
        if all_cached:
            # No simulations to wait for — mark DONE and trigger the handler
            # directly (the change-stream event would otherwise never fire).
            self.mongo.generation_repo.mark_done(gen_oid)
            logger.info(
                "[RandomSearch] Generation %d: all %d genomes cached; marked DONE.",
                gen_index, len(population),
            )
            Thread(
                target=self._fire_generation_done,
                args=(gen_oid,),
                daemon=True,
                name=f"randsearch-gen-done-{gen_index}",
            ).start()
        else:
            logger.info(
                "[RandomSearch] Generation %d enqueued with %d individuals (%d simulations).",
                gen_index, len(population), sims_inserted,
            )

    def _fire_generation_done(self, gen_oid: ObjectId) -> None:
        try:
            with self._lock:
                self._handle_generation_done(gen_oid)
        except Exception:
            logger.exception("[RandomSearch] Direct generation-done trigger failed.")

    # ------------------------------
    # Generation-done handling
    # ------------------------------
    def _handle_generation_done(self, gen_oid: ObjectId):
        if self._stop_flag or self._generation_id is None:
            return
        if gen_oid != self._generation_id:
            return

        logger.info("[RandomSearch] EVENT GENERATION TERMINAL gen_id=%s", self._generation_id)

        map_ind_metrics = self.mongo.generation_repo.get_simulations_metrics_by_individual(
            generation_id=self._generation_id,
            metrics=self._objective_keys,
            aggregator=self._aggregator,
        )

        n_obj = len(self._objective_keys)
        worst_objectives = [WORST_OBJECTIVE] * n_obj

        for ind in self._current_population:
            if self._map_genome_objectives.get(ind) is not None:
                continue
            ind_metrics = map_ind_metrics.get(ind.get_hash())
            if ind_metrics is None:
                logger.warning(
                    "[RandomSearch] No metrics for genome %s; assigning worst objectives.",
                    ind.get_hash(),
                )
                self._map_genome_objectives[ind] = worst_objectives
                continue
            obj_vector = self._extract_objectives_to_minimization(ind_metrics)
            if obj_vector is None:
                logger.warning(
                    "[RandomSearch] Could not extract objective vector for %s; assigning worst.",
                    ind.get_hash(),
                )
                self._map_genome_objectives[ind] = worst_objectives
            else:
                self._map_genome_objectives[ind] = obj_vector

        self._update_individual_objectives()

        # Stop condition: generated max_gen + 1 generations (G_0 .. G_max_gen).
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
                first_pareto_front = (
                    [pareto_items[idx] for idx in pareto_fronts[0]]
                    if pareto_fronts else []
                )
            except Exception:
                logger.exception("[RandomSearch] Could not compute final Pareto front.")
                first_pareto_front = []
            self._finalize_experiment(pareto_front=first_pareto_front)
            return

        # Otherwise, sample the next generation uniformly from X.
        self._current_population = self._problem_adapter.random_individual_generator(self._pop_size)
        self._generation_enqueue()
        logger.info("[RandomSearch] New random generation enqueued; waiting for results.")

    def _update_individual_objectives(self) -> None:
        if self._generation_id is None:
            return
        for genome in self._current_population:
            objectives = self._map_genome_objectives.get(genome)
            if objectives is None:
                continue
            original_objectives = self._objectives_list_to_original(objectives)
            genome_hash = genome.get_hash()
            self.mongo.individual_repo.update_objectives(
                genome_hash, self._generation_id, original_objectives,
            )
            if genome_hash not in self._genome_objectives_cache:
                self.mongo.genome_cache_repo.set_objectives(
                    self._exp_id, genome_hash, objectives,
                )
                self._genome_objectives_cache[genome_hash] = objectives

    # ------------------------------
    # Objective conversion helpers
    # ------------------------------
    def _objectives_list_to_original(self, vec: list[float]) -> list[float]:
        return [float(v * s) for v, s in zip(vec, self._objective_goals)]

    def _objectives_to_original(self, vec: list[float]) -> dict[str, float]:
        return {
            k: float(v * s)
            for k, v, s in zip(self._objective_keys, vec, self._objective_goals)
        }

    def _extract_objectives_to_minimization(
        self, sim_metrics_map: dict[str, float],
    ) -> Optional[list[float]]:
        obj_vector: list[float] = []
        for key, goal in zip(self._objective_keys, self._objective_goals):
            metric_value = sim_metrics_map.get(key)
            if metric_value is None:
                logger.warning("[RandomSearch] Missing metric '%s'.", key)
                return None
            obj_vector.append(metric_value * goal)
        return obj_vector

    # ------------------------------
    # Simulation and topology helpers
    # ------------------------------
    def _convert_genome_to_sim_config(
        self,
        genome: Chromosome,
        gen_index: int,
        ind_idx: int,
        seed: int,
    ) -> SimulationConfig:
        simulationElements = self._problem_adapter.encode_simulation_input(genome)
        config: SimulationConfig = {
            "name": f"random-g{gen_index}-{ind_idx}-{seed}",
            "duration": self._sim_duration,
            "randomSeed": seed,
            "radiusOfReach": self._problem_adapter.radius_of_reach,
            "radiusOfInter": self._problem_adapter.radius_of_inter,
            "region": self._problem_adapter.bounds,
            "simulationElements": simulationElements,
        }
        return config

    def _insert_simulation_db(
        self,
        genome_hash: str,
        exp_oid: ObjectId,
        gen_oid: ObjectId,
        config: SimulationConfig,
    ) -> ObjectId:
        if self._is_synthetic:
            csc_file_id = None
            pos_file_id = None
            src_id = None
        else:
            files_ids = create_files(config, self.mongo.fs_handler)
            csc_file_id = files_ids.get("csc_file_id", "")
            pos_file_id = files_ids.get("pos_file_id", "")
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
            "pos_file_id": pos_file_id,
            "csc_file_id": csc_file_id,
            "source_repository_id": src_id,
            "log_cooja_id": "",
            "runtime_log_id": "",
            "csv_log_id": "",
            "network_metrics": {},
        }
        return self.mongo.simulation_repo.insert(sim_doc)

    def _genome_hash_to_source(self, genome_hash: str):
        for genome in self._current_population:
            if genome.get_hash() == genome_hash:
                return genome.get_source_by_mac_protocol(self.source_repository_options)
        raise ValueError(f"Genome hash {genome_hash} not found in current population")

    def _plot_topology(
        self,
        exp_oid: ObjectId,
        gen_index: int,
        ind_idx: int,
        config: SimulationConfig,
    ) -> ObjectId:
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        image_tmp_path = tmp_dir / f"topology-{exp_oid}-{gen_index}-{ind_idx}.png"

        plot_network.plot_network_save_from_sim(str(image_tmp_path), config)

        topology_picture_id = self.mongo.fs_handler.upload_file(
            str(image_tmp_path),
            f"topology-{exp_oid}-{gen_index}-{ind_idx}",
        )

        if os.path.exists(image_tmp_path):
            os.remove(image_tmp_path)

        return topology_picture_id

    # ------------------------------
    # Finalization
    # ------------------------------
    def _finalize_experiment(
        self,
        system_msg: Optional[str] = None,
        pareto_front: Optional[list[dict]] = None,
    ) -> None:
        assert self._exp_id is not None
        if pareto_front is not None:
            logger.info("[RandomSearch] Experiment %s completed.", self._exp_id)
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now(),
                "system_message": system_msg
                    if system_msg is not None
                    else f"Experiment {self._exp_id} completed.",
                "pareto_front": pareto_front,
            })
        else:
            logger.error("[RandomSearch] Experiment %s finished with error.", self._exp_id)
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.ERROR,
                "system_message": system_msg
                    if system_msg is not None
                    else f"Experiment {self._exp_id} finished.",
                "end_time": datetime.now(),
            })
        self.stop()
