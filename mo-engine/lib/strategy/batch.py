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
from lib.problem.chromosomes import chromosome_from_dict
from lib.problem.resolve import build_adapter


logger = logging.getLogger(__name__)


class BatchStrategy(EngineStrategy):
    """
    Batch execution strategy.

    Unlike evolutionary strategies (e.g. NSGA-III), this strategy does not
    generate, cross-over, mutate or select individuals. The chromosomes are
    provided upfront in the experiment payload (``parameters.chromosomes``)
    and are executed as a single generation.

    Flow:
      1. Parse the chromosomes from the experiment input.
      2. Create a single generation (index 0) with all chromosomes as
         individuals and enqueue their simulations.
      3. Wait for every simulation to reach a terminal state.
      4. Consolidate simulation metrics into per-individual objectives.
      5. Finalize the experiment (compute Pareto front, mark as DONE).
    """

    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        self._watch_thread: Thread | None = None
        self._sim_watch_thread: Thread | None = None
        self._stop_flag: bool = False

        params = experiment.get("parameters", {}) or {}
        problem_config = params.get("problem", {}) or {}
        simulation_config = params.get("simulation", {}) or {}
        self._problem_name: str = str(problem_config.get("name", ""))

        src_repo_opts = experiment.get("source_repository_options", {}) or {}
        self.source_repository_options: dict[str, ObjectId] = {
            str(k): ObjectId(v) if isinstance(v, (str, bytes)) else v
            for k, v in src_repo_opts.items()
        }

        self._sim_duration: int = int(simulation_config.get("duration", 120))
        rng_seed: int = int(params.get("random_seed", 42))
        self._rng = random.Random(rng_seed)
        self._sim_rand_seeds: list[int] = resolve_simulation_seeds(
            simulation_config, self._rng, default_count=1
        )

        self._problem_adapter: ProblemAdapter = build_adapter(
            problem_config,
            {},  # no GA operator configuration in batch mode
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
            "[Batch] Objectives: %s with goals: %s",
            self._objective_keys, self._objective_goals,
        )

        # Chromosomes provided as input. Accepted both under ``parameters.chromosomes``
        # and at the experiment root (``experiment.chromosomes``) for flexibility.
        raw_chromosomes = (
            params.get("chromosomes")
            or experiment.get("chromosomes")
            or []
        )
        if not raw_chromosomes:
            raise ValueError(
                "Batch strategy requires a non-empty list of chromosomes "
                "in parameters.chromosomes."
            )
        self._chromosomes: list[Chromosome] = [
            chromosome_from_dict(self._problem_name, c) for c in raw_chromosomes
        ]
        logger.info("[Batch] Loaded %d chromosomes from input.", len(self._chromosomes))

        self._exp_id: ObjectId | None = None
        self._generation_id: ObjectId | None = None
        self._lock = threading.Lock()
        self._sim_done_count: int = 0
        self._count_sims_inserted: int = 0
        self._map_genome_objectives: dict[Chromosome, list[float]] = {}
        self._inserted_genomes: set[str] = set()

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

        existing_generations = self.mongo.generation_repo.find_by_experiment(self._exp_id)
        if existing_generations:
            last = existing_generations[-1]
            self._generation_id = last["_id"]
            status = last.get("status")
            logger.info(
                "[Batch] Experiment %s already has generation %s (status=%s); resuming.",
                self._exp_id, self._generation_id, status,
            )
            self._start_watcher()
            self._start_generation_poll()
            if status == EnumStatus.DONE:
                with self._lock:
                    self._handle_generation_done(self._generation_id)
            return

        self._enqueue_batch()
        self._start_watcher()
        self._start_generation_poll()

    def event_simulation_done(self, sim_doc: dict):
        sim = sim_doc.get("fullDocument") or {}
        if sim.get("generation_id") != self._generation_id:
            return
        with self._lock:
            self._sim_done_count += 1
            logger.info(
                "[Batch] Simulation terminal (%s): %d/%d for generation %s",
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
    # Watchers
    # ------------------------------
    def _start_watcher(self):
        self._stop_flag = False

        def _run_generation_watcher():
            def _callback(gen_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_generation_done(gen_doc)
                except Exception:
                    logger.exception("[Batch] Generation watcher callback error.")

            logger.info("[Batch] Starting Generation watcher (DONE or ERROR).")
            self.mongo.generation_repo.watch_status_terminal(_callback)

        def _run_simulation_watcher():
            def _callback(sim_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_simulation_done(sim_doc)
                except Exception:
                    logger.exception("[Batch] Simulation watcher callback error.")

            logger.info("[Batch] Starting Simulation watcher (DONE or ERROR).")
            self.mongo.simulation_repo.watch_status_terminal(_callback)

        self._watch_thread = Thread(
            target=_run_generation_watcher, daemon=True, name="batch-gen-watcher"
        )
        self._sim_watch_thread = Thread(
            target=_run_simulation_watcher, daemon=True, name="batch-sim-watcher"
        )
        self._watch_thread.start()
        self._sim_watch_thread.start()

    def _start_generation_poll(self) -> None:
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
                            "[Batch] Generation %s %s detected by polling fallback.",
                            self._generation_id, gen.get("status"),
                        )
                        with self._lock:
                            self._handle_generation_done(ObjectId(gen["_id"]))
                except Exception:
                    logger.exception("[Batch] Polling fallback error.")

        Thread(target=_poll, daemon=True, name="batch-poll").start()

    # ------------------------------
    # Single-generation enqueue
    # ------------------------------
    def _enqueue_batch(self) -> None:
        assert self._exp_id is not None
        exp_oid = self._exp_id
        gen_oid = ObjectId()
        gen_index = 0
        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456

        sims_inserted = 0
        self._sim_done_count = 0
        self._count_sims_inserted = 0

        for i, genome in enumerate(self._chromosomes):
            genome_hash = genome.get_hash()

            if genome_hash in self._inserted_genomes:
                logger.info(
                    "[Batch] Duplicate chromosome %s in input; skipping additional insertion.",
                    genome_hash,
                )
                continue

            penalty = self._problem_adapter.penalty_objectives(
                genome, len(self._objective_keys)
            )
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
                config_topo = self._convert_genome_to_sim_config(
                    genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed
                )
                self._upload_topology_async(
                    exp_oid, gen_oid, gen_index, i, genome_hash, dict(config_topo)
                )
                logger.info(
                    "[Batch] Chromosome %s is infeasible (penalty=%.2e); skipping simulation.",
                    genome_hash, penalty[0],
                )
                continue

            config = self._convert_genome_to_sim_config(
                genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed
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

            for seed in self._sim_rand_seeds:
                config["randomSeed"] = seed
                self._insert_simulation_db(genome_hash, exp_oid, gen_oid, config)
                sims_inserted += 1
                logger.info(
                    "[Batch] SIM inserted SEED=%s genome=%s", seed, genome_hash
                )

            self._upload_topology_async(
                exp_oid, gen_oid, gen_index, i, genome_hash, dict(config)
            )

        all_penalised = sims_inserted == 0
        gen_doc: Generation = {
            "_id": gen_oid,
            "experiment_id": exp_oid,
            "index": gen_index,
            "status": EnumStatus.DONE if all_penalised else EnumStatus.WAITING,
            "start_time": datetime.now(),
            "end_time": datetime.now() if all_penalised else None,
        }
        self._generation_id = self.mongo.generation_repo.insert(gen_doc)

        self.mongo.experiment_repo.update(str(exp_oid), {
            "status": EnumStatus.RUNNING,
            "start_time": datetime.now(),
        })

        if all_penalised:
            logger.info(
                "[Batch] No simulations enqueued (all chromosomes penalised); "
                "generation marked DONE immediately."
            )
            Thread(
                target=self._fire_generation_done,
                args=(gen_oid,),
                daemon=True,
                name="batch-gen-done-0",
            ).start()
        else:
            logger.info(
                "[Batch] Generation 0 enqueued with %d individuals (%d simulations).",
                len(self._chromosomes), sims_inserted,
            )

    def _fire_generation_done(self, gen_oid: ObjectId) -> None:
        try:
            with self._lock:
                self._handle_generation_done(gen_oid)
        except Exception:
            logger.exception("[Batch] Direct generation-done trigger failed.")

    # ------------------------------
    # Generation-done handling
    # ------------------------------
    def _handle_generation_done(self, gen_oid: ObjectId):
        """Core handler — must be called while holding self._lock."""
        if self._stop_flag or self._generation_id is None:
            return
        if gen_oid != self._generation_id:
            return

        logger.info("[Batch] EVENT GENERATION TERMINAL gen_id=%s", self._generation_id)

        map_ind_metrics = self.mongo.generation_repo.get_simulations_metrics_by_individual(
            generation_id=self._generation_id,
            metrics=self._objective_keys,
        )

        n_obj = len(self._objective_keys)
        worst_objectives = [float("inf")] * n_obj

        for ind in self._chromosomes:
            if self._map_genome_objectives.get(ind) is not None:
                continue
            ind_metrics = map_ind_metrics.get(ind.get_hash())
            if ind_metrics is None:
                logger.warning(
                    "[Batch] No metrics for genome %s; assigning worst objectives.",
                    ind.get_hash(),
                )
                self._map_genome_objectives[ind] = worst_objectives
                continue
            obj_vector = self._extract_objectives_to_minimization(ind_metrics)
            if obj_vector is None:
                logger.warning(
                    "[Batch] Could not extract objective vector for %s; assigning worst.",
                    ind.get_hash(),
                )
                self._map_genome_objectives[ind] = worst_objectives
            else:
                self._map_genome_objectives[ind] = obj_vector

        self._update_individual_objectives()

        try:
            pareto_items: list[dict] = []
            all_objectives: list[list[float]] = []
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
            logger.exception("[Batch] Could not compute final Pareto front.")
            first_pareto_front = []

        self._finalize_experiment(pareto_front=first_pareto_front)

    def _update_individual_objectives(self) -> None:
        if self._generation_id is None:
            return
        for genome in self._chromosomes:
            objectives = self._map_genome_objectives.get(genome)
            if objectives is None:
                continue
            self.mongo.individual_repo.update_objectives(
                genome.get_hash(),
                self._generation_id,
                self._objectives_list_to_original(objectives),
            )

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
        self, sim_metrics_map: dict[str, float]
    ) -> Optional[list[float]]:
        obj_vector: list[float] = []
        for key, goal in zip(self._objective_keys, self._objective_goals):
            metric_value = sim_metrics_map.get(key)
            if metric_value is None:
                logger.warning("[Batch] Missing metric '%s'.", key)
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
            "name": f"batch-g{gen_index}-{ind_idx}-{seed}",
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
        files_ids = create_files(config, self.mongo.fs_handler)
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
        for genome in self._chromosomes:
            if genome.get_hash() == genome_hash:
                return genome.get_source_by_mac_protocol(self.source_repository_options)
        raise ValueError(f"Genome hash {genome_hash} not found in batch chromosomes")

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
                    individual_id, gen_oid, topo_id
                )
            except Exception:
                logger.exception(
                    "[Batch] Topology upload failed for gen=%s ind=%s",
                    gen_index, ind_idx,
                )

        Thread(target=_do, daemon=True, name=f"topo-batch-{ind_idx}").start()

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
            logger.info("[Batch] Experiment %s completed.", self._exp_id)
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.DONE,
                "end_time": datetime.now(),
                "system_message": system_msg
                    if system_msg is not None
                    else f"Experiment {self._exp_id} completed.",
                "pareto_front": pareto_front,
            })
        else:
            logger.error("[Batch] Experiment %s finished with error.", self._exp_id)
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.ERROR,
                "system_message": system_msg
                    if system_msg is not None
                    else f"Experiment {self._exp_id} finished.",
                "end_time": datetime.now(),
            })
        self.stop()
