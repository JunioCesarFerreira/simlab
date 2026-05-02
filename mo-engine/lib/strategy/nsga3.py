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
from lib.problem.chromosomes import chromosome_from_dict
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
        self._problem_name: str = str(problem_config.get("name", ""))

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
        self._sim_done_count: int = 0
        self._sim_watch_thread: Thread | None = None
        self._count_sims_inserted: int = 0

        # --- nsga3 workflow ---
        self._current_population: list[Chromosome] = []   # P_t
        self._parents: list[Chromosome] = []              # P_{t-1}
        self._inserted_genomes: set[str] = set()          # hashes already inserted to DB (this session)
        self._map_genome_objectives: dict[Chromosome, list[float]] = {}

        # --- genome cache (persistent deduplication) ---
        # hash -> minimization-space objectives for genomes already evaluated in any
        # prior session. Individuals persisted per generation keep original objective
        # values for UX/analytics, but the cache stays in minimization space so the
        # evolutionary loop can reuse it directly without extra metadata in pylib.
        self._genome_objectives_cache: dict[str, list[float]] = {}


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

        # Restore genome cache from a previous run (if any).
        # This populates _inserted_genomes and _genome_objectives_cache so that
        # genomes already evaluated are never re-submitted to Cooja.
        self._load_genome_cache_from_db()

        existing_generations = self.mongo.generation_repo.find_by_experiment(self._exp_id)
        if existing_generations:
            should_process_terminal = self._resume_existing_generation(existing_generations)
            self._start_watcher()
            self._start_generation_poll()
            if should_process_terminal:
                with self._lock:
                    self._handle_generation_done(self._generation_id)
            return

        self._current_population = self._problem_adapter.random_individual_generator(self._pop_size)

        self._generation_enqueue()
        self._start_watcher()
        self._start_generation_poll()


    def event_simulation_done(self, sim_doc: dict):
        """
        Called for every simulation that reaches a terminal state (DONE or ERROR).
        Accounts for progress only — does not control algorithm flow.
        """
        sim = sim_doc.get("fullDocument") or {}
        if sim.get("generation_id") != self._generation_id:
            return
        
        with self._lock:
            self._sim_done_count += 1
            logger.info(
                "[NSGA-III] Simulation terminal (%s): %d/%d for generation %s",
                sim.get("status"), self._sim_done_count, self._count_sims_inserted, self._generation_id
            )


    def event_generation_done(self, gen_doc: dict):
        """
        Called when a generation reaches a terminal state (DONE or ERROR).
        Controls the algorithm flow: objective extraction, NSGA-III selection, next generation.
        """
        gen = gen_doc.get("fullDocument")
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
        for t in (self._watch_thread, self._sim_watch_thread):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=1.0)
        self._watch_thread = None
        self._sim_watch_thread = None


# ------------------------------
# Genome cache
# ------------------------------
    def _load_genome_cache_from_db(self) -> None:
        """
        Populate in-memory deduplication structures from the persisted genome cache.

        After this call:
        - _inserted_genomes contains every hash ever registered for this experiment,
          preventing re-insertion of Individual documents and re-creation of simulations.
        - _genome_objectives_cache maps hash -> objectives for every genome whose
          evaluation has already completed, enabling immediate reuse without simulation.
        """
        assert self._exp_id is not None
        entries = self.mongo.genome_cache_repo.get_all_by_experiment(self._exp_id)
        for entry in entries:
            h = entry["genome_hash"]
            self._inserted_genomes.add(h)
            if entry.get("objectives") is not None:
                self._genome_objectives_cache[h] = entry["objectives"]
        if entries:
            logger.info(
                "[NSGA-III] Genome cache loaded: %d registered, %d with objectives.",
                len(entries),
                len(self._genome_objectives_cache),
            )

    def _resume_existing_generation(self, generations: list[Generation]) -> bool:
        """
        Resume the latest persisted generation for the current experiment.

        Returns True when the latest generation is already terminal and must be
        processed immediately to continue the evolutionary loop.
        """
        last_generation = generations[-1]
        self._generation_id = last_generation["_id"]
        self._gen_index = int(last_generation["index"]) + 1
        self._restore_population_state(generations, last_generation)

        status = last_generation.get("status")
        if status in (EnumStatus.WAITING, EnumStatus.RUNNING):
            logger.info(
                "[NSGA-III] Resuming experiment %s from existing generation %s (index=%d, status=%s).",
                self._exp_id,
                self._generation_id,
                last_generation["index"],
                status,
            )
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.RUNNING,
            })
            return False

        if status == EnumStatus.DONE:
            logger.info(
                "[NSGA-III] Replaying completed generation %s (index=%d) to continue the experiment.",
                self._generation_id,
                last_generation["index"],
            )
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": EnumStatus.RUNNING,
            })
            return True

        raise RuntimeError(
            f"Cannot resume NSGA-III from generation index={last_generation['index']} with status={status}."
        )

    def _restore_population_state(
        self,
        generations: list[Generation],
        current_generation: Generation,
    ) -> None:
        current_index = int(current_generation["index"])
        self._current_population, current_map, pending_individuals = self._load_generation_population(
            current_generation["_id"]
        )
        self._map_genome_objectives.update(current_map)
        self._count_sims_inserted = pending_individuals * len(self._sim_rand_seeds)

        if current_index == 0:
            self._parents = []
            return

        previous_generation = next(
            (gen for gen in reversed(generations[:-1]) if int(gen["index"]) == current_index - 1),
            None,
        )
        if previous_generation is None:
            raise RuntimeError(
                f"Cannot resume generation {current_index}: previous generation {current_index - 1} not found."
            )

        self._parents, parent_map, _ = self._load_generation_population(previous_generation["_id"])
        self._map_genome_objectives.update(parent_map)

    def _load_generation_population(
        self,
        generation_id: ObjectId,
    ) -> tuple[list[Chromosome], dict[Chromosome, list[float]], int]:
        individuals = self.mongo.individual_repo.find_by_generation(generation_id)
        if not individuals:
            raise RuntimeError(f"Cannot resume generation {generation_id}: no individuals were found.")

        population: list[Chromosome] = []
        objectives_map: dict[Chromosome, list[float]] = {}
        pending_individuals = 0

        for ind in sorted(individuals, key=lambda item: item["individual_id"]):
            chromosome = chromosome_from_dict(self._problem_name, ind["chromosome"])
            population.append(chromosome)

            individual_objectives = ind.get("objectives")
            if individual_objectives:
                objectives_map[chromosome] = self._objectives_list_to_minimization(
                    [float(value) for value in individual_objectives]
                )
                continue

            cache_objectives = self._genome_objectives_cache.get(ind["individual_id"])
            if cache_objectives:
                objectives_map[chromosome] = [float(value) for value in cache_objectives]
            else:
                pending_individuals += 1

        return population, objectives_map, pending_individuals

# ------------------------------
# Watcher (Change Stream) + Polling fallback
# ------------------------------
    def _start_watcher(self):
        self._stop_flag = False

        # Stop any existing threads
        for t in (self._watch_thread, self._sim_watch_thread):
            if t and t.is_alive():
                self._stop_flag = True
                t.join(timeout=1.0)
                self._stop_flag = False

        # --- Generation watcher: controls flow ---
        def _run_generation_watcher():
            def _callback(gen_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_generation_done(gen_doc)
                except Exception:
                    logger.exception("[NSGA-III] Generation watcher callback error.")

            logger.info("[NSGA-III] Starting Generation watcher (DONE or ERROR).")
            self.mongo.generation_repo.watch_status_terminal(_callback)

        # --- Simulation watcher: accounting only ---
        def _run_simulation_watcher():
            def _callback(sim_doc: dict):
                if self._stop_flag:
                    return
                try:
                    self.event_simulation_done(sim_doc)
                except Exception:
                    logger.exception("[NSGA-III] Simulation watcher callback error.")

            logger.info("[NSGA-III] Starting Simulation watcher (DONE or ERROR).")
            self.mongo.simulation_repo.watch_status_terminal(_callback)

        self._watch_thread = Thread(target=_run_generation_watcher, daemon=True, name="nsga3-gen-watcher")
        self._sim_watch_thread = Thread(target=_run_simulation_watcher, daemon=True, name="nsga3-sim-watcher")
        self._watch_thread.start()
        self._sim_watch_thread.start()


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
        self._sim_done_count = 0

        # Pre-generate generation ObjectId so simulations are inserted with generation_id set,
        # avoiding the race where master-node picks up a simulation before the generation exists.
        gen_oid = ObjectId()

        first_seed = self._sim_rand_seeds[0] if self._sim_rand_seeds else 123456
        sims_inserted = 0
        seen_generation_hashes: set[str] = set()
        
        self._count_sims_inserted = 0

        for i, genome in enumerate(population):
            genome_hash = genome.get_hash()
            if genome_hash in seen_generation_hashes:
                logger.info(
                    "Genome %s already present in generation %d; skipping duplicate individual.",
                    genome_hash,
                    gen_index,
                )
                continue
            seen_generation_hashes.add(genome_hash)

            # --- Case A: objectives already in persistent cache ---
            # Pre-populate the objectives map so _handle_generation_done skips this genome.
            # Insert the Individual with objectives already filled; no simulations needed.
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
                    genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed
                )
                self._upload_topology_async(exp_oid, gen_oid, gen_index, i, genome_hash, dict(config_topo))
                logger.info("Genome %s has cached objectives; skipping simulation.", genome_hash)
                continue

            # --- Case B: genome registered in this session but results still pending ---
            # (Rare: same genome produced twice by the GA in one generation.)
            if genome_hash in self._inserted_genomes:
                logger.info("Genome %s already inserted this SESSION; skipping.", genome_hash)
                continue

            # --- Case C: infeasible genome — assign gradient penalty, skip simulation ---
            # Adapters that define hard constraints (e.g. trajectory coverage in P2)
            # return a penalty vector instead of None.  The penalty is larger the more
            # infeasible the chromosome is, so the evolutionary pressure still favours
            # less-infeasible solutions.  Penalised individuals never reach front 0.
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
                    genome=genome, gen_index=gen_index, ind_idx=i, seed=first_seed
                )
                self._upload_topology_async(exp_oid, gen_oid, gen_index, i, genome_hash, dict(config_topo))
                logger.info(
                    "Genome %s is infeasible (penalty=%.2e); skipping simulation.", genome_hash, penalty[0]
                )
                continue

            # --- Case D: new genome — register in cache, insert Individual, queue simulations ---
            config = self._convert_genome_to_sim_config(
                genome=genome,
                gen_index=gen_index,
                ind_idx=i,
                seed=first_seed
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
                logger.info("SIM inserted SEED=%s genome=%s", seed, genome_hash)

            self._upload_topology_async(exp_oid, gen_oid, gen_index, i, genome_hash, dict(config))

        # When every genome already has cached objectives there are no simulations to
        # wait for.  Insert the generation directly as DONE so the change-stream watcher
        # fires immediately and the algorithm can advance to the next generation.
        all_cached = sims_inserted == 0
        gen_doc: Generation = {
            "_id": gen_oid,
            "experiment_id": exp_oid,
            "index": gen_index,
            "status": EnumStatus.DONE if all_cached else EnumStatus.WAITING,
            "start_time": datetime.now(),
            "end_time": datetime.now() if all_cached else None,
        }

        # Increment before inserting so the change-stream callback (which fires
        # asynchronously) always sees the already-updated index.
        self._gen_index += 1
        self._generation_id = self.mongo.generation_repo.insert(gen_doc)

        if gen_index == 0:
            self.mongo.experiment_repo.update(str(exp_oid), {
                "status": EnumStatus.RUNNING,
                "start_time": datetime.now()
            })

        if all_cached:
            logger.info(
                "[NSGA-III] Generation %d: all %d genomes have cached objectives; inserted as DONE.",
                gen_index, len(population),
            )
            # Change-stream event for this insert may be lost: the watcher may not
            # have subscribed yet (first generation) or may be mid-reconnect. Fire
            # the handler directly in a worker thread so the evolution loop always
            # advances, even when every genome is cached or infeasible.
            Thread(
                target=self._fire_generation_done,
                args=(gen_oid,),
                daemon=True,
                name=f"nsga3-gen-done-{gen_index}",
            ).start()
        else:
            logger.info(
                "[NSGA-III] Generation %d enqueued with %d individuals (%d new simulations).",
                gen_index, len(population), sims_inserted,
            )

    def _fire_generation_done(self, gen_oid: ObjectId) -> None:
        """
        Trigger generation-done handling from a worker thread without going
        through the MongoDB change stream. Safe against races: if the change
        stream also delivers the same event, the second call is a no-op
        because `_handle_generation_done` guards on `gen_oid == self._generation_id`.
        """
        try:
            with self._lock:
                self._handle_generation_done(gen_oid)
        except Exception:
            logger.exception("[NSGA-III] Direct generation-done trigger failed.")


    def _update_individual_objectives(self) -> None:
        """
        Persists computed objectives to Individual documents in the DB and to the
        genome cache so that future generations (and future engine restarts) can
        reuse them without re-running simulations.
        Called right after objectives are computed, before evolution.
        """
        if self._generation_id is None:
            return
        for genome in self._current_population:
            objectives = self._map_genome_objectives.get(genome)
            if objectives is None:
                continue
            original_objectives = self._objectives_list_to_original(objectives)
            genome_hash = genome.get_hash()
            self.mongo.individual_repo.update_objectives(
                genome_hash,
                self._generation_id,
                original_objectives,
            )
            # Persist to genome cache only on first evaluation (avoids redundant writes).
            # Cache values remain in minimization space because pylib does not store
            # metadata describing the objective representation.
            if genome_hash not in self._genome_objectives_cache:
                self.mongo.genome_cache_repo.set_objectives(
                    self._exp_id,
                    genome_hash,
                    objectives,
                )
                self._genome_objectives_cache[genome_hash] = objectives


    def _objectives_list_to_original(self, vec: list[float]) -> list[float]:
        return [float(v * s) for v, s in zip(vec, self._objective_goals)]


    def _objectives_list_to_minimization(self, vec: list[float]) -> list[float]:
        return [float(v * s) for v, s in zip(vec, self._objective_goals)]


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
