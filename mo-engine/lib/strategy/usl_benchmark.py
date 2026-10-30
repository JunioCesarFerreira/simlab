"""A scalability benchmark of the simulation infrastructure.

Not an optimisation. The same simulation — one chromosome, one seed — is run
in **waves** of ``N`` identical copies at a time, for several values of
``N``, so that the only thing that changes between waves is how much work the
infrastructure carries at once. The time each wave takes is fitted to the
Universal Scalability Law by :mod:`pylib.scalability`.

Each wave is one generation, tagged ``benchmark: {level, repetition, wave}``.
Reusing the generation is what makes this cheap: the master-node places,
runs, recovers and closes a wave exactly as it does any other generation, so
what is measured is the real path an optimisation takes — not a model of it.

Three choices worth stating:

* **Identical work.** One chromosome and one seed for every simulation. Any
  variation in the work itself would show up as variation in the timing and
  be read as a property of the infrastructure.
* **Counterbalanced order.** Repetitions alternate ascending and descending
  concurrency. A drift in the machines during the run — another load on a
  shared host, a cache warming up — then lands on every level instead of
  biasing the small ones against the large ones.
* **A wave that produces nothing stops the benchmark.** One failure in a wave
  is a measurement that did not happen and is reported as such; a wave with
  no result at all means the workload itself does not run, and measuring it
  further would only burn time.
"""
import logging
import os
import random
import threading
import time
from threading import Thread
from typing import Any, Optional

from bson import ObjectId

from .base import EngineStrategy
from .simulation_seeds import resolve_simulation_seeds

from pylib.db import EnumStatus
from pylib.db.models import Generation, Individual, Simulation
from pylib.config.simulator import SimulationConfig
from pylib.timeutil import utc_now

from lib.problem.adapter import Chromosome, ProblemAdapter
from lib.problem.chromosomes import chromosome_from_dict
from lib.problem.resolve import build_adapter
from lib.util.build_input_sim_cooja import create_files

logger = logging.getLogger(__name__)

STRATEGY_NAME = "usl_benchmark"

#: How many random chromosomes to draw before giving up on finding a feasible
#: one. A problem where fifty draws are all infeasible is one whose random
#: generator does not produce simulable networks; the benchmark says so
#: instead of looping.
FEASIBLE_ATTEMPTS = 50

DEFAULT_REPETITIONS = 3


def build_waves(levels: list[Any], repetitions: int) -> list[tuple[int, int]]:
    """The ordered list of ``(level, repetition)`` to run.

    Levels are de-duplicated and sorted; non-positive ones are dropped. Even
    repetitions run ascending and odd ones descending, so a slow drift in the
    machines is spread over every level rather than accumulating at one end.
    """
    ordered = sorted({int(level) for level in levels if int(level) >= 1})
    if not ordered:
        raise ValueError("a benchmark needs at least one concurrency level >= 1")
    if repetitions < 1:
        raise ValueError("a benchmark needs at least one repetition")
    waves: list[tuple[int, int]] = []
    for repetition in range(repetitions):
        sequence = ordered if repetition % 2 == 0 else list(reversed(ordered))
        waves.extend((level, repetition) for level in sequence)
    return waves


def choose_workload(
    adapter: ProblemAdapter, rng: random.Random, n_objectives: int,
    *, attempts: int = FEASIBLE_ATTEMPTS,
) -> tuple[Chromosome, int]:
    """One feasible chromosome, and how many draws it took.

    Feasible means the adapter would simulate it: an infeasible one is given
    a penalty without running, so it would measure nothing.
    """
    for attempt in range(1, attempts + 1):
        candidate = adapter.random_individual_generator(1)[0]
        if adapter.penalty_objectives(candidate, max(1, n_objectives)) is None:
            return candidate, attempt
    raise ValueError(
        f"no feasible chromosome in {attempts} random draws; this problem's "
        "generator does not produce a simulable network, so there is nothing "
        "to benchmark")


class UslBenchmarkStrategy(EngineStrategy):
    """Runs the waves of a scalability benchmark, one generation each."""

    def __init__(self, experiment: dict, mongo):
        super().__init__(experiment, mongo)
        params = experiment.get("parameters", {}) or {}
        problem_config = params.get("problem", {}) or {}
        simulation_config = params.get("simulation", {}) or {}
        benchmark_config = params.get("benchmark", {}) or {}

        self._problem_name = str(problem_config.get("name", ""))
        self._sim_duration = int(simulation_config.get("duration", 120))
        self._rng = random.Random(int(params.get("random_seed", 42)))
        seeds = resolve_simulation_seeds(simulation_config, self._rng, default_count=1)
        # One seed for every simulation of every wave — the work must be
        # identical for the timing to measure only the infrastructure.
        self._seed = int(seeds[0]) if seeds else 123456
        self._n_objectives = len(params.get("objectives", []) or [])

        self._waves = build_waves(
            benchmark_config.get("levels") or [1],
            int(benchmark_config.get("repetitions") or DEFAULT_REPETITIONS))
        self._provided_chromosome = benchmark_config.get("chromosome")

        src_repo_opts = experiment.get("source_repository_options", {}) or {}
        self._source_options: dict[str, ObjectId] = {
            str(k): ObjectId(v) if isinstance(v, (str, bytes)) else v
            for k, v in src_repo_opts.items()
        }

        self._adapter: ProblemAdapter = build_adapter(problem_config, {}, self._rng)
        # `build_adapter` hands the RNG over only together with GA operator
        # settings, and a benchmark has none — so the generator would draw
        # from an attribute that was never set. Every adapter reads those
        # settings with defaults, so an empty mapping gives it the RNG and
        # nothing else.
        self._adapter.set_ga_operator_configs(self._rng, {})

        self._exp_id: Optional[ObjectId] = None
        self._workload: Optional[Chromosome] = None
        self._source_id: Optional[ObjectId] = None
        self._current_wave: int = -1
        self._current_generation: Optional[ObjectId] = None
        self._handled: set[ObjectId] = set()
        self._lock = threading.RLock()
        self._stop_flag = False
        self._threads: list[Thread] = []

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        raw_id = self.experiment.get("_id")
        self._exp_id = raw_id if isinstance(raw_id, ObjectId) else ObjectId(str(raw_id))

        existing = sorted(
            (g for g in self.mongo.generation_repo.find_by_experiment(self._exp_id)
             if g.get("benchmark")),
            key=lambda g: g["benchmark"].get("wave", 0))

        try:
            self._workload = self._load_or_choose_workload(resume=bool(existing))
        except Exception as error:
            logger.exception("[USL] Cannot choose a workload for %s", self._exp_id)
            self._finish(EnumStatus.ERROR, str(error))
            return

        self._start_watchers()
        with self._lock:
            if not existing:
                self.mongo.experiment_repo.update(str(self._exp_id), {
                    "status": EnumStatus.RUNNING, "start_time": utc_now()})
                self._enqueue(0)
                return
            # Resuming after a restart: pick up at the last wave written.
            last = existing[-1]
            self._current_wave = int(last["benchmark"].get("wave", len(existing) - 1))
            self._current_generation = last["_id"]
            logger.info("[USL] Resuming %s at wave %d of %d",
                        self._exp_id, self._current_wave + 1, len(self._waves))
            if last.get("status") in (EnumStatus.DONE, EnumStatus.ERROR):
                self._handle_generation_done(last["_id"])

    def event_simulation_done(self, sim_doc: dict):
        # Accounting only; the wave advances on its generation.
        return

    def event_generation_done(self, gen_doc: dict):
        gen = gen_doc.get("fullDocument") or {}
        if not gen or gen.get("experiment_id") != self._exp_id:
            return
        with self._lock:
            self._handle_generation_done(ObjectId(gen["_id"]))

    def stop(self):
        self._stop_flag = True
        for thread in self._threads:
            if thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._threads = []

    # ── the workload ───────────────────────────────────────────────────────

    def _load_or_choose_workload(self, *, resume: bool) -> Chromosome:
        """The chromosome every wave runs, persisted so a restart keeps it.

        Recorded on the experiment rather than re-derived from the seed:
        regenerating would give the same answer only while the problem's
        generator draws the same numbers in the same order, which is a
        property of code that may change between the run and the restart.
        """
        doc = self.mongo.experiment_repo.get(str(self._exp_id)) or {}
        stored = doc.get("benchmark_workload") or {}
        if stored.get("chromosome"):
            # Stored after the firmware was resolved, so resolving it again
            # returns it unchanged — and gives back the source id.
            return self._with_source(
                chromosome_from_dict(self._problem_name, stored["chromosome"]))
        if resume:
            raise ValueError(
                "this benchmark has waves but no recorded workload; resuming "
                "with a different chromosome would mix two measurements")

        if self._provided_chromosome:
            chromosome = chromosome_from_dict(self._problem_name, self._provided_chromosome)
            source, attempts = "provided", 0
            if self._adapter.penalty_objectives(
                    chromosome, max(1, self._n_objectives)) is not None:
                raise ValueError(
                    "the chromosome given for this benchmark is infeasible, so "
                    "the adapter would not simulate it")
        else:
            chromosome, attempts = choose_workload(
                self._adapter, self._rng, self._n_objectives)
            source = "generated"
        chromosome = self._with_source(chromosome)

        self.mongo.experiment_repo.update(str(self._exp_id), {"benchmark_workload": {
            "genome_hash": chromosome.get_hash(),
            "chromosome": chromosome.to_dict(),
            "source": source,
            "attempts": attempts,
            "seed": self._seed,
        }})
        logger.info("[USL] Workload for %s: %s chromosome %s (seed %d)",
                    self._exp_id, source, chromosome.get_hash(), self._seed)
        return chromosome

    def _with_source(self, chromosome: Chromosome) -> Chromosome:
        """The chromosome as it will actually run, with its firmware resolved.

        Resolving can *change* the chromosome: with a single firmware
        available, its MAC gene is rewritten to match it. Resolving once,
        here, and keeping what comes back is what keeps the recorded workload,
        every simulation's ``individual_id`` and the protocol actually
        simulated the same thing. Resolving per simulation and discarding the
        result recorded one chromosome and ran another.
        """
        resolved, source_id = chromosome.get_source_by_mac_protocol(self._source_options)
        self._source_id = source_id
        return resolved

    # ── waves ──────────────────────────────────────────────────────────────

    def _enqueue(self, wave: int) -> None:
        """Insert one wave: its generation first, then its simulations.

        The generation goes in before any simulation for the same reason as
        everywhere else in the engine: a fast worker can finish a simulation
        before the last one is inserted, and the close must land on a
        document that exists.
        """
        level, repetition = self._waves[wave]
        exp_oid = self._exp_id
        gen_oid = ObjectId()
        generation: Generation = {
            "_id": gen_oid,
            "experiment_id": exp_oid,
            "index": wave,
            "status": EnumStatus.WAITING,
            "start_time": utc_now(),
            "end_time": None,
            "benchmark": {"level": level, "repetition": repetition, "wave": wave},
        }
        self.mongo.generation_repo.insert(generation)
        self._current_wave = wave
        self._current_generation = gen_oid

        genome = self._workload
        genome_hash = genome.get_hash()
        individual: Individual = {
            "experiment_id": exp_oid,
            "generation_id": gen_oid,
            "individual_id": genome_hash,
            "chromosome": genome.to_dict(),
            "objectives": [],
            "topology_picture_id": None,
        }
        self.mongo.individual_repo.insert(individual)

        for copy in range(level):
            config = self._simulation_config(wave, copy)
            self._insert_simulation(genome, exp_oid, gen_oid, config)
        logger.info("[USL] %s wave %d/%d: %d simulation(s), repetition %d",
                    exp_oid, wave + 1, len(self._waves), level, repetition)

    def _simulation_config(self, wave: int, copy: int) -> SimulationConfig:
        return {
            "name": f"usl-w{wave}-c{copy}",
            "duration": self._sim_duration,
            "randomSeed": self._seed,
            "radiusOfReach": self._adapter.radius_of_reach,
            "radiusOfInter": self._adapter.radius_of_inter,
            "region": self._adapter.bounds,
            "simulationElements": self._adapter.encode_simulation_input(self._workload),
        }

    def _insert_simulation(self, genome: Chromosome, exp_oid: ObjectId,
                           gen_oid: ObjectId, config: SimulationConfig) -> ObjectId:
        # Files are created per simulation, as an optimisation run creates
        # them: the benchmark is only worth anything if it measures the same
        # path the real workload takes.
        files = create_files(config, self.mongo.fs_handler)
        simulation: Simulation = {
            "experiment_id": exp_oid,
            "generation_id": gen_oid,
            "individual_id": genome.get_hash(),
            "status": EnumStatus.WAITING,
            "random_seed": config.get("randomSeed", 0),
            "start_time": None,
            "end_time": None,
            "parameters": config,
            "pos_file_id": files.get("pos_file_id", ""),
            "csc_file_id": files.get("csc_file_id", ""),
            "source_repository_id": self._source_id,
            "log_cooja_id": "",
            "runtime_log_id": "",
            "csv_log_id": "",
            "network_metrics": {},
        }
        return self.mongo.simulation_repo.insert(simulation)

    def _handle_generation_done(self, gen_oid: ObjectId) -> None:
        """Advance to the next wave, once per wave.

        Called by the change stream, the polling fallback and the resume path,
        any of which may deliver the same generation twice; ``_handled`` is
        what makes the second delivery a no-op.
        """
        if gen_oid != self._current_generation or gen_oid in self._handled:
            return
        self._handled.add(gen_oid)

        if self._count_done(gen_oid) == 0:
            level, _ = self._waves[self._current_wave]
            self._finish(EnumStatus.ERROR, (
                f"wave {self._current_wave + 1} ran {level} simulation(s) and "
                "none produced a result. The workload does not run on this "
                "infrastructure, so the benchmark stopped instead of measuring "
                "failures."))
            return

        following = self._current_wave + 1
        if following < len(self._waves):
            self._enqueue(following)
            return
        self._finish(EnumStatus.DONE,
                     f"Scalability benchmark completed: {len(self._waves)} waves.")

    def _count_done(self, gen_oid: ObjectId) -> int:
        with self.mongo.simulation_repo.connection.connect() as db:
            return db["simulations"].count_documents(
                {"generation_id": gen_oid, "status": EnumStatus.DONE})

    def _finish(self, status: str, message: str) -> None:
        if self._exp_id is not None:
            self.mongo.experiment_repo.update(str(self._exp_id), {
                "status": status, "end_time": utc_now(), "system_message": message})
        log = logger.info if status == EnumStatus.DONE else logger.error
        log("[USL] %s: %s", self._exp_id, message)
        self.stop()

    # ── watchers ───────────────────────────────────────────────────────────

    def _start_watchers(self) -> None:
        self._stop_flag = False

        def watch() -> None:
            def callback(gen_doc: dict) -> None:
                if self._stop_flag:
                    return
                try:
                    self.event_generation_done(gen_doc)
                except Exception:
                    logger.exception("[USL] Generation watcher callback error")
            self.mongo.generation_repo.watch_status_terminal(callback)

        def poll() -> None:
            # The change stream is the fast path; this is the backstop that
            # keeps a missed event from stalling the benchmark.
            interval = int(os.getenv("BATCH_POLL_INTERVAL", "30"))
            while not self._stop_flag:
                time.sleep(interval)
                if self._stop_flag or self._current_generation is None:
                    continue
                try:
                    gen = self.mongo.generation_repo.get(str(self._current_generation))
                    if gen and gen.get("status") in (EnumStatus.DONE, EnumStatus.ERROR):
                        with self._lock:
                            self._handle_generation_done(ObjectId(gen["_id"]))
                except Exception:
                    logger.exception("[USL] Polling fallback error")

        for target, name in ((watch, "usl-gen-watcher"), (poll, "usl-poll")):
            thread = Thread(target=target, daemon=True, name=name)
            thread.start()
            self._threads.append(thread)
