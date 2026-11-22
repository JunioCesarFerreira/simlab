"""A resumed random search is the run that was never interrupted.

Random search used to resume with an empty population, an empty archive, the
index of the last generation instead of the next one, and the random stream of
generation 0. Every test here kills the engine at one point of the loop, starts
a new strategy over what was persisted, and compares the result with a run that
nobody killed.

The real ``start()``, enqueue and generation handler run; only MongoDB, the
threads and the CSC files are replaced. Documents go through BSON, so nothing
survives a restart that MongoDB would not have kept.
"""
from __future__ import annotations

import hashlib
import math

import pytest
from bson import BSON, ObjectId

from lib.strategy import random_search
from lib.strategy.random_search import WORST_OBJECTIVE, RandomSearchStrategy
from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.generation import _completeness_of

GENERATIONS = 4        # G_0 .. G_4
POP = 6
KEYS = ["latency", "throughput"]
SEEDS = [11, 23]

_LINE = [{
    "path_segments": [("100 * t", "0")],
    "is_closed": False,
    "is_round_trip": False,
    "speed": 1.0,
    "time_step": 1.0,
}]
_RADIO = {
    "region": [-10.0, -10.0, 110.0, 10.0],
    "sink": [0.0, 0.0],
    "radius_of_reach": 12.0,
    "radius_of_inter": 15.0,
}
PROBLEMS = {
    "analytical": {"name": "problem0", "n": 4},
    "continuous": {
        **_RADIO, "name": "problem1", "mobile_nodes": _LINE,
        "number_of_relays": 5, "min_coverage_percentage": 30.0,
    },
    # A ring around a sink-rooted grid, with no coverage repair: some draws
    # leave part of the ring uncovered and are penalised.
    "infeasible": {
        "name": "problem2",
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink": [0.0, 0.0],
        "radius_of_reach": 45.0,
        "radius_of_inter": 54.0,
        "candidates": [
            [float(x), float(y)]
            for x in range(-90, 91, 30) for y in range(-90, 91, 30) if x or y
        ],
        "mobile_nodes": [{
            **_LINE[0],
            "path_segments": [("80 * np.cos(2 * np.pi * t)", "80 * np.sin(2 * np.pi * t)")],
            "is_closed": True,
        }],
        "min_coverage_percentage": 100.0,
    },
    # Four candidates: genomes repeat within and across generations.
    "repeated": {
        **_RADIO, "name": "problem2", "mobile_nodes": _LINE,
        "candidates": [[10.0, 0.0], [10.0, 3.0], [20.0, 0.0], [20.0, 3.0]],
        "min_coverage_percentage": 0.0,
    },
    "targets": {
        **_RADIO, "name": "problem3",
        "candidates": [[float(x), y] for x in range(10, 101, 10) for y in (-3.0, 3.0)],
        "targets": [[25.0, 5.0]], "k_required": 1, "radius_of_cover": 20.0,
    },
}


class Killed(BaseException):
    """The process dying: not an Exception, so no handler absorbs it."""


def _stored(document: dict) -> dict:
    return BSON.encode(document).decode()


class _Database:
    """The documents, plus the repository methods the strategy calls."""

    def __init__(self, experiment: dict):
        self.experiment = _stored(experiment)
        self.generations: list[dict] = []
        self.individuals: list[dict] = []
        self.simulations: list[dict] = []
        self.cache: dict[str, dict] = {}
        self.die_before_generation = None
        self.die_before_finish = False
        self.die_at_cache_write_of = None
        # (index, what, how many of them were written before the death)
        self.die_while_enqueueing = None

        self.experiment_repo = _ExperimentRepo(self)
        self.generation_repo = _GenerationRepo(self)
        self.individual_repo = _IndividualRepo(self)
        self.simulation_repo = _SimulationRepo(self)
        self.genome_cache_repo = _CacheRepo(self)
        self.fs_handler = None

    def generation(self, oid) -> dict:
        return next(g for g in self.generations if g["_id"] == oid)

    def simulations_of(self, oid) -> list[dict]:
        return [s for s in self.simulations if s["generation_id"] == oid]

    def enqueueing(self, oid, what, written=0):
        if self.die_while_enqueueing == (self.generation(oid)["index"], what, written):
            raise Killed


class _Repo:
    def __init__(self, db: _Database):
        self.db = db


class _ExperimentRepo(_Repo):
    def is_cancelled(self, _exp_id):
        return self.db.experiment["status"] == EnumStatus.CANCELLED

    def update(self, _exp_id, fields):
        if fields.get("status") == EnumStatus.DONE and self.db.die_before_finish:
            raise Killed
        self.db.experiment.update(_stored(fields))
        return True


class _GenerationRepo(_Repo):
    def insert(self, document):
        if document["index"] == self.db.die_before_generation:
            raise Killed
        if any(g["index"] == document["index"] for g in self.db.generations):
            raise RuntimeError("duplicate generation index")
        self.db.generations.append(_stored(document))
        return document["_id"]

    def get(self, oid):
        return _stored(self.db.generation(ObjectId(oid)))

    def find_by_experiment(self, _exp_id):
        return [_stored(g) for g in sorted(self.db.generations, key=lambda g: g["index"])]

    def record_plan(self, oid, seeds):
        self.db.enqueueing(oid, "plan")
        self.db.generation(oid)["expected_seeds"] = list(seeds)

    def update(self, oid, fields):
        self.db.generation(oid).update(_stored(fields))
        return True

    def mark_done(self, oid):
        self.db.enqueueing(oid, "close")
        self.db.generation(oid)["status"] = EnumStatus.DONE

    def pending_retry_round(self, _oid):
        return None

    def retries_used(self, _oid):
        return 0

    def any_simulation_active(self, oid):
        return any(s["status"] == EnumStatus.WAITING for s in self.db.simulations_of(oid))

    def replica_completeness(self, oid, metrics):
        runs: dict[str, list[dict]] = {}
        for sim in self.db.simulations_of(oid):
            runs.setdefault(sim["individual_id"], []).append(sim)
        planned = sorted(self.db.generation(oid)["expected_seeds"])
        return {h: dict(_completeness_of(r, metrics, planned)) for h, r in runs.items()}

    def get_simulations_metrics_by_individual(self, generation_id, metrics, aggregators):
        values: dict[str, dict[str, list[float]]] = {}
        for sim in self.db.simulations_of(generation_id):
            if sim["status"] != EnumStatus.DONE:
                continue
            found = values.setdefault(sim["individual_id"], {})
            for name in metrics:
                value = sim["network_metrics"].get(name, math.nan)
                if math.isfinite(value):
                    found.setdefault(name, []).append(value)
        return {h: {m: sum(v) / len(v) for m, v in found.items()} for h, found in values.items()}


class _IndividualRepo(_Repo):
    def insert(self, document):
        assert not any(
            i["generation_id"] == document["generation_id"]
            and i["individual_id"] == document["individual_id"]
            for i in self.db.individuals
        ), "duplicate individual"
        oid = document["generation_id"]
        written = sum(i["generation_id"] == oid for i in self.db.individuals)
        self.db.enqueueing(oid, "individual", written)
        self.db.individuals.append(_stored({"_id": ObjectId(), **document}))

    def find_by_generation(self, oid):
        return [_stored(i) for i in self.db.individuals if i["generation_id"] == oid]

    def find_grouped_by_experiment(self, _exp_id):
        # Reversed on purpose: MongoDB promises no natural order.
        grouped: dict[ObjectId, list[dict]] = {}
        for document in reversed(self.db.individuals):
            grouped.setdefault(document["generation_id"], []).append(_stored(document))
        return grouped

    def update_evaluation(self, individual_id, oid, objectives, completeness=None):
        for document in self.db.individuals:
            if document["generation_id"] == oid and document["individual_id"] == individual_id:
                document.update(_stored({"objectives": objectives, "completeness": completeness}))
        return True


class _SimulationRepo(_Repo):
    def insert(self, document):
        oid = document["generation_id"]
        self.db.enqueueing(oid, "simulation", len(self.db.simulations_of(oid)))
        self.db.simulations.append(_stored({"_id": ObjectId(), **document}))

    def find_terminal_by_generation(self, oid):
        return [s for s in self.db.simulations_of(oid) if s["status"] != EnumStatus.WAITING]


class _CacheRepo(_Repo):
    def insert(self, _exp_id, genome_hash, chromosome):
        assert genome_hash not in self.db.cache, "duplicate cache entry"
        self.db.cache[genome_hash] = _stored({
            "genome_hash": genome_hash, "chromosome": chromosome, "objectives": None,
        })

    def set_objectives(self, _exp_id, genome_hash, objectives, completeness=None):
        dying_at = self.db.die_at_cache_write_of
        if dying_at is not None and dying_at == len(self.db.generations) - 1:
            raise Killed
        self.db.cache[genome_hash].update(
            _stored({"objectives": objectives, "completeness": completeness}))

    def get_all_by_experiment(self, _exp_id):
        return [_stored(entry) for entry in self.db.cache.values()]


class _Run:
    """One experiment, the engine processes that serve it, and the workers."""

    def __init__(self, problem="repeated", *, seed=42, simulation=None, outcome=None):
        self.outcome = outcome
        self.db = _Database({
            "_id": ObjectId(),
            "status": EnumStatus.RUNNING,
            "source_repository_options": {"csma": ObjectId()},
            "parameters": {
                "strategy": "random_search",
                "problem": PROBLEMS[problem],
                "algorithm": {
                    "population_size": POP,
                    "number_of_generations": GENERATIONS,
                    "random_seed": seed,
                    "apply_coverage_repair": False,
                },
                "simulation": {
                    "synthetic": {"enabled": True, "bench": "DTLZ2"},
                    "random_seeds": SEEDS,
                    "replica_policy": "none",
                    **(simulation or {}),
                },
                "objectives": [
                    {"metric_name": KEYS[0], "goal": "min"},
                    {"metric_name": KEYS[1], "goal": "max"},
                ],
            },
        })
        self.boot(first=True)

    def boot(self, first=False):
        strategy = RandomSearchStrategy(_stored(self.db.experiment), self.db)
        for name in ("_start_watcher", "_start_generation_poll",
                     "_upload_topology_async", "_fire_generation_done"):
            setattr(strategy, name, lambda *_a, **_kw: None)
        if first:
            # So that a test can stop at generation 0 even when it closes at once.
            strategy._reconcile_current_generation = lambda: None
        self.strategy = strategy
        strategy.start()

    def restart(self):
        self.db.die_before_generation = None
        self.db.die_before_finish = False
        self.db.die_at_cache_write_of = None
        self.db.die_while_enqueueing = None
        # The engine resumes only what it finds Running.
        if self.db.experiment["status"] == EnumStatus.RUNNING:
            self.boot()

    @property
    def last(self) -> dict:
        return max(self.db.generations, key=lambda g: g["index"])

    def work(self):
        """The workers deliver whatever the last generation still waits for."""
        generation = self.last
        ok = True
        for sim in self.db.simulations_of(generation["_id"]):
            if sim["status"] == EnumStatus.WAITING:
                digest = hashlib.sha1(
                    f"{sim['individual_id']}:{sim['random_seed']}".encode()).digest()
                metrics = {k: 1.0 + digest[n] / 7 for n, k in enumerate(KEYS)}
                status = EnumStatus.DONE
                if self.outcome:
                    status, metrics = self.outcome(digest, metrics)
                sim.update(status=status, network_metrics=metrics)
            ok = ok and sim["status"] == EnumStatus.DONE
        generation["status"] = EnumStatus.DONE if ok else EnumStatus.ERROR

    def handle(self):
        self.strategy.event_generation_done({"fullDocument": {"_id": self.last["_id"]}})

    def finish(self):
        for _ in range(3 * (GENERATIONS + 1)):
            if self.db.experiment["status"] != EnumStatus.RUNNING:
                return self
            self.work()
            self.handle()
        raise AssertionError("the experiment never finished")

    def until(self, checkpoint: int, point: str):
        """Run up to ``checkpoint`` and die at ``point`` of that generation."""
        while self.last["index"] < checkpoint:
            self.work()
            self.handle()
        if point == "waiting":
            return self
        self.work()
        if point == "closed":
            return self
        if point == "evaluating":
            self.db.die_at_cache_write_of = checkpoint
        elif checkpoint == GENERATIONS:
            self.db.die_before_finish = True
        else:
            self.db.die_before_generation = checkpoint + 1
        try:
            self.handle()
        except Killed:
            pass
        else:
            # Only a generation with nothing new has no cache write to die at.
            assert point == "evaluating"
        return self

    def result(self) -> dict:
        """Everything a reader can see, free of the ids that differ per run."""
        index_of = {g["_id"]: g["index"] for g in self.db.generations}
        front = sorted(
            self.db.experiment.get("pareto_front", []),
            key=lambda item: repr(sorted(item["chromosome"].items())),
        )
        return {
            "status": self.db.experiment["status"],
            "message_warns": "front" in (self.db.experiment.get("system_message") or ""),
            "generations": sorted(index_of.values()),
            "individuals": [
                (index_of[i["generation_id"]], i["individual_id"], i["chromosome"],
                 i["objectives"], i.get("completeness"))
                for i in self.db.individuals
            ],
            "simulations": [
                (index_of[s["generation_id"]], s["individual_id"], s["random_seed"])
                for s in self.db.simulations
            ],
            "cache": self.db.cache,
            "front": front,
            "validity": self.db.experiment.get("pareto_front_validity"),
        }


@pytest.mark.parametrize("point", ["waiting", "closed", "evaluating", "evaluated"])
@pytest.mark.parametrize("checkpoint", [0, 2, GENERATIONS])
@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_resumed_run_matches_the_uninterrupted_one(problem, checkpoint, point):
    straight = _Run(problem).finish()
    interrupted = _Run(problem).until(checkpoint, point)

    interrupted.restart()

    assert interrupted.finish().result() == straight.result()


def test_scenarios_exercise_penalty_and_reuse():
    """Without this the parametrised test could compare trivial runs."""
    infeasible = _Run("infeasible").finish().db.individuals
    states = {(i.get("completeness") or {}).get("state") for i in infeasible}
    assert {"infeasible", "complete"} <= states

    repeated = _Run("repeated").finish().db.individuals
    assert any((i.get("completeness") or {}).get("reused") for i in repeated)


def test_several_restarts_in_one_run():
    straight = _Run("infeasible").finish()
    run = _Run("infeasible")
    for checkpoint in range(GENERATIONS + 1):
        while run.last["index"] < checkpoint:
            run.work()
            run.handle()
        run.restart()

    assert run.finish().result() == straight.result()


@pytest.mark.parametrize("point", ["waiting", "closed", "evaluating", "evaluated"])
@pytest.mark.parametrize("outcome", ["failed_replica", "not_finite"])
def test_a_refused_evaluation_stays_refused(outcome, point):
    """A placeholder of 1e12 must not come back from a restart as a measurement."""
    def deliver(digest, metrics):
        if digest[2] < 90:
            if outcome == "failed_replica":
                return EnumStatus.ERROR, {}
            return EnumStatus.DONE, {**metrics, KEYS[0]: math.nan}
        return EnumStatus.DONE, metrics

    straight = _Run("repeated", outcome=deliver).finish()
    refused = [
        i for i in straight.db.individuals
        if WORST_OBJECTIVE in [abs(v) for v in i["objectives"]]
    ]
    assert refused
    assert all(not i["completeness"]["objective_used"] for i in refused)

    run = _Run("repeated", outcome=deliver).until(0, point)
    run.restart()

    assert run.finish().result() == straight.result()


def test_a_generation_refused_for_a_missing_aggregate_stays_refused():
    """Accepted replicas with no finite value: refused by the strategy itself."""
    def deliver(digest, metrics):
        return EnumStatus.DONE, {**metrics, KEYS[0]: math.nan}

    simulation = {"incomplete_objective": "accept"}
    straight = _Run("repeated", simulation=simulation, outcome=deliver).finish()
    run = _Run("repeated", simulation=simulation, outcome=deliver).until(0, "evaluated")
    run.restart()

    assert run.finish().result() == straight.result()


def test_drawn_simulation_seeds_are_the_same_after_a_restart():
    simulation = {"random_seeds": [], "random_seeds_count": 3}
    straight = _Run("repeated", simulation=simulation).finish()
    run = _Run("repeated", simulation=simulation).until(1, "waiting")
    run.restart()

    assert run.finish().result() == straight.result()


@pytest.mark.parametrize("checkpoint", [0, 2])
def test_checkpoint_without_a_snapshot_continues_the_stream(checkpoint, caplog):
    """Written before the fix: no RNG state and no population index."""
    straight = _Run("infeasible").finish()
    run = _Run("infeasible").until(checkpoint, "closed")
    for generation in run.db.generations:
        generation.pop("rng_state", None)
    for individual in run.db.individuals:
        individual.pop("index", None)

    with caplog.at_level("WARNING"):
        run.restart()

    assert "RNG snapshot" in caplog.text
    assert run.finish().result() == straight.result()


def test_generation_that_closes_while_the_watcher_subscribes_is_processed():
    run = _Run("repeated").until(1, "waiting")
    strategy = RandomSearchStrategy(_stored(run.db.experiment), run.db)
    strategy._upload_topology_async = lambda *_a, **_kw: None
    strategy._fire_generation_done = lambda *_a, **_kw: None
    strategy._start_generation_poll = lambda: None
    # It ends after the generations were read and before anyone is listening.
    strategy._start_watcher = run.work

    strategy.start()

    assert run.last["index"] == 2


def test_resume_refuses_a_generation_without_individuals():
    run = _Run("repeated").until(1, "waiting")
    last = run.last["_id"]
    run.db.individuals = [i for i in run.db.individuals if i["generation_id"] != last]

    with pytest.raises(RuntimeError, match="no individuals"):
        run.restart()


def test_a_firmware_run_queues_the_restored_genomes_firmware(monkeypatch):
    """The restored population is what maps a genome to its firmware."""
    monkeypatch.setattr(random_search, "create_files", lambda *_a, **_kw: {
        "csc_file_id": ObjectId(), "pos_file_id": ObjectId(),
    })
    simulation = {"synthetic": {"enabled": False}}
    run = _Run("repeated", simulation=simulation).until(1, "waiting")
    run.restart()
    run.finish()

    source = run.db.experiment["source_repository_options"]["csma"]
    assert {s["source_repository_id"] for s in run.db.simulations} == {source}


@pytest.mark.parametrize("dies_at", [
    (2, "plan", 0),
    (2, "individual", 0),
    (2, "individual", 3),
    (2, "simulation", 0),
    (2, "simulation", 5),
])
@pytest.mark.parametrize("closed", [False, True], ids=["open", "closed_by_master"])
def test_a_generation_cut_while_enqueueing_is_not_passed_off_as_whole(dies_at, closed):
    """Finishing Done on fewer individuals or seeds than configured is the danger."""
    run = _Run("continuous")
    run.db.die_while_enqueueing = dies_at
    with pytest.raises(Killed):
        run.finish()
    if closed:
        run.last["status"] = EnumStatus.DONE

    with pytest.raises(RuntimeError, match="enqueu"):
        run.restart()

    assert run.db.experiment["status"] != EnumStatus.DONE
    assert "enqueu" in run.db.experiment["system_message"]


def test_a_generation_with_nothing_to_simulate_is_closed_on_resume():
    """Enqueued whole, killed before the engine could close it: no worker will."""
    straight = _Run("analytical").finish()
    run = _Run("analytical")
    run.db.die_while_enqueueing = (2, "close", 0)
    with pytest.raises(Killed):
        run.finish()

    run.restart()

    # Closed and processed by the engine: nothing else would have closed it.
    assert run.last["index"] == 3
    assert run.finish().result() == straight.result()
