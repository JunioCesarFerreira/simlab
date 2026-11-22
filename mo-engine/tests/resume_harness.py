"""An experiment whose engine can be killed at any point, for the resume tests.

The real ``start()``, enqueue and generation handler of a strategy run; only
MongoDB, the threads and the CSC files are replaced. Documents go through
BSON, so nothing survives a restart that MongoDB would not have kept.
"""
from __future__ import annotations

import hashlib
import math

from bson import BSON, ObjectId

from lib.strategy.random_search import RandomSearchStrategy
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


def stored(document: dict) -> dict:
    return BSON.encode(document).decode()


class Database:
    """The documents, plus the repository methods the strategy calls."""

    def __init__(self, experiment: dict):
        self.experiment = stored(experiment)
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
    def __init__(self, db: Database):
        self.db = db


class _ExperimentRepo(_Repo):
    def is_cancelled(self, _exp_id):
        return self.db.experiment["status"] == EnumStatus.CANCELLED

    def update(self, _exp_id, fields):
        if fields.get("status") == EnumStatus.DONE and self.db.die_before_finish:
            raise Killed
        self.db.experiment.update(stored(fields))
        return True


class _GenerationRepo(_Repo):
    def insert(self, document):
        if document["index"] == self.db.die_before_generation:
            raise Killed
        if any(g["index"] == document["index"] for g in self.db.generations):
            raise RuntimeError("duplicate generation index")
        self.db.generations.append(stored(document))
        return document["_id"]

    def get(self, oid):
        return stored(self.db.generation(ObjectId(oid)))

    def find_by_experiment(self, _exp_id):
        return [stored(g) for g in sorted(self.db.generations, key=lambda g: g["index"])]

    def record_plan(self, oid, seeds):
        self.db.enqueueing(oid, "plan")
        self.db.generation(oid)["expected_seeds"] = list(seeds)

    def set_survivors(self, oid, individual_ids):
        self.db.generation(oid)["survivors"] = list(individual_ids)
        return True

    def update(self, oid, fields):
        self.db.generation(oid).update(stored(fields))
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
        self.db.individuals.append(stored({"_id": ObjectId(), **document}))

    def find_by_generation(self, oid):
        return [stored(i) for i in self.db.individuals if i["generation_id"] == oid]

    def find_by_experiment_and_ids(self, _exp_id, ids):
        return [stored(i) for i in self.db.individuals if i["individual_id"] in ids]

    def find_grouped_by_experiment(self, _exp_id):
        # Reversed on purpose: MongoDB promises no natural order.
        grouped: dict[ObjectId, list[dict]] = {}
        for document in reversed(self.db.individuals):
            grouped.setdefault(document["generation_id"], []).append(stored(document))
        return grouped

    def update_evaluation(self, individual_id, oid, objectives, completeness=None):
        for document in self.db.individuals:
            if document["generation_id"] == oid and document["individual_id"] == individual_id:
                document.update(stored({"objectives": objectives, "completeness": completeness}))
        return True


class _SimulationRepo(_Repo):
    def insert(self, document):
        oid = document["generation_id"]
        self.db.enqueueing(oid, "simulation", len(self.db.simulations_of(oid)))
        self.db.simulations.append(stored({"_id": ObjectId(), **document}))

    def find_terminal_by_generation(self, oid):
        return [s for s in self.db.simulations_of(oid) if s["status"] != EnumStatus.WAITING]


class _CacheRepo(_Repo):
    def insert(self, _exp_id, genome_hash, chromosome):
        assert genome_hash not in self.db.cache, "duplicate cache entry"
        self.db.cache[genome_hash] = stored({
            "genome_hash": genome_hash, "chromosome": chromosome, "objectives": None,
        })

    def set_objectives(self, _exp_id, genome_hash, objectives, completeness=None):
        dying_at = self.db.die_at_cache_write_of
        if dying_at is not None and dying_at == len(self.db.generations) - 1:
            raise Killed
        self.db.cache[genome_hash].update(
            stored({"objectives": objectives, "completeness": completeness}))

    def get_all_by_experiment(self, _exp_id):
        return [stored(entry) for entry in self.db.cache.values()]


class Run:
    """One experiment, the engine processes that serve it, and the workers."""

    def __init__(self, problem="repeated", *, strategy=RandomSearchStrategy, seed=42,
                 simulation=None, outcome=None, parameters=None):
        self.cls = strategy
        self.outcome = outcome
        self.db = Database({
            "_id": ObjectId(),
            "status": EnumStatus.RUNNING,
            "source_repository_options": {"csma": ObjectId()},
            "parameters": {
                "problem": PROBLEMS[problem],
                "algorithm": {
                    "population_size": POP,
                    "number_of_generations": GENERATIONS,
                    "random_seed": seed,
                    "apply_coverage_repair": False,
                    "prob_cx": 0.9,
                    "prob_mt": 0.3,
                    "divisions": 4,
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
                **(parameters or {}),
            },
        })
        self.boot(first=True)

    def boot(self, first=False):
        strategy = self.cls(stored(self.db.experiment), self.db)
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
        survivors = sorted(
            (g["index"], g.get("survivors", [])) for g in self.db.generations)
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
            "survivors": survivors,
            "validity": self.db.experiment.get("pareto_front_validity"),
        }
