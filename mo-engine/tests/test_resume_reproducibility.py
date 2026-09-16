"""Phase 3 of the NSGA metrics fix plan: a resumed run continues the
uninterrupted one exactly.

Finding 7 of ``experiments/nsga-metrics-audit`` had two halves:

  * the DEAP and pymoo NSGA-III backends drew from generators SimLab never
    seeded — five identical calls produced five different selections;
  * ``_restore_population_state`` loaded the previous generation's OFFSPRING as
    parents. Those documents are Q_{t-1}, not the survivors P_{t-1}: on a
    checkpoint of population 10 the audit measured 4 preserved parents lost.

Both are covered here, plus the piece that ties them together — the random
stream itself, snapshotted onto each generation document.
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock

import numpy as np
import pytest
from bson import BSON, ObjectId
from pylib.db.models.enums import EnumStatus

from pylib import benchmarks

from lib.strategy.library_rng import (
    derive_generator,
    derive_seed,
    dump_random_state,
    load_random_state,
    numpy_global_seed,
)
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga2_deap import NSGA2DeapStrategy
from lib.strategy.nsga2_pymoo import NSGA2PymooStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.nsga3_deap import NSGA3DeapStrategy
from lib.strategy.nsga3_pymoo import NSGA3PymooStrategy
from tests.regression.kernel import KernelConfig, build_strategy

BENCH, M, N_VARS, POP = "DTLZ2", 3, 6, 12
ALL_STRATEGIES = [
    NSGA2LoopStrategy, NSGA3LoopStrategy,
    NSGA2DeapStrategy, NSGA3DeapStrategy,
    NSGA2PymooStrategy, NSGA3PymooStrategy,
]


def _strategy(cls, seed=42, pop=POP):
    return cls({"parameters": {
        "problem": {"name": "problem0", "n": N_VARS},
        "simulation": {"synthetic": {"enabled": True, "bench": BENCH}, "random_seeds": [42]},
        "algorithm": {
            "population_size": pop, "number_of_generations": 20, "random_seed": seed,
            "prob_cx": 0.9, "prob_mt": 0.1, "per_gene_prob": 0.05,
            "eta_cx": 20, "eta_mt": 20, "divisions": 6,
        },
        "objectives": [{"metric_name": f"f{i + 1}", "goal": "min"} for i in range(M)],
    }}, None)


# ── The library backends now answer to the experiment seed ───────────────────

@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.__name__)
def test_selection_is_reproducible_for_a_seed(cls):
    front = benchmarks.true_front(BENCH, M, 60, seed=9).tolist()
    np.random.seed(123)   # a hostile global state, deliberately left dirty
    results = {
        tuple(_strategy(cls)._select_next_parents(list(range(60)), front))
        for _ in range(5)
    }
    assert len(results) == 1


@pytest.mark.parametrize("cls", [NSGA3LoopStrategy, NSGA3DeapStrategy, NSGA3PymooStrategy],
                         ids=lambda c: c.__name__)
def test_distinct_seeds_still_select_differently(cls):
    """Guards against 'reproducible' meaning 'the seed is ignored'.

    Only the NSGA-III variants: rank-and-crowding survival has no random step,
    so its selection is a pure function of the objective matrix.
    """
    front = benchmarks.true_front(BENCH, M, 60, seed=9).tolist()
    results = {
        tuple(_strategy(cls, seed=s)._select_next_parents(list(range(60)), front))
        for s in (1, 2, 3)
    }
    assert len(results) == 3


def test_deap_selection_leaves_the_global_numpy_stream_alone():
    """DEAP reaches for numpy.random directly; the engine shares its
    interpreter with other work, so the global stream must come back intact."""
    front = benchmarks.true_front(BENCH, M, 60, seed=9).tolist()
    np.random.seed(7)
    expected = np.random.random()

    np.random.seed(7)
    _strategy(NSGA3DeapStrategy)._select_next_parents(list(range(60)), front)
    assert np.random.random() == expected


# ── The random-state snapshot ────────────────────────────────────────────────

def test_random_state_survives_a_round_trip():
    source = random.Random(11)
    [source.random() for _ in range(37)]        # advance to a non-initial state
    snapshot = dump_random_state(source)

    restored = random.Random(999)
    assert load_random_state(restored, snapshot)
    assert [restored.random() for _ in range(10)] == [source.random() for _ in range(10)]


def test_snapshot_is_bson_safe():
    """Mongo stores lists and ints, not tuples."""
    snapshot = dump_random_state(random.Random(3))
    assert isinstance(snapshot["internal"], list)
    assert all(isinstance(v, int) for v in snapshot["internal"])


@pytest.mark.parametrize("snapshot", [None, {}, {"version": 3}, {"internal": "nope"}])
def test_unusable_snapshots_leave_the_generator_untouched(snapshot):
    rng = random.Random(5)
    expected = rng.getstate()
    assert load_random_state(rng, snapshot) is False
    assert rng.getstate() == expected


def test_library_seeds_come_from_the_engine_generator():
    """One generator drives everything, so one snapshot restores everything."""
    a, b = random.Random(4), random.Random(4)
    assert derive_seed(a) == derive_seed(b)
    assert derive_generator(a).random() == derive_generator(b).random()
    with numpy_global_seed(a):
        first = np.random.random()
    with numpy_global_seed(b):
        assert np.random.random() == first


# ── Resume: parents come from the survivors, on the same stream ──────────────

class _ResumeHarness:
    """Drives the REAL enqueue and evolution synchronously over a fake Mongo.

    Only two things are neutralised: ``_fire_generation_done``, which would run
    the loop from a worker thread, and the topology upload, which would spawn
    another. Everything the checkpoint depends on — the population index stamped
    on each individual, the RNG snapshot on the generation, the survivor set —
    is produced by the production code path.
    """

    def __init__(self, cls, seed=42):
        self.cls = cls
        self.seed = seed
        self.generations: list[dict] = []
        self.individuals: dict[ObjectId, list[dict]] = {}
        self.by_hash: dict[str, dict] = {}
        self.strategy = self._build()

    def _build(self):
        strategy = _strategy(self.cls, self.seed)
        strategy._exp_id = ObjectId()
        strategy.mongo = MagicMock()
        strategy.mongo.generation_repo.insert.side_effect = self._insert_generation
        strategy.mongo.generation_repo.set_survivors.side_effect = self._set_survivors
        strategy.mongo.individual_repo.insert.side_effect = self._insert_individual
        strategy.mongo.individual_repo.find_by_generation.side_effect = (
            lambda gid: list(self.individuals.get(gid, []))
        )
        strategy.mongo.individual_repo.find_by_experiment_and_ids.side_effect = (
            lambda _exp, ids: [self.by_hash[h] for h in ids if h in self.by_hash]
        )
        strategy._fire_generation_done = lambda *_a, **_kw: None
        strategy._upload_topology_async = lambda *_a, **_kw: None
        return strategy

    def _insert_generation(self, generation: dict):
        self.generations.append(BSON.encode(generation).decode())
        return generation["_id"]

    def _set_survivors(self, generation_id, hashes):
        for generation in self.generations:
            if generation["_id"] == generation_id:
                generation["survivors"] = list(hashes)
        return True

    def _insert_individual(self, document: dict):
        stored = dict(document)
        self.individuals.setdefault(stored["generation_id"], []).append(stored)
        self.by_hash.setdefault(stored["individual_id"], stored)
        return ObjectId()

    def start(self, generations: int):
        self.strategy._current_population = (
            self.strategy._problem_adapter.random_individual_generator(POP)
        )
        self.strategy._generation_enqueue()
        for _ in range(generations):
            self.strategy._evolution()

    def population_ids(self) -> list[list[str]]:
        return [
            [d["individual_id"] for d in self.individuals[g["_id"]]]
            for g in self.generations
        ]

    def resume(self):
        """A fresh strategy picking the checkpoint up, as ``start()`` would."""
        self.strategy = self._build()
        replay = self.strategy._resume_existing_generation(self.generations)
        assert replay == (self.generations[-1]["status"] == EnumStatus.DONE)
        return self.strategy


@pytest.mark.parametrize("cls", [NSGA2LoopStrategy, NSGA3LoopStrategy], ids=lambda c: c.__name__)
def test_resume_restores_the_surviving_population(cls):
    """Finding 7: the offspring were loaded as parents, losing kept parents."""
    harness = _ResumeHarness(cls)
    harness.start(generations=3)

    survivors = harness.generations[-2]["survivors"]
    offspring = {d["individual_id"] for d in harness.individuals[harness.generations[-2]["_id"]]}
    carried_over = [h for h in survivors if h not in offspring]
    assert carried_over, "the scenario must actually carry parents over to be meaningful"

    restored = harness.resume()
    assert [c.get_hash() for c in restored._parents] == survivors
    assert all(restored._map_genome_objectives.get(c) is not None for c in restored._parents)


@pytest.mark.parametrize("cls", ALL_STRATEGIES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("seed", [1, 42])
@pytest.mark.parametrize("checkpoint", [0, 1, 4])
@pytest.mark.parametrize("status", [EnumStatus.WAITING, EnumStatus.DONE])
def test_resumed_run_matches_the_uninterrupted_one(cls, seed, checkpoint, status):
    """Compare exact trajectories after a BSON round-trip, not just indicators.

    Checkpoints precede the first selection as well as follow several selections;
    seed 1 exposes pymoo's accumulated hyperplane state being lost on restart.
    """
    straight = _ResumeHarness(cls, seed)
    straight.start(generations=8)
    interrupted = _ResumeHarness(cls, seed)
    interrupted.start(generations=checkpoint)
    interrupted.generations[-1]["status"] = status
    strategy = interrupted.resume()
    for _ in range(8 - checkpoint):
        strategy._evolution()

    assert interrupted.population_ids() == straight.population_ids()
    assert [g.get("survivors") for g in interrupted.generations] == [
        g.get("survivors") for g in straight.generations
    ]
    assert strategy._ga_rng.getstate() == straight.strategy._ga_rng.getstate()



@pytest.mark.parametrize("cls", [NSGA2LoopStrategy, NSGA3LoopStrategy], ids=lambda c: c.__name__)
def test_resume_without_a_checkpoint_falls_back_and_warns(cls, caplog):
    """Experiments started before this change carry neither field."""
    harness = _ResumeHarness(cls)
    harness.start(generations=3)
    for generation in harness.generations:
        generation.pop("survivors", None)
        generation.pop("rng_state", None)

    with caplog.at_level("WARNING"):
        restored = harness.resume()

    assert len(restored._parents) == POP        # the offspring, as before
    messages = " ".join(record.message for record in caplog.records)
    assert "RNG snapshot" in messages
    assert "predates survivor sets" in messages


@pytest.mark.parametrize("cls", [NSGA2LoopStrategy, NSGA3LoopStrategy], ids=lambda c: c.__name__)
def test_resume_refuses_a_survivor_with_no_document(cls):
    """Better to stop than to silently resume on a truncated population."""
    harness = _ResumeHarness(cls)
    harness.start(generations=3)
    harness.generations[-2]["survivors"] = ["not-a-real-hash"] + harness.generations[-2]["survivors"]

    with pytest.raises(RuntimeError, match="no individual document"):
        harness.resume()


def test_numpy_global_seed_serializes_concurrent_calls():
    """Two experiments must see their own stream and restore the caller's state."""
    entered, attempted, second_entered, release = Event(), Event(), Event(), Event()
    np.random.seed(123)
    before = np.random.get_state()

    def first():
        with numpy_global_seed(random.Random(1)):
            entered.set()
            assert release.wait(5)
            return np.random.random(5)

    def second():
        attempted.set()
        with numpy_global_seed(random.Random(2)):
            second_entered.set()
            return np.random.random(5)

    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(first)
        assert entered.wait(5)
        b = pool.submit(second)
        try:
            assert attempted.wait(5)
            assert not second_entered.wait(0.1), "global RNG contexts overlapped"
        finally:
            release.set()
        for future, seed in ((a, 1), (b, 2)):
            expected = np.random.RandomState(derive_seed(random.Random(seed))).random(5)
            np.testing.assert_array_equal(future.result(timeout=5), expected)
    for actual, expected in zip(np.random.get_state(), before):
        np.testing.assert_array_equal(actual, expected)


def test_numpy_global_seed_restores_and_unlocks_after_exception():
    np.random.seed(321)
    before = np.random.get_state()
    with pytest.raises(RuntimeError, match="selection failed"):
        with numpy_global_seed(random.Random(1)):
            np.random.random(5)
            raise RuntimeError("selection failed")
    for actual, expected in zip(np.random.get_state(), before):
        np.testing.assert_array_equal(actual, expected)
    with ThreadPoolExecutor(1) as pool:
        def another_experiment():
            with numpy_global_seed(random.Random(2)):
                return np.random.random()
        expected = np.random.RandomState(derive_seed(random.Random(2))).random()
        assert pool.submit(another_experiment).result(timeout=5) == expected


@pytest.mark.parametrize("checkpoint", [0, 1, 4])
def test_pymoo_legacy_checkpoint_remains_readable(checkpoint, caplog):
    harness = _ResumeHarness(NSGA3PymooStrategy, seed=1)
    harness.start(generations=checkpoint)
    for generation in harness.generations:
        generation.pop("selection_state", None)
    with caplog.at_level("WARNING"):
        strategy = harness.resume()
    assert ("no normalization checkpoint" in caplog.text) == (checkpoint > 1)
    strategy._evolution()
    assert len(harness.generations) == checkpoint + 2


@pytest.mark.parametrize("corruption", ["version", "backend", "shape", "nan", "missing", "empty"])
def test_pymoo_rejects_invalid_normalization_checkpoint(corruption):
    harness = _ResumeHarness(NSGA3PymooStrategy, seed=1)
    harness.start(generations=4)
    snapshot = harness.generations[-1]["selection_state"]
    if corruption == "version":
        snapshot["version"] = 999
    elif corruption == "backend":
        snapshot["backend"] = "another_backend"
    elif corruption == "shape":
        snapshot["normalization"]["extreme_points"] = [[1.0]]
    elif corruption == "nan":
        snapshot["normalization"]["ideal_point"][0] = float("nan")
    elif corruption == "empty":
        snapshot["normalization"] = None
    else:
        del snapshot["normalization"]["worst_point"]
    with pytest.raises(RuntimeError, match="invalid pymoo normalization checkpoint"):
        harness.resume()


def test_pymoo_checkpoint_is_a_detached_snapshot():
    harness = _ResumeHarness(NSGA3PymooStrategy, seed=1)
    harness.start(generations=4)
    snapshot = harness.strategy._dump_selection_state()
    frozen = BSON.encode(snapshot)
    harness.strategy._evolution()
    assert BSON.encode(snapshot) == frozen
    restored = harness.resume()
    stored = BSON.encode(harness.generations[-1])
    restored._pymoo_survival.norm.ideal_point[:] = 0
    assert BSON.encode(harness.generations[-1]) == stored
