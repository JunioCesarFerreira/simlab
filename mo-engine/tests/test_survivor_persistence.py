"""Phase 1 of the NSGA metrics fix plan, driven through the real ``_evolution``.

Two findings of ``experiments/nsga-metrics-audit`` are locked here:

  * finding 1 — the population environmental selection keeps (P_t) is persisted
    on every generation, so quality indicators can be plotted over the set the
    search actually carries forward instead of over the offspring Q_t;
  * finding 8 — the last generation gets an environmental selection too, so the
    reported front is ND(P_final) of pop_size individuals and not ND of a union
    of up to 2·pop_size candidates that no selection ever ran on.

The loop runs synchronously: ``_generation_enqueue`` is replaced by a stub that
evaluates the analytical benchmark in place, which is the only piece a real run
gets from MongoDB and the simulation workers.
"""
from __future__ import annotations

import pytest
from bson import ObjectId

from pylib import benchmarks

from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from tests.regression.kernel import KernelConfig, build_strategy

POP_SIZE = 10
GENERATIONS = 2
BENCH, M, N_VARS = "DTLZ2", 3, 6


class _FakeGenerationRepo:
    def __init__(self):
        self.survivors: dict[ObjectId, list[str]] = {}

    def set_survivors(self, generation_id: ObjectId, individual_ids: list[str]) -> bool:
        self.survivors[generation_id] = list(individual_ids)
        return True


class _FakeMongo:
    def __init__(self):
        self.generation_repo = _FakeGenerationRepo()


class _Run:
    """One synchronous experiment, with everything the assertions need."""

    def __init__(self, cls):
        config = KernelConfig(
            algorithm="nsga2" if cls is NSGA2LoopStrategy else "nsga3",
            bench=BENCH, m=M, n=N_VARS,
            pop_size=POP_SIZE, generations=GENERATIONS,
        )
        self.strategy = build_strategy(config, seed=7)
        self.strategy.mongo = _FakeMongo()

        self.enqueued_indices: list[int] = []
        self.selected_at_indices: list[int] = []
        self.generation_index_of: dict[ObjectId, int] = {}
        self.finalized: list[dict] = []

        select = self.strategy._select_next_parents

        def _recording_select(population, objectives):
            self.selected_at_indices.append(self.strategy._gen_index - 1)
            return select(population, objectives)

        def _enqueue():
            """Stand in for the DB write + worker round-trip of one generation."""
            generation_id = ObjectId()
            self.generation_index_of[generation_id] = self.strategy._gen_index
            self.enqueued_indices.append(self.strategy._gen_index)
            self.strategy._generation_id = generation_id
            self.strategy._gen_index += 1
            for chromosome in self.strategy._current_population:
                self.strategy._map_genome_objectives[chromosome] = benchmarks.evaluate(
                    BENCH, chromosome.x, M
                )

        self.strategy._select_next_parents = _recording_select
        self.strategy._generation_enqueue = _enqueue
        self.strategy._finalize_experiment = lambda **kwargs: self.finalized.append(kwargs)

        self.strategy._current_population = (
            self.strategy._problem_adapter.random_individual_generator(POP_SIZE)
        )
        _enqueue()
        while not self.finalized:
            assert len(self.enqueued_indices) < 10, "evolution loop did not terminate"
            self.strategy._evolution()

    @property
    def survivors_by_index(self) -> dict[int, list[str]]:
        return {
            self.generation_index_of[gid]: ids
            for gid, ids in self.strategy.mongo.generation_repo.survivors.items()
        }


@pytest.fixture(params=[NSGA2LoopStrategy, NSGA3LoopStrategy], ids=lambda c: c.__name__)
def run(request):
    return _Run(request.param)


def test_every_generation_records_its_survivors(run: _Run):
    assert sorted(run.survivors_by_index) == run.enqueued_indices


def test_survivor_set_is_the_selected_population(run: _Run):
    """One entry per selected individual, de-duplicated.

    The count can fall below ``POP_SIZE``: a child that reproduces a surviving
    parent exactly enters the union twice, and selection may keep both slots.
    That happens in roughly 4% of slots for both algorithms and predates this
    work, so the survivor set records distinct chromosomes rather than slots.
    """
    for index, hashes in run.survivors_by_index.items():
        assert 0 < len(hashes) <= POP_SIZE, f"generation {index} kept {len(hashes)} of {POP_SIZE}"
        assert len(set(hashes)) == len(hashes), f"generation {index} recorded a duplicate"


def test_last_generation_gets_an_environmental_selection(run: _Run):
    """Finding 8: selection used to stop one generation short of the last."""
    assert run.selected_at_indices == run.enqueued_indices[1:]


def test_reported_front_comes_from_the_selected_population(run: _Run):
    front = run.finalized[0]["pareto_front"]
    assert 0 < len(front) <= POP_SIZE, (
        f"front of {len(front)} points cannot come from a selected population "
        f"of {POP_SIZE} — it is the unselected union"
    )
    selected = {c.get_hash() for c in run.strategy._parents}
    assert set(run.survivors_by_index[run.enqueued_indices[-1]]) == selected


def test_survivors_survive_a_repository_failure(run: _Run):
    """Survivor sets are analysis metadata: a failed write must not abort a run."""

    def _fail(*_args, **_kwargs):
        raise RuntimeError("mongo is down")

    run.strategy.mongo.generation_repo.set_survivors = _fail
    run.strategy._persist_survivors()  # must not propagate
