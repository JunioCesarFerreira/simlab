"""P3's constraints hold, and what breaks them is penalised, not simulated.

The adapter defined no penalty, read no ``g_required``, ran the k-coverage
repair after the connectivity one (undoing it), and judged connectivity among
the sensors without the sink.
"""
from __future__ import annotations

import random

import pytest

from lib.problem.chromosomes import ChromosomeP3
from lib.problem.resolve import build_test_adapter
from lib.strategy.batch import BatchStrategy
from pylib.db.models.enums import EnumStatus
from tests.resume_harness import Run

# Sink at the origin, candidates every 10 m along x, a target at the far end
# that only the last candidate covers.
LINE = {
    "name": "problem3",
    "region": [-10.0, -20.0, 100.0, 20.0],
    "sink": [0.0, 0.0],
    "candidates": [[10.0 * i, 0.0] for i in range(1, 9)],
    "targets": [[85.0, 5.0]],
    "k_required": 1,
    "radius_of_reach": 12.0,
    "radius_of_inter": 15.0,
    "radius_of_cover": 10.0,
}


def adapter(problem=None, seed=1, **settings):
    a = build_test_adapter(problem or LINE)
    a.set_ga_operator_configs(random.Random(seed), {"per_gene_prob": 0.0, **settings})
    return a


def genome(mask, mac=0):
    return ChromosomeP3(mac_protocol=mac, mask=list(mask))


def feasible(a, mask) -> bool:
    return a.penalty_objectives(genome(mask), 2) is None


class TestRepairs:
    def test_mutation_leaves_the_network_connected_and_covered(self):
        a = adapter()

        child = a.mutate(genome([1, 0, 0, 0, 0, 0, 0, 0]))

        # Covering the target took the last candidate; reaching it, the rest.
        assert child.mask == [1] * 8
        assert feasible(a, child.mask)

    def test_crossover_too(self):
        a = adapter()
        parent = genome([1, 0, 0, 0, 0, 0, 0, 0])

        for child in a.crossover([parent, parent]):
            assert feasible(a, child.mask)

    @pytest.mark.parametrize("seed", range(20))
    def test_on_a_grid_every_child_is_feasible(self, seed):
        a = adapter(GRID, seed=seed, per_gene_prob=0.3)
        population = a.random_individual_generator(6)

        children = [c for p in zip(population, population[1:]) for c in a.crossover(list(p))]
        children += [a.mutate(p) for p in population]

        assert all(feasible(a, c.mask) for c in children)


GRID = {
    "name": "problem3",
    "region": [-150.0, -150.0, 150.0, 150.0],
    "sink": [0.0, 0.0],
    "candidates": [[float(x), float(y)] for x in range(-100, 101, 40)
                   for y in range(-100, 101, 40) if (x, y) != (0, 0)],
    "targets": [[20.0, 20.0], [-40.0, 10.0], [60.0, -30.0], [-90.0, -90.0]],
    "k_required": 2,
    "radius_of_reach": 45.0,
    "radius_of_inter": 55.0,
    "radius_of_cover": 45.0,
}


class TestFeasibility:
    def test_sensors_that_cannot_reach_the_sink_are_infeasible(self):
        a = adapter()
        # Connected among themselves, 50 m from the sink.
        assert not feasible(a, [0, 0, 0, 0, 1, 1, 1, 1])

    def test_sensors_joined_through_the_sink_are_feasible(self):
        problem = {**LINE, "candidates": [[-10.0, 0.0], [10.0, 0.0]], "targets": [[10.0, 5.0]]}
        a = adapter(problem)
        # 20 m apart, more than the radius, but each one 10 m from the sink.
        assert feasible(a, [1, 1])

    def test_an_uncovered_target_is_infeasible(self):
        a = adapter()
        assert not feasible(a, [1, 1, 1, 1, 1, 1, 1, 0])

    def test_the_random_generator_judges_with_the_sink(self):
        a = adapter(GRID, seed=4)
        assert all(feasible(a, c.mask) for c in a.random_individual_generator(20))


class TestDegree:
    PAIR = {**LINE, "candidates": [[10.0, 0.0], [20.0, 0.0]], "targets": [[15.0, 0.0]],
            "radius_of_cover": 10.0}

    def test_g_is_read(self):
        # sink - a - b: b has one neighbour.
        assert not feasible(adapter({**self.PAIR, "g_required": 2}), [1, 1])
        assert feasible(adapter({**self.PAIR, "g_required": 1}), [1, 1])

    def test_without_g_one_neighbour_is_enough(self):
        assert feasible(adapter(self.PAIR), [1, 1])

    @pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
    def test_g_must_be_a_positive_whole_number(self, value):
        with pytest.raises(ValueError, match="g_required"):
            build_test_adapter({**self.PAIR, "g_required": value})


class TestPenalty:
    def test_it_dominates_every_measurement_and_grows_with_the_violation(self):
        problem = {**LINE, "targets": [[85.0, 5.0], [45.0, 5.0], [25.0, 5.0]]}
        a = adapter(problem)

        one = a.penalty_objectives(genome([1, 1, 1, 1, 1, 0, 0, 0]), 3)     # misses 1 target
        two = a.penalty_objectives(genome([1, 1, 0, 0, 0, 0, 0, 0]), 3)     # misses 2
        cut = a.penalty_objectives(genome([0, 0, 0, 0, 0, 0, 0, 1]), 3)     # misses 2 and the sink

        assert one == [one[0]] * 3 and one[0] > 1e9
        assert one[0] < two[0] < cut[0] <= 2e9


def test_an_infeasible_individual_is_not_simulated():
    chromosomes = [{"mac_protocol": 0, "mask": [0] * 8}, {"mac_protocol": 0, "mask": [1] * 8}]
    run = Run("targets", strategy=BatchStrategy,
              parameters={"problem": LINE, "chromosomes": chromosomes}).finish()

    simulated = {s["individual_id"] for s in run.db.simulations}
    states = {i["individual_id"]: i["completeness"]["state"] for i in run.db.individuals}
    infeasible = {h for h, state in states.items() if state == "infeasible"}

    assert run.db.experiment["status"] == EnumStatus.DONE
    assert len(infeasible) == 1 and not infeasible & simulated
