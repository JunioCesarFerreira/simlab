"""Objective-sense conversion and margin-aware dominance.

The heuristic must never assume which metric is minimised: everything is
driven by the experiment's ``objectives[].goal`` list.
"""
import numpy as np
import pytest

from lib.adaptive.dominance import (
    dominated_by_any,
    dominates,
    dominates_clearly,
    goal_signs,
    lower_bound,
    non_dominated,
    objective_ranges,
    to_minimization,
    to_original,
    upper_bound,
)


class TestGoalConversion:
    def test_signs_from_goals(self):
        assert goal_signs(["min", "max", "min"]) == [1, -1, 1]
        assert goal_signs(["MIN", " Max "]) == [1, -1]

    def test_unknown_goal_is_rejected(self):
        with pytest.raises(ValueError):
            goal_signs(["min", "maximise"])

    def test_maximised_objective_is_negated(self):
        signs = goal_signs(["min", "min", "max"])
        assert to_minimization([12.0, 3.0, 80.0], signs) == [12.0, 3.0, -80.0]

    def test_conversion_round_trips(self):
        signs = goal_signs(["max", "min", "max", "max"])
        original = [1.5, -2.0, 30.0, 0.0]
        assert to_original(to_minimization(original, signs), signs) == original

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            to_minimization([1.0, 2.0], [1])

    def test_higher_throughput_wins_after_conversion(self):
        signs = goal_signs(["max"])
        fast = to_minimization([120.0], signs)
        slow = to_minimization([80.0], signs)
        assert dominates(fast, slow)
        assert not dominates(slow, fast)


class TestDominance:
    def test_plain_dominance(self):
        assert dominates([1.0, 2.0], [2.0, 3.0])
        assert dominates([1.0, 2.0], [1.0, 3.0])
        assert not dominates([1.0, 2.0], [1.0, 2.0])
        assert not dominates([1.0, 5.0], [2.0, 3.0])

    def test_clear_dominance_requires_the_full_margin(self):
        assert dominates_clearly([1.0, 1.0], [2.0, 2.0], margin=0.5)
        assert not dominates_clearly([1.0, 1.0], [2.0, 2.0], margin=1.5)
        # A tie is never clear domination, whatever the margin.
        assert not dominates_clearly([1.0, 1.0], [1.0, 1.0], margin=0.0)

    def test_clear_dominance_accepts_a_per_objective_margin(self):
        assert dominates_clearly([1.0, 1.0], [3.0, 1.5], margin=[2.0, 0.5])
        assert not dominates_clearly([1.0, 1.0], [3.0, 1.5], margin=[2.0, 0.6])

    @pytest.mark.parametrize("n_obj", [1, 2, 3, 7])
    def test_no_assumption_on_the_objective_count(self, n_obj):
        better = [0.0] * n_obj
        worse = [1.0] * n_obj
        assert dominates(better, worse)
        assert dominates_clearly(better, worse, margin=0.5)
        assert not dominates(worse, better)

    def test_dominated_by_any_scans_the_reference_front(self):
        front = [[0.0, 5.0], [5.0, 0.0], [2.0, 2.0]]
        assert dominated_by_any([3.0, 3.0], front, margin=0.5)
        assert not dominated_by_any([1.0, 1.0], front, margin=0.0)

    def test_non_dominated_indices(self):
        points = [[0.0, 5.0], [5.0, 0.0], [2.0, 2.0], [3.0, 3.0]]
        assert set(non_dominated(points)) == {0, 1, 2}


class TestBounds:
    def test_lower_bound_is_optimistic_and_upper_pessimistic(self):
        mean = [10.0, 20.0]
        sigma = [1.0, 4.0]
        low = lower_bound(mean, sigma, kappa=2.0)
        high = upper_bound(mean, sigma, kappa=2.0)

        assert list(low) == [8.0, 12.0]
        assert list(high) == [12.0, 28.0]
        assert dominates(list(low), list(high))

    def test_optimistic_bound_survives_where_the_mean_does_not(self):
        front = [[5.0, 5.0]]
        mean, sigma = [6.0, 6.0], [2.0, 2.0]

        assert dominated_by_any(mean, front)
        assert not dominated_by_any(lower_bound(mean, sigma, 1.0), front)

    def test_pessimistic_bound_is_used_for_provisional_screening(self):
        front = [[5.0, 5.0]]
        upper = upper_bound([6.0, 6.0], [2.0, 2.0], 1.0)
        assert dominated_by_any(upper, front)

    def test_objective_ranges_never_return_zero(self):
        ranges = objective_ranges([[1.0, 7.0], [1.0, 9.0]])
        assert list(ranges) == [1.0, 2.0]
        assert list(objective_ranges([])) == []

    def test_ranges_scale_a_relative_margin(self):
        points = [[100.0, 0.1], [200.0, 0.3]]
        margin = 0.05 * objective_ranges(points)
        assert np.allclose(margin, [5.0, 0.01])
