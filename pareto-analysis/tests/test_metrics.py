"""Unit tests for the Pareto-quality metric primitives used to evaluate
experiments (dominance, non-dominated sorting, minimization mapping,
reference point, and Generational Distance).
"""
import math

import numpy as np
import pytest

from plot_pareto_results import (
    dominates,
    fast_nondominated_sort,
    to_minimization_array,
    compute_worst_point,
    compute_gd,
)
from lib.true_fronts import sample_true_front, dtlz2_front, zdt1_front, sch1_front

OBJ = ["latency", "energy"]
MIN = [True, True]


# ── dominance ────────────────────────────────────────────────────────────────

class TestDominates:
    def test_strict_domination(self):
        a = {"latency": 1.0, "energy": 1.0}
        b = {"latency": 2.0, "energy": 2.0}
        assert dominates(a, b, OBJ, MIN) is True
        assert dominates(b, a, OBJ, MIN) is False

    def test_incomparable_pair(self):
        a = {"latency": 1.0, "energy": 3.0}
        b = {"latency": 3.0, "energy": 1.0}
        assert dominates(a, b, OBJ, MIN) is False
        assert dominates(b, a, OBJ, MIN) is False

    def test_equal_is_not_domination(self):
        a = {"latency": 1.0, "energy": 1.0}
        assert dominates(a, dict(a), OBJ, MIN) is False

    def test_maximization_orientation(self):
        # higher is better on 'energy'
        a = {"latency": 1.0, "energy": 5.0}
        b = {"latency": 1.0, "energy": 2.0}
        assert dominates(a, b, OBJ, [True, False]) is True

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            dominates({"latency": 1}, {"latency": 2}, ["latency", "energy"], [True])


# ── non-dominated sorting ────────────────────────────────────────────────────

class TestFastNonDominatedSort:
    def test_two_fronts(self):
        pop = [
            {"id": "A", "objectives": {"latency": 1.0, "energy": 1.0}},
            {"id": "D", "objectives": {"latency": 0.0, "energy": 5.0}},
            {"id": "B", "objectives": {"latency": 2.0, "energy": 2.0}},
            {"id": "C", "objectives": {"latency": 1.0, "energy": 3.0}},
        ]
        fronts = fast_nondominated_sort(pop, OBJ, MIN)
        assert {p["id"] for p in fronts[0]} == {"A", "D"}
        assert {p["id"] for p in fronts[1]} == {"B", "C"}

    def test_single_point(self):
        pop = [{"id": "A", "objectives": {"latency": 1.0, "energy": 1.0}}]
        fronts = fast_nondominated_sort(pop, OBJ, MIN)
        assert len(fronts[0]) == 1


# ── minimization mapping ─────────────────────────────────────────────────────

class TestToMinimizationArray:
    def test_negates_maximization_columns(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = to_minimization_array(pts, OBJ, [True, False])
        assert out.tolist() == [[1.0, -2.0], [3.0, -4.0]]

    def test_all_minimize_is_identity(self):
        pts = np.array([[1.0, 2.0]])
        out = to_minimization_array(pts, OBJ, [True, True])
        assert out.tolist() == [[1.0, 2.0]]

    def test_does_not_mutate_input(self):
        pts = np.array([[1.0, 2.0]])
        to_minimization_array(pts, OBJ, [True, False])
        assert pts.tolist() == [[1.0, 2.0]]

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            to_minimization_array(np.array([[1.0, 2.0, 3.0]]), OBJ, MIN)


# ── worst point (HV reference) ───────────────────────────────────────────────

class TestComputeWorstPoint:
    def test_max_per_axis_in_min_space(self):
        per_gen = {
            0: [
                {"objectives": {"latency": 1.0, "energy": 2.0}},
                {"objectives": {"latency": 3.0, "energy": 1.0}},
            ]
        }
        worst = compute_worst_point(per_gen, tuple(OBJ), MIN)
        assert worst == [3.0, 2.0]


# ── generational distance ────────────────────────────────────────────────────

class TestComputeGD:
    def test_single_point_euclidean(self):
        # distance from (0,0) to (3,4) = 5
        gd = compute_gd(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
        assert gd == pytest.approx(5.0)

    def test_rms_of_min_distances(self):
        # front (0,0),(1,1); ref {(0,0)} → dists 0 and sqrt(2)
        # GD = sqrt(mean(0², (√2)²)) = sqrt((0+2)/2) = 1
        gd = compute_gd(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0]]))
        assert gd == pytest.approx(1.0)

    def test_zero_when_front_on_reference(self):
        pts = np.array([[1.0, 1.0], [2.0, 2.0]])
        assert compute_gd(pts, pts) == pytest.approx(0.0)

    def test_empty_is_inf(self):
        assert math.isinf(compute_gd(np.empty((0, 2)), np.array([[1.0, 1.0]])))


# ── analytical true fronts ───────────────────────────────────────────────────

class TestTrueFronts:
    def test_dtlz2_on_unit_sphere_positive_orthant(self):
        f = dtlz2_front(3, 200)
        assert f.shape == (200, 3)
        assert np.allclose(np.linalg.norm(f, axis=1), 1.0)
        assert (f >= 0).all()

    def test_dtlz2_requires_m_at_least_2(self):
        with pytest.raises(ValueError):
            dtlz2_front(1)

    def test_dtlz2_is_deterministic(self):
        assert np.array_equal(dtlz2_front(3, 50), dtlz2_front(3, 50))

    def test_zdt1_curve(self):
        f = zdt1_front(100)
        assert np.allclose(f[:, 1], 1.0 - np.sqrt(f[:, 0]))

    def test_sch1_vertices(self):
        f = sch1_front(100)
        assert np.allclose(f[0], [0.0, 4.0])
        assert np.allclose(f[-1], [4.0, 0.0])

    def test_dispatch_case_insensitive(self):
        assert sample_true_front("zdt1", 2).shape[1] == 2

    def test_dispatch_unknown_raises(self):
        with pytest.raises(ValueError):
            sample_true_front("nope", 2)

    def test_gd_zero_when_front_sampled_from_true_front(self):
        # a subset of the true front has GD ~ 0 against the full true front
        true = zdt1_front(500)
        sample = true[::25]
        assert compute_gd(sample, true) == pytest.approx(0.0, abs=1e-9)
