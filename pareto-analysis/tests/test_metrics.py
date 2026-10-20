"""Unit tests for the Pareto-quality metric primitives used to evaluate
experiments (dominance, non-dominated sorting, minimization mapping,
reference point, and the GD / IGD / IGD+ convergence indicators).
"""
import math

import numpy as np
import pytest

from plot_pareto_results import (
    dominates,
    fast_nondominated_sort,
    to_minimization_array,
    compute_worst_point,
    nondominated_rows_min,
    generation_front_min,
    compute_convergence_metrics,
)
from lib import metrics
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


# ── reference-front sanitation ───────────────────────────────────────────────

class TestSanitizeReferenceFront:
    def test_drops_penalized_duplicate_and_dominated_rows(self):
        rows = [
            [1.0, 1.0],
            [1.0, 1.0],                                     # duplicate
            [2.0, 2.0],                                     # dominated
            [0.0, 5.0],
            [metrics.PENALTY_THRESHOLD, 0.1],               # penalized
        ]
        out = metrics.sanitize_reference_front(rows).tolist()
        assert sorted(out) == sorted([[1.0, 1.0], [0.0, 5.0]])

    def test_penalized_row_would_otherwise_wreck_igd(self):
        # IGD averages over the REFERENCE, so an unfiltered 1e9 row alone puts
        # the indicator in the 1e8 range. GD hides it (it minimizes over the
        # reference instead), which is why the bug can go unnoticed.
        front = np.array([[1.0, 1.0]])
        dirty = np.array([[1.0, 1.0], [1e9, 1e9]])
        assert metrics.igd(front, dirty, normalized=False) > 1e8

        clean = metrics.sanitize_reference_front(dirty)
        assert metrics.igd(front, clean, normalized=False) == pytest.approx(0.0)

    def test_all_rows_penalized_yields_empty(self):
        assert metrics.sanitize_reference_front([[1e9, 1e9]]).size == 0


# ── generational distance ────────────────────────────────────────────────────

class TestGD:
    def test_single_point_euclidean(self):
        # distance from (0,0) to (3,4) = 5; a one-point reference is degenerate
        # on every axis, so normalization is a pure translation and cannot
        # change the distance.
        gd = metrics.gd(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
        assert gd == pytest.approx(5.0)

    def test_mean_of_min_distances(self):
        # front (0,0),(1,1); ref {(0,0)} → dists 0 and sqrt(2)
        # GD = mean(0, √2) = √2 / 2.  The p=1 (mean) form is what the SimLab
        # API serves to the GUI, so the tool must report the same number — and
        # notably NOT the RMS variant sqrt(mean(d²)) = 1.
        gd = metrics.gd(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([[0.0, 0.0]]))
        assert gd == pytest.approx(math.sqrt(2) / 2)
        assert gd != pytest.approx(1.0)

    def test_zero_when_front_on_reference(self):
        pts = np.array([[1.0, 1.0], [2.0, 2.0]])
        assert metrics.gd(pts, pts) == pytest.approx(0.0)

    def test_empty_is_none(self):
        assert metrics.gd(np.empty((0, 2)), np.array([[1.0, 1.0]])) is None
        assert metrics.igd(np.array([[1.0, 1.0]]), np.empty((0, 2))) is None


# ── inverted generational distance ───────────────────────────────────────────

class TestIGD:
    REF = np.column_stack([np.linspace(0.0, 1.0, 200),
                           1.0 - np.sqrt(np.linspace(0.0, 1.0, 200))])

    def test_asymmetry_with_gd_on_a_collapsed_front(self):
        # One point sitting exactly ON the reference has a perfect GD but a
        # terrible IGD: it converged, it did not spread. Reporting only GD
        # would call this run a success, which is the whole reason IGD is
        # shown next to it.
        one = self.REF[:1]
        assert metrics.gd(one, self.REF) == pytest.approx(0.0)
        assert metrics.igd(one, self.REF) > 0.5

    def test_zero_when_front_covers_the_reference(self):
        assert metrics.igd(self.REF, self.REF) == pytest.approx(0.0)

    def test_igd_plus_never_exceeds_igd(self):
        # d+ only charges the components where the solution is worse than the
        # reference point, so IGD+ ≤ IGD by construction.
        front = np.array([[0.2, 0.9], [0.8, 0.4]])
        assert metrics.igd_plus(front, self.REF) <= metrics.igd(front, self.REF)

    def test_igd_improves_as_the_front_fills_in(self):
        sparse = self.REF[::100]
        dense = self.REF[::5]
        assert metrics.igd(dense, self.REF) < metrics.igd(sparse, self.REF)

    def test_width_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics.igd(np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0]]))


# ── normalization ────────────────────────────────────────────────────────────

class TestNormalization:
    def test_rescales_axes_of_very_different_magnitude(self):
        # f2 spans 1000x the range of f1. Unnormalized, f2 alone decides the
        # distance; normalized, both axes weigh the same.
        ref = np.array([[0.0, 1000.0], [1.0, 0.0]])
        front = np.array([[0.5, 900.0]])
        assert metrics.gd(front, ref, normalized=False) > 100.0
        assert metrics.gd(front, ref) < 1.0

    def test_is_the_default(self):
        ref = np.array([[0.0, 1000.0], [1.0, 0.0]])
        front = np.array([[0.5, 900.0]])
        assert metrics.gd(front, ref) == metrics.gd(front, ref, normalized=True)

    def test_degenerate_axis_is_a_pure_translation(self):
        # A single-point reference has zero range on every axis: the transform
        # must fall back to a unit scale rather than dividing by zero.
        ideal, scale = metrics.normalization_bounds(np.array([[3.0, 4.0]]))
        assert ideal.tolist() == [3.0, 4.0]
        assert scale.tolist() == [1.0, 1.0]
        assert np.isfinite(metrics.gd(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])))

    def test_invariant_to_a_uniform_rescale_of_an_objective(self):
        # Scaling one objective by a constant rescales the reference range by
        # the same constant, so the normalized indicator is unchanged.
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        front = np.array([[0.3, 0.8]])
        stretched_ref = ref * np.array([1.0, 1000.0])
        stretched_front = front * np.array([1.0, 1000.0])
        assert metrics.gd(front, ref) == pytest.approx(
            metrics.gd(stretched_front, stretched_ref)
        )


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
        # a subset of the true front has GD ~ 0 against the full true front —
        # but a non-zero IGD, because it covers only part of it.
        true = zdt1_front(500)
        sample = true[::25]
        assert metrics.gd(sample, true) == pytest.approx(0.0, abs=1e-9)
        assert metrics.igd(sample, true) > 0.0


# ── minimization-space non-dominated filter ──────────────────────────────────

class TestNondominatedRowsMin:
    def test_keeps_only_non_dominated(self):
        rows = [[1.0, 1.0], [2.0, 2.0], [0.0, 5.0]]
        keep = nondominated_rows_min(rows)
        assert sorted(keep) == sorted([[1.0, 1.0], [0.0, 5.0]])

    def test_duplicates_survive_together(self):
        # equal points never dominate each other, matching the GUI convention
        rows = [[1.0, 1.0], [1.0, 1.0]]
        assert len(nondominated_rows_min(rows)) == 2

    def test_empty_and_single(self):
        assert nondominated_rows_min([]) == []
        assert nondominated_rows_min([[1.0, 2.0]]) == [[1.0, 2.0]]


# ── per-generation front in minimization space ───────────────────────────────

class TestGenerationFrontMin:
    def test_negates_max_objective_and_dedups(self):
        inds = [
            {"objectives": {"latency": 1.0, "energy": 5.0}},   # max on energy
            {"objectives": {"latency": 1.0, "energy": 5.0}},   # exact duplicate
            {"objectives": {"latency": 2.0, "energy": 1.0}},   # dominated
        ]
        front = generation_front_min(inds, OBJ, [True, False])
        assert front == [[1.0, -5.0]]

    def test_empty_generation(self):
        assert generation_front_min([], OBJ, MIN) == []


# ── convergence metrics (HV, cumulative HV, GD, IGD, IGD+) ───────────────────

class TestComputeConvergenceMetrics:
    REF = np.array([[0.0, 0.0]])
    HV_REF = [10.0, 10.0]

    def _gen(self, *points):
        return [{"objectives": {"latency": x, "energy": y}} for x, y in points]

    def test_metrics_follow_sorted_generation_order(self):
        # Insertion order is deliberately scrambled: the metrics must be
        # reported against sorted generation indices, never insertion order.
        per_gen = {
            2: self._gen((1.0, 1.0)),
            0: self._gen((5.0, 5.0)),
            1: self._gen((3.0, 3.0)),
        }
        conv = compute_convergence_metrics(
            per_gen, OBJ, MIN, self.HV_REF, self.REF
        )
        assert conv.generations == [0, 1, 2]
        # HV grows as the front approaches the origin: gen 0 worst, gen 2 best.
        assert conv.hv[0] < conv.hv[1] < conv.hv[2]
        # Every distance indicator to the origin shrinks accordingly.
        assert conv.gd[0] > conv.gd[1] > conv.gd[2]
        assert conv.igd[0] > conv.igd[1] > conv.igd[2]
        assert conv.igd_plus[0] > conv.igd_plus[1] > conv.igd_plus[2]

    def test_cumulative_is_monotonic_and_dominates_per_generation(self):
        per_gen = {
            0: self._gen((1.0, 9.0)),
            1: self._gen((9.0, 1.0)),   # incomparable with gen 0
            2: self._gen((8.0, 8.0)),   # a regression
        }
        _, hv, hv_cum, _, _, _ = compute_convergence_metrics(
            per_gen, OBJ, MIN, self.HV_REF, self.REF
        )
        assert all(hv_cum[i] <= hv_cum[i + 1] + 1e-9 for i in range(len(hv_cum) - 1))
        assert all(c >= v - 1e-9 for c, v in zip(hv_cum, hv))
        # Gen 2 regressed on its own, but the best-so-far front must not shrink.
        assert hv[2] < hv_cum[2]
        # The accumulated front keeps both earlier corner points.
        assert hv_cum[1] > hv_cum[0]

    def test_empty_generation_carries_cumulative_and_gaps_gd(self):
        per_gen = {
            0: self._gen((2.0, 2.0)),
            1: [],
            2: self._gen((1.0, 1.0)),
        }
        conv = compute_convergence_metrics(
            per_gen, OBJ, MIN, self.HV_REF, self.REF
        )
        assert conv.generations == [0, 1, 2]
        assert conv.hv[1] == 0.0                    # nothing of its own to measure
        assert conv.hv_cumulative[1] == conv.hv_cumulative[0]   # best-so-far carried
        # A gap, not a fake zero — on every distance indicator.
        assert math.isnan(conv.gd[1])
        assert math.isnan(conv.igd[1])
        assert math.isnan(conv.igd_plus[1])
        assert not math.isnan(conv.gd[0])

    def test_result_is_tuple_compatible(self):
        # The named result must still unpack positionally, so the field order
        # is part of the contract.
        per_gen = {0: self._gen((1.0, 1.0))}
        gens, hv, hv_cum, gd, igd, igd_plus = compute_convergence_metrics(
            per_gen, OBJ, MIN, self.HV_REF, self.REF
        )
        conv = compute_convergence_metrics(per_gen, OBJ, MIN, self.HV_REF, self.REF)
        assert (gens, hv, hv_cum, gd, igd, igd_plus) == tuple(conv)
