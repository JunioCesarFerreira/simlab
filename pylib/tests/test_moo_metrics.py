"""Unit tests for the canonical convergence indicators (pylib/moo_metrics.py).

This is the definition the live ``GET /experiments/{id}/hv-gd`` endpoint serves.
``pareto-analysis/tests/test_metrics_parity.py`` separately pins the standalone
CLI mirror to these same numbers.
"""
import math

import numpy as np
import pytest

from pylib import moo_metrics as mm


# A ZDT1 front is convenient here: convex, densely sampled, and every point on
# it is mutually non-dominated.
def zdt1(n: int = 200) -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, n)
    return np.column_stack([f1, 1.0 - np.sqrt(f1)])


# ── reference-front sanitation ───────────────────────────────────────────────

class TestSanitizeReferenceFront:
    def test_drops_penalized_duplicate_and_dominated_rows(self):
        rows = [
            [1.0, 1.0],
            [1.0, 1.0],                     # duplicate
            [2.0, 2.0],                     # dominated by (1,1)
            [0.0, 5.0],
            [mm.PENALTY_THRESHOLD, 0.1],    # penalized
        ]
        out = mm.sanitize_reference_front(rows).tolist()
        assert sorted(out) == sorted([[1.0, 1.0], [0.0, 5.0]])

    def test_penalty_is_detected_on_either_sign(self):
        # Maximised objectives are negated into minimisation space before they
        # get here, so a penalty can arrive as a large NEGATIVE number.
        assert mm.sanitize_reference_front([[-1e9, 0.5], [1.0, 1.0]]).tolist() == [[1.0, 1.0]]

    def test_all_rows_penalized_yields_empty(self):
        assert mm.sanitize_reference_front([[1e9, 1e9]]).size == 0

    def test_empty_input_yields_empty(self):
        assert mm.sanitize_reference_front([]).size == 0

    def test_is_idempotent(self):
        once = mm.sanitize_reference_front([[1.0, 1.0], [2.0, 2.0], [0.0, 5.0]])
        assert np.array_equal(once, mm.sanitize_reference_front(once))

    def test_penalized_row_would_otherwise_wreck_igd(self):
        # IGD averages over the REFERENCE, so one unfiltered 1e9 row alone puts
        # the indicator in the 1e8 range. GD hides it — it minimises over the
        # reference instead — which is why the contamination went unnoticed
        # until IGD was reported.
        front = np.array([[1.0, 1.0]])
        dirty = np.array([[1.0, 1.0], [1e9, 1e9]])
        assert mm.igd(front, dirty, normalized=False) > 1e8
        assert mm.gd(front, dirty, normalized=False) == pytest.approx(0.0)

        clean = mm.sanitize_reference_front(dirty)
        assert mm.igd(front, clean, normalized=False) == pytest.approx(0.0)


# ── GD ───────────────────────────────────────────────────────────────────────

class TestGD:
    def test_is_the_p1_mean_not_the_rms_variant(self):
        # dists 0 and √2 → mean = √2/2 ≈ 0.707, RMS = 1.0.
        front = np.array([[0.0, 0.0], [1.0, 1.0]])
        ref = np.array([[0.0, 0.0]])
        assert mm.gd(front, ref, normalized=False) == pytest.approx(math.sqrt(2) / 2)

    def test_zero_when_front_lies_on_the_reference(self):
        ref = zdt1()
        assert mm.gd(ref, ref) == pytest.approx(0.0)

    def test_shrinks_as_the_front_approaches_the_reference(self):
        ref = zdt1()
        far = ref + 0.5
        near = ref + 0.05
        assert mm.gd(near, ref) < mm.gd(far, ref)


# ── IGD / IGD+ ───────────────────────────────────────────────────────────────

class TestIGD:
    def test_asymmetry_with_gd_is_what_makes_both_worth_showing(self):
        # A front collapsed onto a single optimal point has a perfect GD (it
        # converged) and a poor IGD (it did not spread).
        ref = zdt1()
        collapsed = ref[:1]
        assert mm.gd(collapsed, ref) == pytest.approx(0.0)
        assert mm.igd(collapsed, ref) > 0.5

    def test_zero_when_the_front_covers_the_reference(self):
        ref = zdt1()
        assert mm.igd(ref, ref) == pytest.approx(0.0)

    def test_improves_as_coverage_densifies(self):
        ref = zdt1()
        assert mm.igd(ref[::5], ref) < mm.igd(ref[::100], ref)

    def test_igd_plus_never_exceeds_igd(self):
        # d+ charges only the components where the solution is worse than the
        # reference point, so IGD+ ≤ IGD by construction.
        ref = zdt1()
        front = np.array([[0.2, 0.9], [0.8, 0.4], [0.5, 0.5]])
        assert mm.igd_plus(front, ref) <= mm.igd(front, ref)

    def test_igd_plus_rewards_a_dominating_front(self):
        # Moving the whole front to strictly better values can only improve a
        # Pareto-compliant indicator.
        ref = zdt1()
        worse = ref + 0.1
        better = ref + 0.02
        assert mm.igd_plus(better, ref) < mm.igd_plus(worse, ref)


# ── normalisation ────────────────────────────────────────────────────────────

class TestNormalization:
    REF = np.array([[0.0, 1000.0], [1.0, 0.0]])
    FRONT = np.array([[0.5, 900.0]])

    def test_is_on_by_default(self):
        assert mm.gd(self.FRONT, self.REF) == mm.gd(self.FRONT, self.REF, normalized=True)

    def test_rescales_axes_of_very_different_magnitude(self):
        # Raw, the 1000-wide f2 axis alone decides the distance.
        assert mm.gd(self.FRONT, self.REF, normalized=False) > 100.0
        assert mm.gd(self.FRONT, self.REF) < 1.0

    def test_invariant_to_a_uniform_rescale_of_one_objective(self):
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        front = np.array([[0.3, 0.8]])
        stretch = np.array([1.0, 1000.0])
        assert mm.gd(front, ref) == pytest.approx(mm.gd(front * stretch, ref * stretch))

    def test_degenerate_axis_falls_back_to_unit_scale(self):
        ideal, scale = mm.normalization_bounds(np.array([[3.0, 4.0]]))
        assert ideal.tolist() == [3.0, 4.0]
        assert scale.tolist() == [1.0, 1.0]
        # ...so a one-point reference stays a pure translation, not a div by 0.
        assert mm.gd(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])) == pytest.approx(5.0)

    def test_uses_the_reference_not_the_front_for_the_bounds(self):
        # The reference is fixed for a whole experiment, so the transform is
        # too — that is what keeps the per-generation curves comparable. A
        # front far outside the reference range must not move the bounds.
        ref = np.array([[0.0, 1.0], [1.0, 0.0]])
        ideal, scale = mm.normalization_bounds(ref)
        assert ideal.tolist() == [0.0, 0.0]
        assert scale.tolist() == [1.0, 1.0]


# ── contract ─────────────────────────────────────────────────────────────────

class TestContract:
    @pytest.mark.parametrize("fn", [mm.gd, mm.igd, mm.igd_plus])
    def test_empty_either_side_is_none(self, fn):
        pts = np.array([[1.0, 1.0]])
        assert fn(np.empty((0, 2)), pts) is None
        assert fn(pts, np.empty((0, 2))) is None

    @pytest.mark.parametrize("fn", [mm.gd, mm.igd, mm.igd_plus])
    def test_width_mismatch_raises(self, fn):
        with pytest.raises(ValueError):
            fn(np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0]]))

    @pytest.mark.parametrize("fn", [mm.gd, mm.igd, mm.igd_plus])
    def test_returns_a_plain_float(self, fn):
        # The endpoint serialises these straight to JSON, where a numpy scalar
        # would not survive.
        assert type(fn(zdt1(10), zdt1())) is float

    def test_inputs_are_not_mutated(self):
        front = np.array([[0.5, 900.0]])
        ref = np.array([[0.0, 1000.0], [1.0, 0.0]])
        mm.gd(front, ref)
        assert front.tolist() == [[0.5, 900.0]]
        assert ref.tolist() == [[0.0, 1000.0], [1.0, 0.0]]
