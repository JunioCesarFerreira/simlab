"""Tests for the canonical analytical benchmarks (pylib/benchmarks.py).

Locks the reference values (so master-node / rest-api / pareto-analysis cannot
drift), the structural metadata, the reproducible+clamped noise model, and the
true fronts / nadir points.
"""
import math
import random

import numpy as np
import pytest

from pylib import benchmarks as bm


# ── Objective functions (reference values) ───────────────────────────────────

class TestDTLZ2:
    def test_on_front_unit_sphere(self):
        f = bm.dtlz2([0.5] * 6, 3)
        assert len(f) == 3
        assert math.sqrt(sum(v * v for v in f)) == pytest.approx(1.0, abs=1e-9)

    def test_g_offset_inflates_radius(self):
        assert math.sqrt(sum(v * v for v in bm.dtlz2([0.5, 0.5, 0.5, 0.5, 0.5, 1.0], 3))) > 1.0

    def test_objective_count_matches_M(self):
        assert len(bm.dtlz2([0.5] * 5, 2)) == 2
        assert len(bm.dtlz2([0.5] * 5, 4)) == 4

    def test_raises_when_n_below_M_minus_1(self):
        with pytest.raises(ValueError):
            bm.dtlz2([0.5], 3)  # n=1 < M-1=2


class TestZDT1:
    def test_on_front(self):
        f1, f2 = bm.zdt1([0.25, 0.0, 0.0])
        assert f1 == pytest.approx(0.25)
        assert f2 == pytest.approx(1.0 - math.sqrt(0.25))

    def test_no_nan_when_f1_exceeds_g(self):
        assert math.isfinite(bm.zdt1([1.0])[1])

    def test_empty(self):
        assert bm.zdt1([]) == [1.0, 1.0]


class TestSCH1:
    def test_default_domain_is_wide(self):
        # Default domain (-5,5) is a convergence test: x01=0.5 → x=0 → (0,4);
        # the Pareto set x∈[0,2] is a strict subset the optimiser must reach.
        assert bm.sch1([0.5]) == [0.0, 4.0]      # x = 0  (on the front)
        assert bm.sch1([0.0]) == [25.0, 49.0]    # x = -5 (far off the front)
        assert bm.sch1([1.0]) == [25.0, 9.0]     # x = +5 (far off the front)

    def test_explicit_narrow_domain_maps_linearly(self):
        # The classic spread-only (0,2) mapping remains available explicitly.
        assert bm.sch1([0.0], domain=(0.0, 2.0)) == [0.0, 4.0]
        assert bm.sch1([0.5], domain=(0.0, 2.0)) == [1.0, 1.0]
        assert bm.sch1([1.0], domain=(0.0, 2.0)) == [4.0, 0.0]


# ── Structural metadata & validation ─────────────────────────────────────────

class TestMetadata:
    def test_min_variables(self):
        assert bm.min_variables("DTLZ2", 3) == 2
        assert bm.min_variables("ZDT1", 2) == 2
        assert bm.min_variables("SCH1", 2) == 1

    def test_validate_raises_below_minimum(self):
        with pytest.raises(ValueError):
            bm.validate("DTLZ2", 1, 3)
        bm.validate("DTLZ2", 2, 3)  # ok, no raise

    def test_all_known_non_negative(self):
        assert all(bm.is_non_negative(b) for b in ("DTLZ2", "ZDT1", "SCH1"))

    def test_unknown_benchmark_raises(self):
        with pytest.raises(ValueError):
            bm.min_variables("NOPE", 2)


# ── Dispatch + noise ─────────────────────────────────────────────────────────

class TestEvaluate:
    def test_dispatch_case_insensitive(self):
        assert bm.evaluate("dtlz2", [0.5] * 6, 3) == bm.evaluate("DTLZ2", [0.5] * 6, 3)

    def test_sch1_center_on_front(self):
        # Wide default domain: x01=0.5 → x=0 → the (0,4) front vertex.
        assert bm.evaluate("SCH1", [0.5], 2) == [0.0, 4.0]
        # The opposite vertex (4,0) is reachable at x=2 (x01=0.7).
        assert bm.evaluate("SCH1", [0.7], 2) == pytest.approx([4.0, 0.0], abs=1e-9)


class TestNoise:
    def test_zero_noise_is_deterministic(self):
        rng = random.Random(0)
        assert bm.evaluate_noisy("ZDT1", [0.3, 0.1], 2, 0.0, rng) == bm.evaluate("ZDT1", [0.3, 0.1], 2)

    def test_reproducible_for_same_seed(self):
        a = bm.evaluate_noisy("ZDT1", [0.3, 0.1], 2, 0.5, random.Random(42))
        b = bm.evaluate_noisy("ZDT1", [0.3, 0.1], 2, 0.5, random.Random(42))
        assert a == b

    def test_different_seed_differs(self):
        a = bm.evaluate_noisy("ZDT1", [0.3, 0.1], 2, 0.5, random.Random(1))
        b = bm.evaluate_noisy("ZDT1", [0.3, 0.1], 2, 0.5, random.Random(2))
        assert a != b

    def test_clamped_non_negative(self):
        # Large negative-leaning noise must never push a non-negative objective below 0.
        vals = [bm.evaluate_noisy("DTLZ2", [0.5] * 6, 3, 100.0, random.Random(s)) for s in range(50)]
        assert all(v >= 0.0 for row in vals for v in row)


# ── True fronts + nadir ──────────────────────────────────────────────────────

class TestTrueFront:
    def test_dtlz2_on_unit_sphere(self):
        front = bm.true_front("DTLZ2", 3, n_points=200)
        assert front.shape == (200, 3)
        assert np.allclose(np.linalg.norm(front, axis=1), 1.0, atol=1e-9)

    def test_dtlz2_deterministic(self):
        assert np.array_equal(bm.true_front("DTLZ2", 3), bm.true_front("DTLZ2", 3))

    def test_zdt1_relation(self):
        front = bm.true_front("ZDT1", 2, n_points=100)
        assert np.allclose(front[:, 1], 1.0 - np.sqrt(front[:, 0]))

    def test_sch1_endpoints(self):
        front = bm.true_front("SCH1", 2, n_points=100)
        assert np.allclose(front[0], [0.0, 4.0])
        assert np.allclose(front[-1], [4.0, 0.0])

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            bm.true_front("NOPE", 2)


class TestNadir:
    def test_values(self):
        assert bm.nadir("DTLZ2", 3) == [1.0, 1.0, 1.0]
        assert bm.nadir("ZDT1", 2) == [1.0, 1.0]
        assert bm.nadir("SCH1", 2) == [4.0, 4.0]

    def test_nadir_dominates_true_front(self):
        # Every true-front point must be <= the nadir componentwise.
        for b, M in (("DTLZ2", 3), ("ZDT1", 2), ("SCH1", 2)):
            front = bm.true_front(b, M, n_points=100)
            assert np.all(front <= np.array(bm.nadir(b, M)) + 1e-9)
