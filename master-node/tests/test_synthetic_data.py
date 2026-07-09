"""Unit tests for the synthetic benchmark evaluation (lib/synthetic_data.py).

These lock in the correctness fixes for synthetic mode:
  * the sink is excluded from the genome (decision vars = relay coords);
  * region normalization respects a custom Ω;
  * DTLZ2 lands on the unit hypersphere on the Pareto front;
  * ZDT1 / SCH1 match their analytical definitions.
"""
import math

import pytest

from lib.synthetic_data import (
    _extract_genome_from_sim,
    _scale_to_unit,
    _dtlz2,
    _zdt1,
    _sch1,
    _eval_benchmark,
    resolve_synthetic_settings,
)

REGION = (-100.0, -100.0, 100.0, 100.0)


# ── Genome extraction ────────────────────────────────────────────────────────

class TestExtractGenome:
    def test_excludes_sink(self):
        sim = {"parameters": {"simulationElements": {"fixedMotes": [
            {"name": "sink", "position": [50.0, 50.0]},
            {"name": "relay_0", "position": [10.0, 20.0]},
            {"name": "relay_1", "position": [-30.0, 40.0]},
        ]}}}
        # sink dropped → 2 relays → 4 floats
        assert _extract_genome_from_sim(sim) == [10.0, 20.0, -30.0, 40.0]

    def test_empty_when_only_sink(self):
        sim = {"parameters": {"simulationElements": {"fixedMotes": [
            {"name": "sink", "position": [0.0, 0.0]},
        ]}}}
        assert _extract_genome_from_sim(sim) == []

    def test_missing_structure_is_empty(self):
        assert _extract_genome_from_sim({}) == []


# ── Region normalization ─────────────────────────────────────────────────────

class TestScaleToUnit:
    def test_center_maps_to_half(self):
        assert _scale_to_unit([0.0, 0.0], REGION) == [0.5, 0.5]

    def test_custom_region_respected(self):
        # region [0,0,200,200]: point (50,150) → (0.25, 0.75)
        assert _scale_to_unit([50.0, 150.0], (0.0, 0.0, 200.0, 200.0)) == [0.25, 0.75]

    def test_clipping_outliers(self):
        # points outside Ω clip to [0,1]
        assert _scale_to_unit([1000.0, -1000.0], REGION) == [1.0, 0.0]


# ── DTLZ2 ────────────────────────────────────────────────────────────────────

class TestDTLZ2:
    def test_on_front_is_unit_sphere(self):
        # tail variables = 0.5 → g = 0 → point lies on unit hypersphere ‖f‖₂ = 1
        x = [0.5] * 6
        f = _dtlz2(x, 3)
        assert len(f) == 3
        assert math.sqrt(sum(v * v for v in f)) == pytest.approx(1.0, abs=1e-9)

    def test_g_offset_scales_radius(self):
        # a tail variable off 0.5 inflates the radius above 1
        x = [0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
        f = _dtlz2(x, 3)
        assert math.sqrt(sum(v * v for v in f)) > 1.0

    def test_objective_count_matches_M(self):
        assert len(_dtlz2([0.5] * 5, 2)) == 2
        assert len(_dtlz2([0.5] * 5, 4)) == 4


# ── ZDT1 ─────────────────────────────────────────────────────────────────────

class TestZDT1:
    def test_on_front(self):
        # tail variables = 0 → g = 1 → f2 = 1 - sqrt(f1)
        f1, f2 = _zdt1([0.25, 0.0, 0.0])
        assert f1 == pytest.approx(0.25)
        assert f2 == pytest.approx(1.0 - math.sqrt(0.25))

    def test_no_nan_when_f1_exceeds_g(self):
        # guarded sqrt(max(0, f1/g)) must never produce NaN
        f1, f2 = _zdt1([1.0])  # single var → g = 1, f1 = 1
        assert math.isfinite(f2)

    def test_empty_genome(self):
        assert _zdt1([]) == [1.0, 1.0]


# ── SCH1 ─────────────────────────────────────────────────────────────────────

class TestSCH1:
    def test_midpoint(self):
        # x01=0.5 → x = 0.5*2 = 1 → f1=1, f2=(1-2)²=1
        assert _sch1([0.5]) == [1.0, 1.0]

    def test_vertex_f1_zero(self):
        # x01=0 → x = 0 → f1=0, f2=4
        assert _sch1([0.0]) == [0.0, 4.0]

    def test_vertex_f2_zero(self):
        # x01=1 → x = 2 → f1=4, f2=0
        assert _sch1([1.0]) == [4.0, 0.0]


# ── Integration through _eval_benchmark ──────────────────────────────────────

class TestEvalBenchmark:
    def test_dtlz2_dispatch(self):
        vals = _eval_benchmark([0.5] * 6, REGION, bench="DTLZ2", M=3, noise_std=0.0)
        assert len(vals) == 3

    def test_zdt1_dispatch_two_objectives(self):
        vals = _eval_benchmark([10.0, 10.0], REGION, bench="ZDT1", M=2, noise_std=0.0)
        assert len(vals) == 2

    def test_case_insensitive_bench(self):
        a = _eval_benchmark([0.5] * 6, REGION, bench="dtlz2", M=3, noise_std=0.0)
        b = _eval_benchmark([0.5] * 6, REGION, bench="DTLZ2", M=3, noise_std=0.0)
        assert a == b

    def test_noise_perturbs_output(self):
        base = _eval_benchmark([10.0, 20.0], REGION, bench="ZDT1", M=2, noise_std=0.0)
        noisy = _eval_benchmark([10.0, 20.0], REGION, bench="ZDT1", M=2, noise_std=5.0)
        # with a large sigma the noisy result almost surely differs
        assert noisy != base


# ── Mode resolution (per-experiment vs env) ──────────────────────────────────

class TestResolveSyntheticSettings:
    def _exp(self, synthetic):
        return {"parameters": {"simulation": {"synthetic": synthetic}}}

    def test_per_experiment_enabled_takes_priority(self):
        exp = self._exp({"enabled": True, "bench": "ZDT1", "noise_std": 0.1})
        # env says disabled, but per-experiment wins
        enabled, bench, noise = resolve_synthetic_settings(exp, env={})
        assert enabled is True
        assert bench == "ZDT1"
        assert noise == 0.1

    def test_env_fallback_when_no_block(self):
        enabled, bench, noise = resolve_synthetic_settings(
            {"parameters": {"simulation": {}}},
            env={"ENABLE_DATA_SYNTHETIC": "true", "BENCH": "SCH1", "NOISE_STD": "0.2"},
        )
        assert enabled is True
        assert bench == "SCH1"
        assert noise == 0.2

    def test_disabled_by_default(self):
        enabled, bench, noise = resolve_synthetic_settings({"parameters": {}}, env={})
        assert enabled is False
        assert bench == "DTLZ2"  # default
        assert noise == 0.0

    def test_enabled_accepts_string_flag(self):
        exp = self._exp({"enabled": "true", "bench": "DTLZ2"})
        enabled, _, _ = resolve_synthetic_settings(exp, env={})
        assert enabled is True
        exp_false = self._exp({"enabled": "false"})
        assert resolve_synthetic_settings(exp_false, env={})[0] is False

    def test_none_experiment_falls_back_to_env(self):
        enabled, _, _ = resolve_synthetic_settings(None, env={"ENABLE_DATA_SYNTHETIC": "true"})
        assert enabled is True

    def test_tolerates_none_nested_levels(self):
        # parameters present but simulation is None — must not raise
        exp = {"parameters": {"simulation": None}}
        enabled, bench, noise = resolve_synthetic_settings(exp, env={})
        assert enabled is False

    def test_per_experiment_bench_overrides_env(self):
        exp = self._exp({"enabled": True, "bench": "ZDT1"})
        _, bench, _ = resolve_synthetic_settings(exp, env={"BENCH": "DTLZ2"})
        assert bench == "ZDT1"
