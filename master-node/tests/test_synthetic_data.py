"""Unit tests for the synthetic benchmark evaluation (lib/synthetic_data.py).

These lock in the correctness fixes for synthetic mode:
  * the sink is excluded from the genome (decision vars = relay coords);
  * region normalization respects a custom Ω;
  * DTLZ2 lands on the unit hypersphere on the Pareto front;
  * ZDT1 / SCH1 match their analytical definitions.
"""
import math
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.synthetic_data import (
    _extract_genome_from_sim,
    _scale_to_unit,
    _dtlz2,
    _zdt1,
    _sch1,
    _eval_benchmark,
    _benchmark_values,
    _decision_vector_from_sim,
    resolve_synthetic_settings,
    run_synthetic_simulation,
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
    # Default domain is now the wide convergence domain (-5, 5); x01=0.5 → x=0.
    def test_center_is_pareto_optimal(self):
        assert _sch1([0.5]) == [0.0, 4.0]

    def test_extremes_are_off_front(self):
        assert _sch1([0.0]) == [25.0, 49.0]   # x = -5
        assert _sch1([1.0]) == [25.0, 9.0]    # x = +5

    def test_explicit_narrow_domain(self):
        # The classic spread-only (0,2) mapping is still available on request.
        assert _sch1([0.5], (0.0, 2.0)) == [1.0, 1.0]


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


# ── P0 decision-vector path (pure synthetic) ─────────────────────────────────

class TestDecisionVectorP0:
    def test_present_returns_vector(self):
        sim = {"parameters": {"simulationElements": {
            "fixedMotes": [], "mobileMotes": [], "decisionVector": [0.0, 0.5, 1.0],
        }}}
        assert _decision_vector_from_sim(sim) == [0.0, 0.5, 1.0]

    def test_absent_returns_none(self):
        # P1-encoded sim (relay motes, no decisionVector) → falls back to physical path
        sim = {"parameters": {"simulationElements": {
            "fixedMotes": [{"name": "sink", "position": [0.0, 0.0]}], "mobileMotes": [],
        }}}
        assert _decision_vector_from_sim(sim) is None

    def test_missing_structure_returns_none(self):
        assert _decision_vector_from_sim({}) is None

    def test_sch1_pareto_vertices_reachable_directly(self):
        # P0 evaluates directly on x ∈ [0,1]^n. With the wide default domain the
        # Pareto vertices (0,4) and (4,0) are reachable at x=0 (x01=0.5) and
        # x=2 (x01=0.7) — the optimiser must converge onto x ∈ [0,2].
        assert _benchmark_values([0.5], "SCH1", 2) == [0.0, 4.0]
        assert _benchmark_values([0.7], "SCH1", 2) == pytest.approx([4.0, 0.0], abs=1e-9)

    def test_dtlz2_on_sphere_directly(self):
        f = _benchmark_values([0.5] * 6, "DTLZ2", 3)
        assert math.sqrt(sum(v * v for v in f)) == pytest.approx(1.0, abs=1e-9)


# ── run_synthetic_simulation: seeded/clamped noise + validation (integration) ─

def _mongo_stub(objective_names=("f1", "f2"), problem=None):
    mongo = MagicMock()
    mongo.experiment_repo.get.return_value = {
        "parameters": {
            "objectives": [{"metric_name": n} for n in objective_names],
            "problem": problem or {"name": "problem0", "n": 6},
        }
    }
    mongo.generation_repo.all_simulations_done.return_value = False
    mongo.generation_repo.any_simulation_active.return_value = True
    return mongo


def _sim(decision_vector, seed=42, individual="genome-abc"):
    return {
        "_id": ObjectId(),
        "experiment_id": ObjectId(),
        "individual_id": individual,
        "random_seed": seed,
        "generation_id": ObjectId(),
        "parameters": {"simulationElements": {"decisionVector": decision_vector}},
    }


class TestRunSyntheticNoiseAndValidation:
    def test_noise_is_reproducible_for_same_seed_and_genome(self):
        # #2: seeded per-(seed, genome) noise → two evaluations agree exactly.
        objs = []
        for _ in range(2):
            mongo = _mongo_stub()
            run_synthetic_simulation(_sim([0.3, 0.1, 0.5, 0.5, 0.5, 0.5], seed=7),
                                     mongo, bench="ZDT1", noise_std=0.5)
            objs.append(mongo.simulation_repo.mark_done.call_args[0][3])
        assert objs[0] == objs[1]

    def test_noise_differs_across_genomes(self):
        mongo_a = _mongo_stub(); mongo_b = _mongo_stub()
        run_synthetic_simulation(_sim([0.3, 0.1, 0.5, 0.5, 0.5, 0.5], seed=7, individual="A"),
                                 mongo_a, bench="ZDT1", noise_std=0.5)
        run_synthetic_simulation(_sim([0.3, 0.1, 0.5, 0.5, 0.5, 0.5], seed=7, individual="B"),
                                 mongo_b, bench="ZDT1", noise_std=0.5)
        assert mongo_a.simulation_repo.mark_done.call_args[0][3] != \
               mongo_b.simulation_repo.mark_done.call_args[0][3]

    def test_noisy_objectives_clamped_non_negative(self):
        # #8: heavy noise must never drive a non-negative objective below 0.
        mongo = _mongo_stub()
        run_synthetic_simulation(_sim([0.5] * 6, seed=1), mongo, bench="DTLZ2", noise_std=100.0)
        objectives = mongo.simulation_repo.mark_done.call_args[0][3]
        assert all(v >= 0.0 for v in objectives.values())

    def test_dtlz2_too_few_variables_marks_error_not_crash(self):
        # #7: DTLZ2 with n < M-1 → clear error status, no crash, no mark_done.
        mongo = _mongo_stub(objective_names=("f1", "f2", "f3"))  # M = 3, needs n >= 2
        run_synthetic_simulation(_sim([0.5], seed=1), mongo, bench="DTLZ2", noise_std=0.0)
        mongo.simulation_repo.mark_error.assert_called_once()
        mongo.simulation_repo.mark_done.assert_not_called()

    def test_zero_noise_matches_pure_evaluation(self):
        mongo = _mongo_stub()
        run_synthetic_simulation(_sim([0.25, 0.0, 0.0, 0.0, 0.0, 0.0], seed=3),
                                 mongo, bench="ZDT1", noise_std=0.0)
        objectives = mongo.simulation_repo.mark_done.call_args[0][3]
        assert objectives["f1"] == pytest.approx(0.25)


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
