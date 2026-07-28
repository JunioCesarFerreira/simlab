"""Unit tests for GA parameter alignment between launcher payloads and adapters.

These lock in the fail-fast contract of build_adapter:
  * P1 defaults crossover_method to sbx_with_radial_translate when absent;
  * an unsupported P1 crossover_method fails at configuration time, before
    any simulation is spent (regression: the old code crashed mid-evolution);
  * keys a problem does not consume are ignored with a warning, not an error;
  * every adapter accepts the launch wizard's default payload.
"""
import logging
import random

import pytest

from lib.problem.resolve import build_adapter, STRATEGY_GA_KEYS

P1_PROBLEM = {
    "name": "problem1",
    "region": [-100.0, -100.0, 100.0, 100.0],
    "sink": (0.0, 0.0),
    "mobile_nodes": [],
    "min_coverage_percentage": 0.0,
    "number_of_relays": 2,
    "radius_of_reach": 50.0,
    "radius_of_inter": 60.0,
}

P2_PROBLEM = {
    "name": "problem2",
    "region": [-100.0, 100.0, -100.0, 100.0],
    "sink": (0.0, 0.0),
    "candidates": [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
    "mobile_nodes": [
        {
            "path_segments": [("0 + 5*t", "0")],
            "is_closed": False,
            "is_round_trip": False,
            "speed": 1.0,
            "time_step": 1.0,
        }
    ],
    "radius_of_reach": 50.0,
    "radius_of_inter": 60.0,
}

P3_PROBLEM = {
    "name": "problem3",
    "region": [-100.0, 100.0, -100.0, 100.0],
    "sink": (0.0, 0.0),
    "candidates": [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
    "targets": [(15.0, 10.0)],
    "k_required": 1,
    "radius_of_reach": 50.0,
    "radius_of_inter": 60.0,
    "radius_of_cover": 30.0,
}

P4_PROBLEM = {
    "name": "problem4",
    "region": [-100.0, 100.0, -100.0, 100.0],
    "radius_of_reach": 50.0,
    "radius_of_inter": 60.0,
    "nodes": [(10.0, 0.0), (20.0, 0.0)],
    "sink_base": (0.0, 0.0),
    "initial_energy": 100.0,
    "buffer_capacity": 50.0,
    "data_rate": 1.0,
    "speed": 10.0,
    "time_step": 1.0,
    "sojourns": [
        {"id": 0, "position": (0.0, 0.0), "adjacency": [1], "visibleNodes": [0]},
        {"id": 1, "position": (30.0, 0.0), "adjacency": [0], "visibleNodes": [1]},
    ],
}

P0_PROBLEM = {"name": "problem0", "n": 4}

# Payload the problem-editor launch wizard sends for an evolutionary strategy
# (LaunchWizard.vue), after the per-problem capability filtering.
WIZARD_DEFAULTS = {
    "population_size": 8,
    "number_of_generations": 3,
    "random_seed": 42,
    "prob_cx": 0.8,
    "prob_mt": 0.15,
    "per_gene_prob": 0.1,
}


def _rng() -> random.Random:
    return random.Random(42)


# ── P1 crossover_method contract ─────────────────────────────────────────────

class TestP1CrossoverMethod:
    def test_defaults_to_sbx_when_absent(self):
        adapter = build_adapter(P1_PROBLEM, WIZARD_DEFAULTS, _rng())
        assert adapter._crossover_method == "sbx_with_radial_translate"

    def test_rand_network_accepted(self):
        params = {**WIZARD_DEFAULTS, "crossover_method": "rand_network"}
        adapter = build_adapter(P1_PROBLEM, params, _rng())
        assert adapter._crossover_method == "rand_network"

    def test_case_insensitive(self):
        params = {**WIZARD_DEFAULTS, "crossover_method": "SBX_With_Radial_Translate"}
        adapter = build_adapter(P1_PROBLEM, params, _rng())
        assert adapter._crossover_method == "sbx_with_radial_translate"

    def test_unsupported_method_fails_at_config_time(self):
        # Regression: the pre-fix GUI default for every problem.
        params = {**WIZARD_DEFAULTS, "crossover_method": "uniform_mask"}
        with pytest.raises(ValueError, match="uniform_mask"):
            build_adapter(P1_PROBLEM, params, _rng())


# ── Ignored-key warnings ─────────────────────────────────────────────────────

class TestIgnoredKeyWarnings:
    def test_p3_warns_on_operator_keys(self, caplog):
        params = {
            **WIZARD_DEFAULTS,
            "crossover_method": "uniform_mask",
            "mutation_method": "bitflip",
            "apply_coverage_repair": True,
        }
        with caplog.at_level(logging.WARNING, logger="lib.problem.resolve"):
            build_adapter(P3_PROBLEM, params, _rng())
        joined = " ".join(r.message for r in caplog.records)
        assert "crossover_method" in joined
        assert "mutation_method" in joined
        assert "apply_coverage_repair" in joined

    def test_consumed_keys_do_not_warn(self, caplog):
        params = {**WIZARD_DEFAULTS, "apply_coverage_repair": True, "repair_coverage_budget": 4}
        with caplog.at_level(logging.WARNING, logger="lib.problem.resolve"):
            build_adapter(P2_PROBLEM, params, _rng())
        assert not caplog.records

    def test_strategy_keys_do_not_warn(self, caplog):
        params = {**WIZARD_DEFAULTS, "divisions": 10, "selection_method": "tournament"}
        with caplog.at_level(logging.WARNING, logger="lib.problem.resolve"):
            build_adapter(P0_PROBLEM, params, _rng())
        assert not caplog.records
        assert "selection_method" in STRATEGY_GA_KEYS


# ── Wizard defaults never break any adapter ──────────────────────────────────

class TestWizardDefaultsAccepted:
    @pytest.mark.parametrize(
        "problem",
        [P0_PROBLEM, P1_PROBLEM, P2_PROBLEM, P3_PROBLEM, P4_PROBLEM],
        ids=["p0", "p1", "p2", "p3", "p4"],
    )
    def test_build_adapter_with_defaults(self, problem):
        adapter = build_adapter(problem, WIZARD_DEFAULTS, _rng())
        assert adapter is not None
