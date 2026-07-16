"""
Tests for the P2 discrete coverage repair and its `apply_coverage_repair` toggle.

The repair activates candidates via deterministic greedy set-cover over the
pre-built bitset coverage matrix, then restores connectivity to the sink.
`penalty_objectives` remains the safety net for masks the repair cannot fix.

Geometry mirrors test_trajectory_coverage.py: a horizontal trajectory from
(0,0) to (100,0), candidates along the x axis, sink at the origin.
"""
import random

import pytest

from lib.problem.resolve import build_test_adapter
from lib.problem.chromosomes import ChromosomeP2
from lib.util.connectivity import is_connected
from lib.util.trajectory_sampling import (
    sample_trajectories,
    build_coverage_matrix,
    build_candidate_cover_bits,
    check_coverage,
    greedy_coverage_repair_mask,
)


_TRAJ_LINE = [("100 * t", "0")]
_REGION = [-10.0, -10.0, 110.0, 10.0]
_R = 10.0
_SINK = (0.0, 0.0)


def _mobile_nodes():
    class _MN:
        pass
    mn = _MN()
    mn.path_segments = _TRAJ_LINE
    mn.is_closed = False
    mn.is_round_trip = False
    mn.speed = 1.0
    mn.time_step = 1.0
    return [mn]


def _p2_problem(min_pct: float, candidates):
    return {
        "name": "problem2",
        "region": _REGION,
        "sink": _SINK,
        "candidates": candidates,
        "mobile_nodes": [{
            "path_segments": _TRAJ_LINE,
            "is_closed": False,
            "is_round_trip": False,
            "speed": 1.0,
            "time_step": 1.0,
        }],
        "radius_of_reach": _R,
        "radius_of_inter": _R * 1.2,
        "min_coverage_percentage": min_pct,
    }


def _line_candidates():
    # Chain along the trajectory, spacing R: any prefix is sink-connected.
    return [(float(x), 0.0) for x in range(0, 101, 10)]


def _matrix_and_bits(candidates):
    sampled = sample_trajectories(_mobile_nodes(), step=_R / 2)
    matrix = build_coverage_matrix(sampled, candidates, _R)
    return matrix, build_candidate_cover_bits(matrix, len(candidates))


def _adapter(min_pct: float, candidates, apply_repair: bool = True, budget: int = 8):
    adapter = build_test_adapter(_p2_problem(min_pct, candidates))
    adapter.set_ga_operator_configs(
        random.Random(7),
        {
            "per_gene_prob": 0.1,
            "apply_coverage_repair": apply_repair,
            "repair_coverage_budget": budget,
        },
    )
    return adapter


def _active_positions(mask, candidates):
    return [pos for bit, pos in zip(mask, candidates) if bit]


# ──────────────────────────────────────────────────────────────────────────────
# Pure greedy repair function
# ──────────────────────────────────────────────────────────────────────────────

def test_candidate_cover_bits_transposes_matrix():
    candidates = _line_candidates()
    matrix, cover_bits = _matrix_and_bits(candidates)

    assert len(cover_bits) == len(candidates)
    for i, row in enumerate(matrix):
        for j in range(len(candidates)):
            assert bool(row >> j & 1) == bool(cover_bits[j] >> i & 1)


def test_greedy_repair_reaches_threshold_when_possible():
    candidates = _line_candidates()
    matrix, cover_bits = _matrix_and_bits(candidates)
    mask = [0] * len(candidates)
    mask[0] = 1

    repaired = greedy_coverage_repair_mask(matrix, cover_bits, mask, 100.0, len(candidates))

    assert check_coverage(matrix, repaired) == pytest.approx(100.0)


def test_greedy_repair_only_activates_bits():
    candidates = _line_candidates()
    matrix, cover_bits = _matrix_and_bits(candidates)
    mask = [0] * len(candidates)
    mask[3] = 1

    repaired = greedy_coverage_repair_mask(matrix, cover_bits, mask, 100.0, len(candidates))

    for before, after in zip(mask, repaired):
        assert after >= before


def test_greedy_repair_respects_budget():
    candidates = _line_candidates()
    matrix, cover_bits = _matrix_and_bits(candidates)
    mask = [0] * len(candidates)

    repaired = greedy_coverage_repair_mask(matrix, cover_bits, mask, 100.0, budget=2)

    assert sum(repaired) - sum(mask) <= 2


def test_greedy_repair_noop_when_no_candidate_helps():
    # All candidates far from the trajectory: zero gain everywhere.
    candidates = [(float(x), 500.0) for x in range(0, 101, 10)]
    matrix, cover_bits = _matrix_and_bits(candidates)
    mask = [0] * len(candidates)
    mask[0] = 1

    repaired = greedy_coverage_repair_mask(matrix, cover_bits, mask, 100.0, len(candidates))

    assert repaired == mask


def test_greedy_repair_is_deterministic_and_does_not_mutate_input():
    candidates = _line_candidates()
    matrix, cover_bits = _matrix_and_bits(candidates)
    mask = [0] * len(candidates)
    mask[5] = 1
    snapshot = mask[:]

    r1 = greedy_coverage_repair_mask(matrix, cover_bits, mask, 100.0, len(candidates))
    r2 = greedy_coverage_repair_mask(matrix, cover_bits, mask, 100.0, len(candidates))

    assert r1 == r2
    assert mask == snapshot


def test_greedy_repair_noop_when_already_feasible():
    candidates = _line_candidates()
    matrix, cover_bits = _matrix_and_bits(candidates)
    full = [1] * len(candidates)

    assert greedy_coverage_repair_mask(matrix, cover_bits, full, 100.0, 4) == full


# ──────────────────────────────────────────────────────────────────────────────
# Adapter integration: _repair_mask
# ──────────────────────────────────────────────────────────────────────────────

def test_p2_repair_mask_reaches_threshold_and_stays_connected():
    candidates = _line_candidates()
    adapter = _adapter(min_pct=100.0, candidates=candidates, budget=len(candidates))

    # Connected to sink (candidate at x=0) but far from full coverage.
    mask = [0] * len(candidates)
    mask[0] = 1

    repaired = adapter._repair_mask(mask)

    assert adapter.coverage_score(repaired) >= 100.0
    assert is_connected([_SINK, *_active_positions(repaired, candidates)], _R)
    chrm = ChromosomeP2(mac_protocol=0, mask=repaired)
    assert adapter.penalty_objectives(chrm, n_objectives=2) is None


def test_p2_repair_mask_disabled_returns_mask_unchanged():
    candidates = _line_candidates()
    adapter = _adapter(min_pct=100.0, candidates=candidates, apply_repair=False)

    mask = [0] * len(candidates)
    mask[0] = 1

    assert adapter._repair_mask(mask) == mask
    # The viability test still penalises the untouched infeasible mask.
    chrm = ChromosomeP2(mac_protocol=0, mask=mask)
    penalty = adapter.penalty_objectives(chrm, n_objectives=2)
    assert penalty is not None
    assert all(v >= 1e9 for v in penalty)


def test_p2_repair_mask_irreparable_keeps_penalty_applicable():
    # A single candidate cannot cover the whole line no matter the budget.
    candidates = [(50.0, 0.0)]
    adapter = _adapter(min_pct=100.0, candidates=candidates, budget=10)

    mask = [1]
    repaired = adapter._repair_mask(mask)

    assert repaired == mask
    chrm = ChromosomeP2(mac_protocol=0, mask=repaired)
    assert adapter.penalty_objectives(chrm, n_objectives=2) is not None


def test_p2_repair_mask_noop_when_already_feasible():
    candidates = _line_candidates()
    adapter = _adapter(min_pct=50.0, candidates=candidates)

    mask = [1] * len(candidates)
    assert adapter._repair_mask(mask) == mask


# ──────────────────────────────────────────────────────────────────────────────
# Adapter integration: genetic operators
# ──────────────────────────────────────────────────────────────────────────────

def test_p2_random_individuals_are_feasible_with_repair_on():
    candidates = _line_candidates()
    adapter = _adapter(min_pct=100.0, candidates=candidates, budget=len(candidates))

    for chrm in adapter.random_individual_generator(8):
        assert adapter.penalty_objectives(chrm, n_objectives=2) is None
        assert is_connected([_SINK, *_active_positions(chrm.mask, candidates)], _R)


def test_p2_crossover_children_are_feasible_with_repair_on():
    candidates = _line_candidates()
    adapter = _adapter(min_pct=100.0, candidates=candidates, budget=len(candidates))

    parents = adapter.random_individual_generator(2)
    for child in adapter.crossover(parents):
        assert adapter.penalty_objectives(child, n_objectives=2) is None
        assert is_connected([_SINK, *_active_positions(child.mask, candidates)], _R)


def test_p2_mutation_output_is_feasible_with_repair_on():
    candidates = _line_candidates()
    adapter = _adapter(min_pct=100.0, candidates=candidates, budget=len(candidates))

    parent = adapter.random_individual_generator(1)[0]
    child = adapter.mutate(parent)

    assert adapter.penalty_objectives(child, n_objectives=2) is None
    assert is_connected([_SINK, *_active_positions(child.mask, candidates)], _R)
