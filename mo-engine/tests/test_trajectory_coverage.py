"""
Consistency tests for the trajectory coverage constraint shared by P1 and P2.

Both problems must:
- Return a coverage score in [0, 100] (percent).
- Use the same `min_coverage_percentage` semantics (direct comparison).
- Produce `penalty_objectives` ≥ 1e9 when infeasible, `None` when feasible.
- Give monotonic penalties (smaller coverage → larger penalty).
- Return full coverage (100) when the trajectory set is empty / vacuous.

P1 uses the W1..W8 region-partition algorithm with continuous relay positions.
P2 uses a pre-built bitset coverage matrix over a fixed candidate set.
For an equivalent geometric setup both must yield the same score.
"""
import pytest

from lib.problem.resolve import build_test_adapter
from lib.problem.chromosomes import ChromosomeP1, ChromosomeP2
from lib.util.region_partition import TrajectoryConstraintP1
from lib.util.trajectory_sampling import (
    sample_trajectories,
    build_coverage_matrix,
    check_coverage,
)


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

# Horizontal trajectory from (0,0) to (100,0), arc length 100.
_TRAJ_LINE = [("100 * t", "0")]
_REGION = [-10.0, -10.0, 110.0, 10.0]   # [xmin, ymin, xmax, ymax]
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


# ──────────────────────────────────────────────────────────────────────────────
# Scale / range
# ──────────────────────────────────────────────────────────────────────────────

def test_p1_score_range_and_full_coverage():
    c = TrajectoryConstraintP1(_SINK, _mobile_nodes(), _R, _REGION)

    # Relays evenly along the line, spacing R (so balls overlap)
    relays = [(x, 0.0) for x in range(10, 101, 10)]
    score = c.check_coverage(relays)
    assert score == pytest.approx(100.0)


def test_p1_score_zero_when_nothing_reaches():
    c = TrajectoryConstraintP1((1000.0, 1000.0), _mobile_nodes(), _R, _REGION)
    # Sink is far away; no relays → 0% coverage
    assert c.check_coverage([]) == 0.0


def test_p2_score_range_and_full_coverage():
    candidates = [(x, 0.0) for x in range(0, 101, 10)]
    sampled = sample_trajectories(_mobile_nodes(), step=_R / 2)
    matrix = build_coverage_matrix(sampled, candidates, _R)

    full_mask = [1] * len(candidates)
    assert check_coverage(matrix, full_mask) == pytest.approx(100.0)

    empty_mask = [0] * len(candidates)
    # Sink is NOT part of the P2 matrix — all-zero mask gives 0 coverage here.
    assert check_coverage(matrix, empty_mask) == 0.0


def test_p1_vacuous_when_no_trajectory_points():
    # Empty mobile_nodes → no sampled points → vacuously covered
    c = TrajectoryConstraintP1(_SINK, [], _R, _REGION)
    assert c.has_points is False
    assert c.check_coverage([]) == 100.0


def test_p2_vacuous_empty_matrix():
    assert check_coverage([], [1, 0, 1]) == 100


# ──────────────────────────────────────────────────────────────────────────────
# P1/P2 geometric equivalence
# ──────────────────────────────────────────────────────────────────────────────

def test_p1_and_p2_agree_on_same_geometry():
    """
    Same sink, trajectory, radius and active positions → same score.
    (P2's matrix does not include the sink, so we exclude it from P1 too
    by placing it outside coverage range.)
    """
    far_sink = (1000.0, 1000.0)
    positions = [(10.0, 0.0), (30.0, 0.0), (50.0, 0.0)]

    c1 = TrajectoryConstraintP1(far_sink, _mobile_nodes(), _R, _REGION)
    s1 = c1.check_coverage(positions)

    sampled = sample_trajectories(_mobile_nodes(), step=_R / 2)
    matrix = build_coverage_matrix(sampled, positions, _R)
    s2 = check_coverage(matrix, [1, 1, 1])

    assert s1 == pytest.approx(s2)


# ──────────────────────────────────────────────────────────────────────────────
# Penalty semantics
# ──────────────────────────────────────────────────────────────────────────────

def _p1_problem(min_pct: float):
    return {
        "name": "problem1",
        "region": _REGION,
        "sink": _SINK,
        "mobile_nodes": [{
            "path_segments": _TRAJ_LINE,
            "is_closed": False,
            "is_round_trip": False,
            "speed": 1.0,
            "time_step": 1.0,
        }],
        "number_of_relays": 3,
        "radius_of_reach": _R,
        "radius_of_inter": _R * 1.2,
        "min_coverage_percentage": min_pct,
    }


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


def test_p1_penalty_none_when_feasible():
    adapter = build_test_adapter(_p1_problem(min_pct=50.0))
    relays = [(x, 0.0) for x in range(10, 101, 10)]
    chrm = ChromosomeP1(mac_protocol=0, relays=relays)
    assert adapter.penalty_objectives(chrm, n_objectives=2) is None


def test_p1_penalty_monotonic_with_deficit():
    adapter = build_test_adapter(_p1_problem(min_pct=100.0))

    # Fewer relays → less coverage → larger penalty
    chrm_high = ChromosomeP1(mac_protocol=0, relays=[(20.0, 0.0), (50.0, 0.0), (80.0, 0.0)])
    chrm_low  = ChromosomeP1(mac_protocol=0, relays=[(200.0, 200.0)] * 3)  # all out of range

    p_high = adapter.penalty_objectives(chrm_high, n_objectives=2)
    p_low  = adapter.penalty_objectives(chrm_low,  n_objectives=2)

    assert p_high is not None and p_low is not None
    assert all(v >= 1e9 for v in p_high)
    assert all(v >= 1e9 for v in p_low)
    # Worse coverage → strictly higher penalty
    assert p_low[0] > p_high[0]


def test_p2_penalty_none_when_feasible():
    candidates = [(x, 0.0) for x in range(0, 101, 10)]
    adapter = build_test_adapter(_p2_problem(min_pct=50.0, candidates=candidates))
    chrm = ChromosomeP2(mac_protocol=0, mask=[1] * len(candidates))
    assert adapter.penalty_objectives(chrm, n_objectives=2) is None


def test_p2_penalty_monotonic_with_deficit():
    candidates = [(x, 0.0) for x in range(0, 101, 10)]
    adapter = build_test_adapter(_p2_problem(min_pct=100.0, candidates=candidates))

    # Only one candidate active → partial coverage, infeasible under min_pct=100
    mask_high = [0] * len(candidates)
    mask_high[5] = 1  # cover the middle only (x=50)
    mask_low  = [0] * len(candidates)

    chrm_high = ChromosomeP2(mac_protocol=0, mask=mask_high)
    chrm_low  = ChromosomeP2(mac_protocol=0, mask=mask_low)

    p_high = adapter.penalty_objectives(chrm_high, n_objectives=2)
    p_low  = adapter.penalty_objectives(chrm_low,  n_objectives=2)

    assert p_high is not None and p_low is not None
    assert all(v >= 1e9 for v in p_high)
    assert all(v >= 1e9 for v in p_low)
    assert p_low[0] > p_high[0]


def test_p1_and_p2_penalty_match_for_equivalent_setup():
    """
    P1 and P2 adapters fed with the same geometry and min_pct must yield
    the same penalty value for an equivalent infeasible chromosome.
    """
    candidates = [(50.0, 0.0)]  # only the middle — neither end covered
    min_pct = 100.0

    p1_adapter = build_test_adapter({**_p1_problem(min_pct=min_pct), "number_of_relays": 1})
    p2_adapter = build_test_adapter(_p2_problem(min_pct=min_pct, candidates=candidates))

    # Use a far sink for P1 so only the single relay matters — equal to P2's no-sink matrix.
    # Build an alternate P1 adapter with far sink:
    p1_far = build_test_adapter({
        **_p1_problem(min_pct=min_pct),
        "number_of_relays": 1,
        "sink": (1000.0, 1000.0),
    })

    c1 = ChromosomeP1(mac_protocol=0, relays=[(50.0, 0.0)])
    c2 = ChromosomeP2(mac_protocol=0, mask=[1])

    p1 = p1_far.penalty_objectives(c1, n_objectives=2)
    p2 = p2_adapter.penalty_objectives(c2, n_objectives=2)

    assert p1 is not None and p2 is not None
    assert p1[0] == pytest.approx(p2[0], rel=1e-6)
    # silence unused-var warning
    _ = p1_adapter
