"""
Tests for the sink-rooting guarantee of `repair_connectivity_to_sink`.

A previous version seeded the sink component with the closest *active* node
without checking that it lay within the reach radius. When no active node
reached the sink, the repair therefore reported success on a mask whose
components were not connected to the root, and such a mask could reach the
simulator and feed the cache as if it were structurally feasible.

Geometry: sink at the origin, candidates along the x axis, reach radius 10.
"""
import math

from lib.util.connectivity import repair_connectivity_to_sink

RADIUS = 10.0
SINK = (0.0, 0.0)


def _is_rooted(candidates, mask, sink, radius):
    """Independent check: is every active node connected to the sink?"""
    active = [i for i, bit in enumerate(mask) if bit == 1]
    if not active:
        return True
    points = [sink] + [candidates[i] for i in active]
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in range(len(points)):
            if v not in seen and math.dist(points[u], points[v]) <= radius:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(points)


def test_activates_reachable_candidate_when_no_active_node_reaches_sink():
    """Actives are all beyond the radius, but a candidate does reach the sink."""
    candidates = [(8.0, 0.0), (16.0, 0.0), (25.0, 0.0), (32.0, 0.0)]
    mask = [0, 0, 1, 1]

    err, repaired = repair_connectivity_to_sink(candidates, mask, SINK, RADIUS)

    assert not err
    assert _is_rooted(candidates, repaired, SINK, RADIUS)
    assert repaired[0] == 1, "the candidate within reach of the sink must be activated"


def test_reports_error_when_no_candidate_reaches_sink():
    """No candidate position is within reach: no rooted mask exists."""
    candidates = [(25.0, 0.0), (32.0, 0.0)]
    mask = [1, 1]

    err, _ = repair_connectivity_to_sink(candidates, mask, SINK, RADIUS)

    assert err, "an unrootable mask must not be reported as repaired"


def test_does_not_regress_the_ordinary_case():
    """A mask already touching the sink is bridged as before."""
    candidates = [(5.0, 0.0), (12.0, 0.0), (19.0, 0.0)]
    mask = [1, 0, 1]

    err, repaired = repair_connectivity_to_sink(candidates, mask, SINK, RADIUS)

    assert not err
    assert _is_rooted(candidates, repaired, SINK, RADIUS)


def test_empty_mask_is_left_untouched():
    candidates = [(5.0, 0.0), (12.0, 0.0)]

    err, repaired = repair_connectivity_to_sink(candidates, [0, 0], SINK, RADIUS)

    assert not err
    assert repaired == [0, 0]
