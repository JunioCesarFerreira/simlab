"""Finding 3 of the NSGA metrics audit: bounded SBX gave both children the same
spread factor, computed from the LOWER bound.

The two children are pushed away from the parents in opposite directions, so
each one's distribution depends on the distance to the bound it is heading
towards. Reusing the lower child's factor skews the upper child by however
asymmetrically the parents sit inside the box, and the clip that follows does
not restore the distribution — it piles probability mass onto the bound.

DEAP's ``cxSimulatedBinaryBounded`` is the reference implementation. Its child
*pair* must match ours exactly; which child lands in which slot may differ,
because SimLab restores the original parent order and DEAP assigns by index.
"""
from __future__ import annotations

import random
from unittest.mock import patch

import pytest
from deap import tools

from lib.genetic_operators.crossover.simulated_binary_crossover import sbx


class _FixedRng:
    """Feeds ``sbx`` a scripted sequence: the spread draw, then the swap draw."""

    def __init__(self, *values: float):
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


def _deap_children(p1: float, p2: float, eta: float, draw: float, swap: float):
    # DEAP draws three numbers per variable: the per-gene crossover test, the
    # spread draw, then its own swap. 0.0 always passes the first test, so the
    # remaining two line up with ours.
    with patch("random.random", side_effect=[0.0, draw, swap]):
        c1, c2 = tools.cxSimulatedBinaryBounded([p1], [p2], eta, 0.0, 1.0)
    return c1[0], c2[0]


def test_audit_case_upper_child_is_no_longer_pinned_to_the_bound():
    """The exact reproduction from the audit report.

    With the shared spread factor the upper child came out at 1.0 — the bound
    itself, produced by the clip — instead of 0.999287945.
    """
    lower, upper = sbx(0.8, 0.99, _FixedRng(0.99, 0.75), 20, (0.0, 1.0))
    assert lower == pytest.approx(0.780547064, abs=1e-9)
    assert upper == pytest.approx(0.999287945, abs=1e-9)
    assert upper < 1.0


def test_child_pair_matches_deap_across_the_box():
    """Parity sweep: parents anywhere in [0,1]², four spreads, 2000 draws."""
    rng = random.Random(12345)
    for _ in range(2000):
        p1, p2 = rng.random(), rng.random()
        if abs(p1 - p2) < 1e-14:
            continue
        draw = rng.random()
        eta = rng.choice([2.0, 5.0, 20.0, 50.0])
        # swap draw ≥ 0.5 in both implementations → neither swaps.
        ours = sbx(p1, p2, _FixedRng(draw, 0.75), eta, (0.0, 1.0))
        theirs = _deap_children(p1, p2, eta, draw, 0.75)
        assert sorted(ours) == pytest.approx(sorted(theirs), abs=1e-12)


def test_children_of_asymmetric_parents_differ_from_the_shared_factor():
    """Guards the regression directly, without reference to DEAP.

    Parents crowded against the upper bound: the upper child has far less room
    left than the lower one, so its spread factor must be smaller. A shared
    factor puts both children exactly the same distance from the midpoint —
    which here pushes the upper one past the bound, where the clip pins it.
    """
    p1, p2 = 0.80, 0.99
    lower, upper = sbx(p1, p2, _FixedRng(0.99, 0.75), 20, (0.0, 1.0))
    midpoint = 0.5 * (p1 + p2)

    shared_factor_upper = midpoint + (midpoint - lower)
    assert shared_factor_upper > 1.0, "the buggy child was out of bounds, hence the clip"
    assert upper < shared_factor_upper
    assert (upper - midpoint) < (midpoint - lower)


def test_symmetric_parents_keep_symmetric_children():
    """Sanity check on the fix: equal room on both sides → equal offsets."""
    lower, upper = sbx(0.4, 0.6, _FixedRng(0.9, 0.75), 20, (0.0, 1.0))
    assert (0.5 - lower) == pytest.approx(upper - 0.5, abs=1e-12)


def test_identical_parents_are_returned_unchanged():
    assert sbx(0.5, 0.5, _FixedRng(), 20, (0.0, 1.0)) == (0.5, 0.5)


def test_children_stay_inside_the_bounds():
    rng = random.Random(7)
    for _ in range(2000):
        p1, p2 = rng.random(), rng.random()
        if abs(p1 - p2) < 1e-14:
            continue
        for child in sbx(p1, p2, _FixedRng(rng.random(), rng.random()), 20, (0.0, 1.0)):
            assert 0.0 <= child <= 1.0
