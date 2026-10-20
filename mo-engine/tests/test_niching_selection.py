"""Finding 2 of the NSGA metrics audit: the native NSGA-III niching was not the
canonical one. Four defects, one test each, plus parity with the references.

The reference implementations are ``deap.tools.emo`` and pymoo's
``HyperplaneNormalization``. They agree with each other — and with this module —
on the extreme points and the association, and they disagree with each other on
the intercepts: DEAP solves the hyperplane in the TRANSLATED space (so ``1/x``
is already a distance from the ideal point) and then divides by
``intercepts - ideal``, subtracting the ideal a second time. pymoo returns
``ideal + 1/x`` and divides by ``nadir - ideal``. We follow pymoo, which is what
Deb & Jain's Eq. (4) describes; ``test_intercepts_match_pymoo_not_deap`` pins
that choice so it cannot be "fixed" back by accident.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from lib.nsga.niching_selection import (
    _find_extreme_points,
    _find_intercepts,
    associate_to_niches,
    generate_reference_points,
    niching_selection,
)


def _rng() -> random.Random:
    return random.Random(20260909)


# ── Parity with the reference implementations ────────────────────────────────

def test_extreme_points_match_deap():
    from deap.tools import emo

    rng = np.random.default_rng(0)
    for _ in range(200):
        M = int(rng.choice([2, 3, 5]))
        F = np.abs(rng.standard_normal((40, M))) + rng.choice([0.0, 3.0])
        ideal = F.min(axis=0)
        assert np.array_equal(_find_extreme_points(F, ideal), emo.find_extreme_points(F, ideal))


def test_intercepts_match_pymoo_not_deap():
    """The one place the two references disagree; we follow pymoo."""
    from deap.tools import emo
    from pymoo.algorithms.moo.nsga3 import get_nadir_point

    ideal = np.array([2.0, 2.0, 2.0])
    extreme = np.array([[5.0, 2.0, 2.0], [2.0, 6.0, 2.0], [2.0, 2.0, 7.0]])
    worst = np.array([9.0, 9.0, 9.0])

    ours = _find_intercepts(extreme, ideal, worst)
    assert np.allclose(ours, get_nadir_point(extreme, ideal, worst, worst, worst))
    # DEAP returns the same intercepts relative to the ideal point, then divides
    # by (intercepts - ideal) anyway — an ideal point subtracted twice.
    assert np.allclose(emo.find_intercepts(extreme, ideal, worst, worst), ours - ideal)


def test_intercepts_fall_back_when_the_hyperplane_is_degenerate():
    """Linearly dependent extreme points must not blow the normalisation up."""
    ideal = np.zeros(3)
    worst = np.array([4.0, 5.0, 6.0])
    degenerate = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    assert np.array_equal(_find_intercepts(degenerate, ideal, worst), worst)


def test_association_matches_deap():
    from deap.tools import emo

    rng = np.random.default_rng(1)
    H = generate_reference_points(3, 6)
    for _ in range(100):
        F = np.abs(rng.standard_normal((40, 3))) + rng.choice([0.0, 3.0])
        ideal, worst = F.min(axis=0), F.max(axis=0)
        intercepts = _find_intercepts(_find_extreme_points(F, ideal), ideal, worst)
        niches, distances = associate_to_niches(F, H, ideal, intercepts)
        # Feed DEAP the intercepts in ITS convention so the scales coincide.
        deap_niches, deap_dist = emo.associate_to_niche(F, H, ideal, intercepts)
        assert np.array_equal(niches, deap_niches)
        assert np.allclose(distances, deap_dist)


# ── The four defects, one test each ──────────────────────────────────────────

def test_association_is_perpendicular_not_euclidean_to_the_point():
    """Defect 2: distance was measured to the reference POINT, not to its ray.

    A solution far out along a direction is perfectly aligned with it, however
    distant the reference point itself is.
    """
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    ideal, intercepts = np.zeros(2), np.ones(2)
    # (3, 0) lies exactly on the first axis but far from the point (1, 0);
    # (0.6, 0.5) is closer to that point in plain Euclidean terms yet is not
    # aligned with either axis.
    niches, distances = associate_to_niches(
        np.array([[3.0, 0.0], [0.6, 0.5]]), H, ideal, intercepts
    )
    assert niches[0] == 0
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(np.array([3.0, 0.0]) - H[0]) > distances[1]


def test_niche_occupancy_starts_from_the_already_accepted():
    """Defect 3: occupancy started at zero, ignoring the accepted fronts.

    Index 0 is already accepted on the first axis, so the single free slot must
    go to the candidate on the SECOND axis, even though a candidate sits closer
    to the first.
    """
    objectives = [
        [0.0, 1.0],   # 0: accepted, associated with the f2 axis
        [0.02, 1.0],  # 1: candidate, same crowded direction, very close to it
        [1.0, 0.0],   # 2: candidate, the empty direction
    ]
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    chosen = niching_selection([1, 2], objectives, H, 1, _rng(), accepted=[0])
    assert chosen == [2]

    # Without the accepted set the empty-niche rule cannot see the crowding, and
    # the tie is decided by distance alone.
    assert niching_selection([1, 2], objectives, H, 1, _rng()) == [1]


def test_normalization_uses_the_accepted_fronts_too():
    """Defect 1: normalising on the truncated front alone.

    The accepted individuals stretch the ideal point and the intercepts. Two
    calls that differ only in the accepted set must therefore normalise
    differently — the front in isolation cannot know the pool it joins.
    """
    front_objs = {0: [0.4, 0.6], 1: [0.6, 0.4]}
    objectives = [front_objs[0], front_objs[1], [0.0, 5.0], [5.0, 0.0]]
    H = generate_reference_points(2, 4)

    alone = niching_selection([0, 1], objectives, H, 1, _rng())
    with_pool = niching_selection([0, 1], objectives, H, 1, _rng(), accepted=[2, 3])
    assert alone != with_pool


def test_empty_niches_do_not_trigger_a_blind_random_pick():
    """Defect 4: empty niches kept winning the minimum-occupancy race, which
    pushed the loop into a uniform random pick over everything left — a pick
    that did not even update the occupancy.

    Here there are 15 reference directions and candidates in only two of them.
    Selection must spread across those two, never stack one.
    """
    H = generate_reference_points(2, 14)
    assert len(H) == 15
    # Six candidates: three tightly clustered on each of two directions.
    objectives = [
        [0.00, 1.00], [0.01, 0.99], [0.02, 0.98],
        [1.00, 0.00], [0.99, 0.01], [0.98, 0.02],
    ]
    front = list(range(6))
    chosen = niching_selection(front, objectives, H, 4, _rng())
    assert len(chosen) == 4
    assert len(set(chosen)) == 4
    low = sum(1 for i in chosen if i < 3)
    assert low == 2, f"selection stacked one direction: {chosen}"


# ── Contract ─────────────────────────────────────────────────────────────────

def test_returns_the_whole_front_when_it_fits():
    H = generate_reference_points(2, 4)
    objectives = [[0.0, 1.0], [1.0, 0.0]]
    assert niching_selection([0, 1], objectives, H, 5, _rng()) == [0, 1]


def test_selection_is_reproducible_for_a_seed():
    H = generate_reference_points(3, 6)
    rng = np.random.default_rng(3)
    objectives = (np.abs(rng.standard_normal((60, 3)))).tolist()
    front = list(range(60))
    first = niching_selection(front, objectives, H, 20, _rng(), accepted=[])
    second = niching_selection(front, objectives, H, 20, _rng(), accepted=[])
    assert first == second


def test_rejects_reference_points_of_the_wrong_width():
    objectives = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
    with pytest.raises(ValueError):
        niching_selection([0, 1, 2], objectives, generate_reference_points(3, 4), 2, _rng())
