"""Parity guard: the offline indicators (lib/metrics.py) must produce exactly the
numbers the live quality-indicator endpoint produces via pylib.moo_metrics.

pareto-analysis keeps its own standalone copy on purpose (the CLIs must run with
no pylib on the path), so this test — not a shared import — is what prevents the
two from drifting apart. Same arrangement as test_benchmarks_parity.py.
"""
import numpy as np
import pytest

from lib import metrics
from pylib import moo_metrics as canonical


@pytest.fixture(scope="module")
def sets():
    rng = np.random.default_rng(20260902)
    front = rng.random((13, 3))
    reference = rng.random((37, 3))
    return front, reference


def test_penalty_threshold_matches():
    assert metrics.PENALTY_THRESHOLD == canonical.PENALTY_THRESHOLD


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("name", ["gd", "igd", "igd_plus"])
def test_indicator_matches_canonical(sets, name, normalized):
    front, reference = sets
    mine = getattr(metrics, name)(front, reference, normalized=normalized)
    theirs = getattr(canonical, name)(front, reference, normalized=normalized)
    assert mine == theirs


def test_normalization_bounds_match(sets):
    _, reference = sets
    mine = metrics.normalization_bounds(reference)
    theirs = canonical.normalization_bounds(reference)
    assert np.array_equal(mine[0], theirs[0])
    assert np.array_equal(mine[1], theirs[1])


def test_sanitize_reference_front_matches():
    rows = [
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [0.0, 5.0],
        [1e9, 0.1],
        [5.0, 0.0],
    ]
    assert np.array_equal(
        metrics.sanitize_reference_front(rows),
        canonical.sanitize_reference_front(rows),
    )


def test_normalization_is_on_by_default_in_both(sets):
    front, reference = sets
    assert metrics.gd(front, reference) == metrics.gd(front, reference, normalized=True)
    assert canonical.gd(front, reference) == canonical.gd(front, reference, normalized=True)


def test_analytical_scale_matches(sets):
    ideal, nadir = np.zeros(3), np.array([1.0, 4.0, 0.0])
    assert np.array_equal(
        metrics.analytical_scale(ideal, nadir),
        canonical.analytical_scale(ideal, nadir),
    )


@pytest.mark.parametrize("name", ["gd", "igd", "igd_plus"])
def test_explicit_bounds_match(sets, name):
    """The theoretical ideal-nadir range must be honoured identically."""
    front, reference = sets
    bounds = (np.zeros(3), np.array([1.0, 1.0, 1.0]))
    mine = getattr(metrics, name)(front, reference, bounds=bounds)
    theirs = getattr(canonical, name)(front, reference, bounds=bounds)
    assert mine == theirs
    # And they must actually differ from the sampled-reference normalisation,
    # or the parameter would be untested.
    assert mine != getattr(metrics, name)(front, reference)


def test_gd_analytical_matches():
    distances = np.array([0.0, 0.25, 1.5, 3.0])
    assert metrics.gd_analytical(distances, scale=4.0) == canonical.gd_analytical(
        distances, scale=4.0
    )
