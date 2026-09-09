"""Finding 5 of the NSGA metrics audit: GD measured against a SAMPLED reference
front cannot go below that sample's fill distance.

The audit measured points lying exactly on the DTLZ2 front scoring GD 0.0016 at
M=2, 0.030 at M=3 and 0.194 at M=6 — pure discretisation error, reported as if
it were lack of convergence. The plan's original exit criterion was "get it
under 1e-3 with a denser sample"; that is unreachable. The front is an
(M-1)-dimensional manifold, so fill distance shrinks as N**(-1/(M-1)) and M=6
would need ~1e15 points. 200 000 still leaves 0.05.

``benchmarks.front_distance`` sidesteps the sampling entirely with the closed
form, which is what these tests pin.
"""
from __future__ import annotations

import numpy as np
import pytest

from pylib import benchmarks, moo_metrics

ALL = [("DTLZ2", 2), ("DTLZ2", 3), ("DTLZ2", 6), ("ZDT1", 2), ("SCH1", 2)]
IDS = [f"{b}-M{m}" for b, m in ALL]


@pytest.mark.parametrize(("bench", "m"), ALL, ids=IDS)
def test_points_on_the_front_have_zero_distance(bench, m):
    on_front = benchmarks.true_front(bench, m, 300, seed=123)
    assert np.max(benchmarks.front_distance(bench, on_front, m)) < 1e-9


@pytest.mark.parametrize(("bench", "m"), ALL, ids=IDS)
def test_matches_a_brute_force_search_over_a_dense_front(bench, m):
    """Independent check: the closed form against 200 000 sampled front points.

    The brute force is the LOOSE side of this comparison — it can only report a
    distance to a point it happens to have sampled — so the closed form is
    allowed to come in lower, never higher.
    """
    rng = np.random.default_rng(0)
    scale = 4.0 if bench == "SCH1" else 1.5
    points = rng.random((60, m)) * scale
    dense = benchmarks.true_front(bench, m, 200_000, seed=7)

    exact = benchmarks.front_distance(bench, points, m)
    brute = np.min(np.linalg.norm(points[:, None, :] - dense[None, :, :], axis=2), axis=1)
    assert np.all(exact <= brute + 1e-9)
    if m == 2:                       # a curve is sampled densely enough to agree
        assert np.allclose(exact, brute, atol=1e-4)


def test_dtlz2_distance_is_the_radial_error():
    """The front is the unit sphere, so the nearest point to f is f/‖f‖."""
    points = np.array([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0], [1.0, 1.0, 1.0]])
    assert np.allclose(
        benchmarks.front_distance("DTLZ2", points, 3),
        np.abs(np.linalg.norm(points, axis=1) - 1.0),
    )


@pytest.mark.parametrize("m", [2, 3, 6])
def test_the_sampled_reference_floor_is_real_and_the_closed_form_removes_it(m):
    """The finding itself, as a test: same points, two ways of measuring."""
    on_front = benchmarks.true_front("DTLZ2", m, 200, seed=123)
    sampled = moo_metrics.gd(on_front, benchmarks.true_front("DTLZ2", m))
    analytical = moo_metrics.gd_analytical(benchmarks.front_distance("DTLZ2", on_front, m))

    assert analytical < 1e-9
    assert sampled > 100 * analytical
    if m >= 3:
        assert sampled > 0.02, "the floor should be glaring in three or more objectives"


def test_analytical_scale_divides_into_normalised_space():
    distances = np.array([1.0, 3.0])
    assert moo_metrics.gd_analytical(distances) == pytest.approx(2.0)
    assert moo_metrics.gd_analytical(distances, scale=4.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        moo_metrics.gd_analytical(distances, scale=0.0)


def test_theoretical_bounds_are_the_benchmark_corners():
    for bench, m in ALL:
        ideal = np.array(benchmarks.ideal(bench, m))
        nadir = np.array(benchmarks.nadir(bench, m))
        front = benchmarks.true_front(bench, m, 2000, seed=1)
        assert np.all(front >= ideal - 1e-12)
        assert np.all(front <= nadir + 1e-12)
        # Isotropic on every current benchmark, which is what lets the closed
        # form be carried into normalised space by a single division.
        scale = moo_metrics.analytical_scale(ideal, nadir)
        assert np.allclose(scale, scale[0])


def test_normalising_by_theoretical_bounds_differs_from_the_sample():
    """A sparse reference's extremes fall short of the real corners."""
    front = benchmarks.true_front("DTLZ2", 3, 40, seed=5)
    reference = benchmarks.true_front("DTLZ2", 3, 50, seed=11)
    bounds = (np.array(benchmarks.ideal("DTLZ2", 3)), np.array(benchmarks.nadir("DTLZ2", 3)))
    assert moo_metrics.igd(front, reference) != moo_metrics.igd(front, reference, bounds=bounds)


@pytest.mark.parametrize("bench", ["ZDT1", "SCH1"])
def test_two_objective_benchmarks_reject_wider_points(bench):
    with pytest.raises(ValueError):
        benchmarks.front_distance(bench, np.zeros((3, 3)), 3)


def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError):
        benchmarks.front_distance("NOPE", np.zeros((3, 2)), 2)
    with pytest.raises(ValueError):
        benchmarks.ideal("NOPE", 2)
