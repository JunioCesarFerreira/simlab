"""Analytical (true) Pareto fronts for the synthetic benchmark functions.

For synthetic experiments the true Pareto front is known in closed form, so
GD / IGD can be measured against the *real* optimum instead of the experiment's
own final front. Self-reference does not drive the indicators to zero — the
stored front comes from the merged pool (surviving parents u last offspring)
while a generation holds only its own offspring — but it does make them measure
progress towards that one run's result, which is not comparable across runs and
says nothing about distance to the actual optimum.

All fronts are returned in the same objective space and orientation
(minimization) that ``synthetic_data.py`` writes to the database.
"""
from __future__ import annotations

import numpy as np


def dtlz2_front(m: int, n_points: int = 500, seed: int = 0) -> np.ndarray:
    """Sample the DTLZ2 true front: the unit hypersphere segment in the first
    orthant, i.e. ``{ f in R^m_{>=0} : ||f||_2 = 1 }``.

    Points are drawn by normalizing ``|N(0, 1)|`` vectors, giving a spread over
    the positive-orthant sphere. Deterministic for a fixed ``seed``.
    """
    if m < 2:
        raise ValueError(f"DTLZ2 requires m >= 2, got {m}")
    rng = np.random.default_rng(seed)
    pts = np.abs(rng.standard_normal((n_points, m)))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms


def zdt1_front(n_points: int = 500) -> np.ndarray:
    """Sample the ZDT1 true front: ``f2 = 1 - sqrt(f1)`` for ``f1 in [0, 1]``."""
    f1 = np.linspace(0.0, 1.0, n_points)
    f2 = 1.0 - np.sqrt(f1)
    return np.column_stack([f1, f2])


def sch1_front(n_points: int = 500) -> np.ndarray:
    """Sample the SCH1 true front: ``f1 = x^2``, ``f2 = (x - 2)^2`` for the
    Pareto-optimal range ``x in [0, 2]`` (matches the raw objective scale
    produced by ``synthetic_data._sch1``)."""
    x = np.linspace(0.0, 2.0, n_points)
    f1 = x ** 2
    f2 = (x - 2.0) ** 2
    return np.column_stack([f1, f2])


def sample_true_front(bench: str, m: int, n_points: int = 500) -> np.ndarray:
    """Dispatch to the analytical front of a benchmark by id (case-insensitive).

    Returns an ``(n_points, M)`` array of objective vectors (minimization).
    """
    b = bench.upper()
    if b == "DTLZ2":
        return dtlz2_front(m, n_points)
    if b == "ZDT1":
        return zdt1_front(n_points)
    if b == "SCH1":
        return sch1_front(n_points)
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: DTLZ2, ZDT1, SCH1")


def true_ideal(bench: str, m: int) -> list[float]:
    """Best-corner (ideal) point of the analytical front.

    With :func:`true_nadir` it gives the benchmark's THEORETICAL range, which
    normalises the indicators independently of how the reference was sampled.
    Mirrors ``pylib.benchmarks.ideal``.
    """
    b = bench.upper()
    if b == "DTLZ2":
        return [0.0] * m
    if b in ("ZDT1", "SCH1"):
        return [0.0, 0.0]
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: DTLZ2, ZDT1, SCH1")


def _nearest_on_curve(points, to_point, t_lo: float, t_hi: float,
                      grid: int = 2048, iterations: int = 60) -> np.ndarray:
    """Distance from each row of *points* to a 1-D parametric curve.

    Bracket the minimiser on a coarse grid, then ternary-search inside it.
    Mirrors ``pylib.benchmarks._nearest_on_curve``.
    """
    ts = np.linspace(t_lo, t_hi, grid)
    distances = np.linalg.norm(points[:, None, :] - to_point(ts)[None, :, :], axis=2)
    step = (t_hi - t_lo) / (grid - 1)
    closest = ts[np.argmin(distances, axis=1)]
    lo = np.clip(closest - step, t_lo, t_hi)
    hi = np.clip(closest + step, t_lo, t_hi)

    for _ in range(iterations):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        left_is_better = (
            np.linalg.norm(points - to_point(left), axis=1)
            < np.linalg.norm(points - to_point(right), axis=1)
        )
        hi = np.where(left_is_better, right, hi)
        lo = np.where(left_is_better, lo, left)

    return np.linalg.norm(points - to_point(0.5 * (lo + hi)), axis=1)


def front_distance(bench: str, points, m: int) -> np.ndarray:
    """Exact distance from each row of *points* to the true front.

    A sampled reference cannot measure this below its own fill distance, which
    shrinks only as ``N**(-1/(m-1))``: points lying EXACTLY on the DTLZ2 front
    score GD 0.19 at m=6 against a 500-point reference and still 0.05 against
    200 000 points. These closed forms return ~5e-17 for the same points.
    Mirrors ``pylib.benchmarks.front_distance``.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points must be 2-D, got shape {pts.shape}")

    b = bench.upper()
    if b == "DTLZ2":
        if pts.shape[1] != m:
            raise ValueError(f"points have width {pts.shape[1]}, expected m={m}")
        return np.abs(np.linalg.norm(pts, axis=1) - 1.0)
    if pts.shape[1] != 2:
        raise ValueError(f"{b} is a two-objective benchmark; points have width {pts.shape[1]}")
    if b == "ZDT1":
        # Parametrised by u with f1 = u²: 1 - sqrt(f1) has a vertical tangent at
        # the origin, so the search stalls there under the direct parametrisation.
        return _nearest_on_curve(pts, lambda u: np.column_stack([u ** 2, 1.0 - u]), 0.0, 1.0)
    if b == "SCH1":
        return _nearest_on_curve(pts, lambda t: np.column_stack([t ** 2, (t - 2.0) ** 2]), 0.0, 2.0)
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: DTLZ2, ZDT1, SCH1")


def true_nadir(bench: str, m: int) -> list[float]:
    """Worst-corner (nadir) of the analytical front, in the same minimization
    space as :func:`sample_true_front`.

    Used as a FIXED hypervolume reference point so HV is comparable across runs
    of the same synthetic benchmark (instead of a population-derived worst point
    that shifts from run to run). Mirrors ``pylib.benchmarks.nadir``.
    """
    b = bench.upper()
    if b == "DTLZ2":
        return [1.0] * m
    if b == "ZDT1":
        return [1.0, 1.0]
    if b == "SCH1":
        return [4.0, 4.0]
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: DTLZ2, ZDT1, SCH1")
