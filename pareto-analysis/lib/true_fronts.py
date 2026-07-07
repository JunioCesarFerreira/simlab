"""Analytical (true) Pareto fronts for the synthetic benchmark functions.

For synthetic experiments the true Pareto front is known in closed form, so
GD / IGD can be measured against the *real* optimum instead of the experiment's
own final front (which would trivially drive GD to zero on the last generation).

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
