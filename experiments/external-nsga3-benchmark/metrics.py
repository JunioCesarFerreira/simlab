"""
Pareto-quality metrics used by Table 3 of the SimLab paper.

Implements:
  * Hypervolume (HV) using the ``deap.tools._hypervolume.hv`` helper, which
    matches the implementation already used by ``pareto-analysis/``.
  * Generational Distance (GD)        — Veldhuizen & Lamont, 1998.
  * Inverted Generational Distance (IGD) — Coello & Sierra, 2004.
  * Coverage (set-coverage operator C(A,B)) — Zitzler & Thiele, 1998.

All metrics assume *minimisation* of every objective.
"""
from __future__ import annotations

import numpy as np
from deap.tools._hypervolume import hv as deap_hv


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """HV of `front` w.r.t. `ref_point` (worst corner)."""
    if front.size == 0:
        return 0.0
    return float(deap_hv.hypervolume(np.asarray(front, dtype=float), np.asarray(ref_point, dtype=float)))


def generational_distance(front: np.ndarray, ref_front: np.ndarray) -> float:
    """
    GD = sqrt( mean( min_j ||a_i - r_j||^2 ) )
    Smaller is better. ``ref_front`` is the true / reference Pareto front.
    """
    if front.size == 0 or ref_front.size == 0:
        return float("nan")
    a = np.asarray(front, dtype=float)
    r = np.asarray(ref_front, dtype=float)
    # pairwise squared distances
    diff = a[:, None, :] - r[None, :, :]
    sq = np.sum(diff * diff, axis=-1)
    nearest_sq = sq.min(axis=1)
    return float(np.sqrt(nearest_sq.mean()))


def inverted_generational_distance(front: np.ndarray, ref_front: np.ndarray) -> float:
    """
    IGD = mean over reference points of nearest-Euclidean-distance to `front`.
    """
    if front.size == 0 or ref_front.size == 0:
        return float("nan")
    a = np.asarray(front, dtype=float)
    r = np.asarray(ref_front, dtype=float)
    diff = r[:, None, :] - a[None, :, :]
    sq = np.sum(diff * diff, axis=-1)
    nearest = np.sqrt(sq.min(axis=1))
    return float(nearest.mean())


def coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    """
    Zitzler & Thiele C-metric: fraction of points in B that are weakly
    dominated by at least one point in A.  C(A, B) ∈ [0, 1].
    """
    if front_a.size == 0 or front_b.size == 0:
        return 0.0
    a = np.asarray(front_a, dtype=float)
    b = np.asarray(front_b, dtype=float)
    dominated = 0
    for bp in b:
        if np.any(np.all(a <= bp, axis=1) & np.any(a < bp, axis=1)):
            dominated += 1
    return float(dominated) / float(len(b))


def dtlz2_reference_front(M: int, n_points: int = 1000, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample `n_points` points uniformly from the DTLZ2 Pareto front (unit
    hypersphere in the first orthant).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    raw = np.abs(rng.normal(size=(n_points, M)))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return raw / norms
