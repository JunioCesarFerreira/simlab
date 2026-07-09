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

import math

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


# ── Multi-run statistics ─────────────────────────────────────────────────────
# Standalone copies of the platform helpers in pareto-analysis/lib/stats.py: a
# single run (or only the mean of several) cannot support "method A beats B".

def summary(values) -> dict:
    """Return ``{n, mean, std, ci95}`` (sample std, ddof=1; 95% CI half-width)."""
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    return {"n": n, "mean": float(a.mean()), "std": std,
            "ci95": 1.96 * std / math.sqrt(n) if n > 1 else 0.0}


def wilcoxon_pvalue(a, b) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value (scipy if available, else a
    numpy-only normal approximation). ``nan`` for degenerate input."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    d = a - b
    d = d[d != 0]
    n = int(d.size)
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        pass
    order = np.argsort(np.abs(d), kind="mergesort")
    sx = np.abs(d)[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return float("nan")
    z = (w_plus - mean) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


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
