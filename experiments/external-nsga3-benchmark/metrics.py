"""
Pareto-quality metrics used by Table 3 of the SimLab paper.

Implements:
  * Hypervolume (HV), exact, with ``moocore`` as the platform
    (``pylib/moo_metrics.py``, pareto-analysis).
  * Generational Distance (GD)        — Veldhuizen & Lamont, 1998.
  * Inverted Generational Distance (IGD) — Coello & Sierra, 2004.
  * IGD+ — Ishibuchi et al., 2015: IGD with the part of each distance that
    lies on the dominated side removed, so it is weakly Pareto compliant.
    ``moocore``, as ``pylib/moo_metrics.py``.

And the comparison of two methods over independent runs: the two-sided
Mann–Whitney U test and the Vargha–Delaney Â12 effect size.

All metrics assume *minimisation* of every objective.
"""
from __future__ import annotations

import math

import moocore
import numpy as np


def non_dominated(front: np.ndarray) -> np.ndarray:
    """The non-dominated rows of ``front``, one copy of each.

    Every method is measured on this set of its final population: the runners
    return the whole population, and pymoo's own result is a different set.
    """
    a = np.asarray(front, dtype=float)
    if a.size == 0:
        return a
    return a[moocore.is_nondominated(a)]


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """HV of `front` w.r.t. `ref_point` (worst corner). Points that do not
    dominate it add nothing."""
    if front.size == 0:
        return 0.0
    return float(moocore.hypervolume(np.asarray(front, dtype=float), ref=np.asarray(ref_point, dtype=float)))


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


def igd_plus(front: np.ndarray, ref_front: np.ndarray) -> float:
    """
    IGD+ = mean over reference points r of min over a of ||max(a - r, 0)||.
    Smaller is better. It replaces the coverage C(A, R) against the true
    front, which is 0 by construction: no feasible point dominates a
    Pareto-optimal one.
    """
    if front.size == 0 or ref_front.size == 0:
        return float("nan")
    return float(moocore.igd_plus(np.asarray(front, dtype=float), ref=np.asarray(ref_front, dtype=float)))


# ── Multi-run statistics ─────────────────────────────────────────────────────
# Standalone copies of the platform helpers in pareto-analysis/lib/stats.py: a
# single run (or only the mean of several) cannot support "method A beats B".

def summary(values) -> dict:
    """Return ``{n, mean, std, ci95}`` (sample std, ddof=1; 95% CI half-width, Student's t)."""
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return {"n": n, "mean": float(a.mean()), "std": std, "ci95": 0.0}
    from scipy.stats import t as student_t   # Student's t, as pareto-analysis/lib/stats.py
    return {"n": n, "mean": float(a.mean()), "std": std,
            "ci95": float(student_t.ppf(0.975, n - 1)) * std / math.sqrt(n)}


def mann_whitney_pvalue(a, b) -> float:
    """Two-sided Mann–Whitney U p-value of two independent samples.

    The runs of two methods share nothing but a seed number (each library
    draws from its own generator), so a paired test has nothing to pair.
    Exact when there are no ties and the samples are small. ``nan`` when a
    sample is empty or every value is equal.
    """
    from scipy.stats import mannwhitneyu

    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0 or np.unique(np.concatenate([a, b])).size == 1:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def a12_lower(a, b) -> float:
    """Vargha–Delaney Â12 for a metric where lower is better: the probability
    that a run of ``a`` scores lower than a run of ``b``, ties counting half.
    0.5 is no difference; Vargha and Delaney call 0.56, 0.64 and 0.71 (and
    their mirrors) small, medium and large."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lower = (a[:, None] < b[None, :]).sum()
    ties = (a[:, None] == b[None, :]).sum()
    return float((lower + 0.5 * ties) / (a.size * b.size))


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
