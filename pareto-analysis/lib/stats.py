"""Statistical utilities for multi-run algorithm validation.

Comparing metaheuristic implementations (or the same implementation across
configurations) requires *multiple independent runs* and reporting central
tendency AND dispersion — mean ± std / 95% CI — plus a significance test
between methods. Reporting a single run, or only the mean of several, is not
enough to claim one method is better than another.

Implemented with numpy only; scipy is used opportunistically for an exact
Wilcoxon test when it happens to be installed.
"""
from __future__ import annotations

import math

import numpy as np


#: Two-sided 95% quantiles of Student's t for 1 to 30 degrees of freedom.
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def t975(df: int) -> float:
    """The 97.5% quantile of Student's t with ``df`` degrees of freedom.

    Tabulated up to 30, then the Cornish-Fisher expansion around the normal
    quantile, within 1e-3 of the exact value from 30 on. The normal quantile
    (1.96) understates the interval for few runs: by 31% with 5.
    """
    if df < 1:
        raise ValueError("df must be at least 1")
    if df <= len(_T975):
        return _T975[df - 1]
    z = 1.959964
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def summary(values) -> dict:
    """Return ``{n, mean, std, ci95}`` for a 1-D sample.

    ``std`` is the sample standard deviation (ddof=1); ``ci95`` is the half-width
    of the 95% confidence interval of the mean, with Student's t.
    """
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    ci95 = t975(n - 1) * std / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": float(a.mean()), "std": std, "ci95": ci95}


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of ``x`` with ties resolved by their average rank."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average of 1-based ranks i+1..j+1
        i = j + 1
    return ranks


def wilcoxon_pvalue(a, b) -> float:
    """Two-sided p-value of the paired Wilcoxon signed-rank test (a vs b).

    Prefers ``scipy.stats.wilcoxon`` when available; otherwise falls back to a
    numpy-only normal approximation. Returns ``nan`` when the samples have
    mismatched shapes or no non-zero paired differences.
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    d = a - b
    d = d[d != 0]
    n = int(d.size)
    if n == 0:
        return float("nan")
    try:  # exact / well-calibrated test when scipy is present
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        pass
    ranks = _average_ranks(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return float("nan")
    z = (w_plus - mean) / math.sqrt(var)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))  # two-sided normal approximation


def compare_to_baseline(per_run_by_method: dict, baseline: str) -> dict:
    """Summarise each method and test it against a baseline.

    Parameters
    ----------
    per_run_by_method:
        ``{method_name: [metric value for each independent run]}`` — the runs
        must be aligned by index across methods (same seed per position) for the
        paired test to be meaningful.
    baseline:
        Key in ``per_run_by_method`` used as the reference method.

    Returns ``{method: {n, mean, std, ci95, p_vs_baseline}}`` where
    ``p_vs_baseline`` is ``nan`` for the baseline itself.
    """
    if baseline not in per_run_by_method:
        raise KeyError(f"baseline method '{baseline}' not present in {list(per_run_by_method)}.")
    base = per_run_by_method[baseline]
    out: dict = {}
    for method, values in per_run_by_method.items():
        stats = summary(values)
        stats["p_vs_baseline"] = (
            float("nan") if method == baseline else wilcoxon_pvalue(values, base)
        )
        out[method] = stats
    return out
