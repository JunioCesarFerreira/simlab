"""Canonical analytical multi-objective benchmarks (DTLZ2, ZDT1, SCH1).

Single source of truth for the synthetic-benchmark machinery, so the function
that is *evaluated* at runtime and the true Pareto front it is *compared
against* can never drift apart. Provides:

  * objective evaluation on a normalised decision vector x ∈ [0,1]^n
    (master-node synthetic evaluator and the in-process P0 adapter);
  * reproducible observation noise (seeded, clamped for non-negative objectives);
  * the analytical (true) Pareto front and its nadir point
    (quality-indicator endpoints and offline analysis).

All objectives are in MINIMISATION orientation, matching what the platform
stores in MongoDB. The evaluation helpers depend only on the stdlib; the
front/nadir helpers lazily import numpy.

References:
  * DTLZ2 — Deb, Thiele, Laumanns & Zitzler (2005).
  * ZDT1  — Zitzler, Deb & Thiele (2000).
  * SCH1  — Schaffer (1985).
"""
from __future__ import annotations

import math
from typing import Sequence

Bench = str  # one of {"DTLZ2", "ZDT1", "SCH1"} (case-insensitive)

# SCH1 maps x01[0] ∈ [0,1] linearly onto this decision domain. The Pareto-optimal
# set is x ∈ [0,2]; a domain of exactly (0,2) makes every point optimal
# (spread-only, no convergence to test). The default is deliberately wider so the
# optimiser must *converge* the population onto [0,2] and then spread along it.
# Callers may override per experiment via ``synthetic.sch1_domain``.
SCH1_DEFAULT_DOMAIN: tuple[float, float] = (-5.0, 5.0)

_KNOWN = ("DTLZ2", "ZDT1", "SCH1")


# ── Low-level objective functions (minimisation) ─────────────────────────────

def dtlz2(x: Sequence[float], M: int) -> list[float]:
    """DTLZ2 — Pareto front is the unit hypersphere segment in the first orthant.

    Requires ``len(x) >= M-1``; raises ``ValueError`` otherwise (instead of the
    latent ``IndexError`` the previous inline copy produced).
    """
    n = len(x)
    if n < M - 1:
        raise ValueError(f"DTLZ2 requires n >= M-1 = {M - 1} variables, got n={n} (M={M}).")
    # Standard DTLZ2 split: the first M-1 variables are position variables, the
    # remaining k = n-(M-1) are distance variables. k may be 0 (n == M-1), in
    # which case g ≡ 0 and every point lies exactly on the unit-sphere front —
    # matching pymoo/Deb. (A previous max(1, ...) here forced k=1, making the
    # last POSITION variable double as distance variable: with M=3, n=2 only
    # solutions with x1=0.5 — the f1=f2 arc — could reach the sphere.)
    k = n - (M - 1)
    g = sum((xi - 0.5) ** 2 for xi in x[n - k:]) if k > 0 else 0.0
    f: list[float] = []
    for m in range(M):
        val = 1.0 + g
        for i in range(0, M - 1 - m):
            val *= math.cos(0.5 * math.pi * x[i])
        if m > 0:
            val *= math.sin(0.5 * math.pi * x[M - 1 - m])
        f.append(float(val))
    return f


def zdt1(x: Sequence[float]) -> list[float]:
    """ZDT1 — convex Pareto front, exactly 2 objectives. f1=x1, f2=g·(1−√(f1/g))."""
    if not x:
        return [1.0, 1.0]
    f1 = float(x[0])
    g = 1.0 if len(x) == 1 else 1.0 + 9.0 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - math.sqrt(max(0.0, f1 / g)))
    return [f1, float(f2)]


def sch1(x01: Sequence[float], domain: tuple[float, float] = SCH1_DEFAULT_DOMAIN) -> list[float]:
    """SCH1 (Schaffer) — 2 objectives, 1 effective variable.

    ``x01[0] ∈ [0,1]`` maps linearly onto *domain*; f1=x², f2=(x−2)². The
    Pareto-optimal set is x ∈ [0,2].
    """
    lo, hi = domain
    t = x01[0] if x01 else 0.0
    x = lo + t * (hi - lo)
    return [float(x * x), float((x - 2.0) ** 2)]


# ── Structural metadata ──────────────────────────────────────────────────────

def min_variables(bench: Bench, M: int) -> int:
    """Minimum number of decision variables the benchmark needs for the given M."""
    b = bench.upper()
    if b == "DTLZ2":
        return max(1, M - 1)
    if b == "ZDT1":
        return 2
    if b == "SCH1":
        return 1
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: {', '.join(_KNOWN)}.")


def validate(bench: Bench, n: int, M: int) -> None:
    """Raise ``ValueError`` with a clear message when n is below the minimum."""
    need = min_variables(bench, M)
    if n < need:
        raise ValueError(
            f"{bench.upper()} requires n >= {need} decision variables (got n={n}, M={M})."
        )


def is_non_negative(bench: Bench) -> bool:
    """True when every objective of *bench* is >= 0 over its whole domain, so
    observation noise must be clamped at 0 to stay inside the objective range."""
    return bench.upper() in _KNOWN


# ── Dispatch: evaluation (+ optional reproducible noise) ─────────────────────

def evaluate(
    bench: Bench,
    x01: Sequence[float],
    M: int,
    sch1_domain: tuple[float, float] = SCH1_DEFAULT_DOMAIN,
) -> list[float]:
    """Evaluate *bench* on a normalised decision vector (deterministic)."""
    b = bench.upper()
    if b == "ZDT1":
        return zdt1(x01)
    if b == "SCH1":
        return sch1(x01, sch1_domain)
    return dtlz2(x01, max(2, int(M)))


def evaluate_noisy(
    bench: Bench,
    x01: Sequence[float],
    M: int,
    noise_std: float,
    rng,
    sch1_domain: tuple[float, float] = SCH1_DEFAULT_DOMAIN,
) -> list[float]:
    """Evaluate, then add reproducible Gaussian observation noise.

    *rng* is a ``random.Random`` seeded by the caller, so a fixed seed gives a
    fixed result. For non-negative benchmarks the noisy values are clamped at 0
    (negative objectives would corrupt HV/GD/IGD and the dominance relation).
    """
    vals = evaluate(bench, x01, M, sch1_domain=sch1_domain)
    if noise_std and noise_std > 0.0:
        vals = [v + rng.gauss(0.0, noise_std) for v in vals]
        if is_non_negative(bench):
            vals = [max(0.0, v) for v in vals]
    return vals


# ── Analytical (true) Pareto front + nadir ───────────────────────────────────

def true_front(bench: Bench, M: int, n_points: int = 500, seed: int = 0):
    """Return an ``(n_points, M)`` numpy array of true Pareto-front objective
    vectors (minimisation). Deterministic for a fixed *seed* (DTLZ2 only)."""
    import numpy as np

    b = bench.upper()
    if b == "DTLZ2":
        if M < 2:
            raise ValueError(f"DTLZ2 requires M >= 2, got {M}.")
        rng = np.random.default_rng(seed)
        pts = np.abs(rng.standard_normal((n_points, M)))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms
    if b == "ZDT1":
        f1 = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([f1, 1.0 - np.sqrt(f1)])
    if b == "SCH1":
        x = np.linspace(0.0, 2.0, n_points)
        return np.column_stack([x ** 2, (x - 2.0) ** 2])
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: {', '.join(_KNOWN)}.")


def nadir(bench: Bench, M: int) -> list[float]:
    """Worst-corner (nadir) of the true front — a FIXED hypervolume reference,
    making HV comparable across runs/experiments of the same benchmark."""
    b = bench.upper()
    if b == "DTLZ2":
        return [1.0] * M
    if b == "ZDT1":
        return [1.0, 1.0]
    if b == "SCH1":
        return [4.0, 4.0]
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: {', '.join(_KNOWN)}.")
