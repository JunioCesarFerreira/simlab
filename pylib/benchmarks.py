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


def ideal(bench: Bench, M: int) -> list[float]:
    """Best-corner (ideal) point of the true front.

    Together with :func:`nadir` it gives the benchmark's THEORETICAL
    ideal-nadir range. Normalising indicators by that range instead of by the
    extremes of a sampled reference front makes the numbers independent of how
    the reference happened to be drawn, and comparable across runs.
    """
    b = bench.upper()
    if b in ("DTLZ2",):
        return [0.0] * M
    if b in ("ZDT1", "SCH1"):
        return [0.0, 0.0]
    raise ValueError(f"Unknown benchmark '{bench}'. Valid: {', '.join(_KNOWN)}.")


def _nearest_on_curve(points, to_point, t_lo: float, t_hi: float,
                      grid: int = 2048, iterations: int = 60):
    """Distance from each row of *points* to a 1-D parametric curve.

    Bracket the minimiser on a coarse grid, then ternary-search inside the
    bracket. The bracket shrinks by a third each iteration, so 60 rounds take a
    ~1e-3 interval down to machine precision. Vectorised over all points.
    """
    import numpy as np

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


def front_distance(bench: Bench, points, M: int):
    """Exact distance from each row of *points* to the true Pareto front.

    This is what a sampled reference front only approximates, and the
    approximation is poor exactly where it matters. Generational distance is the
    mean of these values, so with a sampled reference its floor is the sample's
    fill distance — which shrinks as ``N**(-1/(M-1))``. Measured on points lying
    EXACTLY on the DTLZ2 front, a 500-point reference reports GD 0.0016 at M=2,
    0.030 at M=3 and 0.194 at M=6; raising it to 200 000 points still leaves
    0.051 at M=6. No practical sample fixes that. These closed forms do: the
    same points come back at ~5e-17.

    * DTLZ2 — the front is the unit hypersphere in the first orthant, so for any
      point of that orthant the nearest front point is ``f/‖f‖`` and the
      distance is exactly ``|‖f‖₂ − 1|``.
    * ZDT1 / SCH1 — plane curves, solved by parametric minimisation to machine
      precision.

    Only the distance is closed-form. IGD and IGD+ average over the REFERENCE
    set, so they still need one and keep the discretisation floor; swapping in
    this distance would not measure coverage.
    """
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"points must be 2-D, got shape {pts.shape}.")

    b = bench.upper()
    if b == "DTLZ2":
        if pts.shape[1] != M:
            raise ValueError(f"points have width {pts.shape[1]}, expected M={M}.")
        return np.abs(np.linalg.norm(pts, axis=1) - 1.0)
    if pts.shape[1] != 2:
        raise ValueError(f"{b} is a two-objective benchmark; points have width {pts.shape[1]}.")
    if b == "ZDT1":
        # Parametrised by u with f1 = u², not by f1 directly: 1 - sqrt(f1) has a
        # vertical tangent at the origin, so equal steps in f1 are wildly uneven
        # steps along the curve and the search stalls there. In u the curve is
        # polynomial and evenly conditioned end to end.
        return _nearest_on_curve(
            pts, lambda u: np.column_stack([u ** 2, 1.0 - u]), 0.0, 1.0
        )
    if b == "SCH1":
        return _nearest_on_curve(
            pts, lambda t: np.column_stack([t ** 2, (t - 2.0) ** 2]), 0.0, 2.0
        )
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
