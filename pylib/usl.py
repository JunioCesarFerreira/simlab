"""The Universal Scalability Law, fitted to measured throughput.

Gunther's USL models the throughput of a system under concurrency ``N`` as

    X(N) = λ N / (1 + σ (N − 1) + κ N (N − 1))

* ``λ`` — the throughput of a single unit of concurrency, ideally ``X(1)``;
* ``σ`` — **contention**: the fraction of work that serialises on a shared
  resource (a disk, a lock, the dispatcher). With ``κ = 0`` this is Amdahl's
  law, and throughput approaches ``λ / σ`` without ever falling;
* ``κ`` — **coherency**: the cost of keeping parallel units consistent with
  each other, which grows with the *number of pairs*. Any ``κ > 0`` makes the
  curve retrograde: it peaks at ``N* = sqrt((1 − σ) / κ)`` and then *loses*
  throughput as concurrency grows.

Pure and dependency-free beyond NumPy, so it can be tested against curves
whose parameters are known, and so the API that serves it does not need a
scientific stack the rest of the service does not use.

How the fit works
-----------------
For fixed ``(σ, κ)`` the model is linear in ``λ``, so the best ``λ`` has a
closed form. That leaves a two-dimensional bounded search over ``σ ∈ [0, 1]``
and ``κ ≥ 0``, done as a grid followed by a few rounds of local refinement.
A gradient method would be faster and is not needed: the problem is tiny, and
a grid cannot wander outside the bounds or stop at a saddle.

``λ`` is fitted rather than pinned to the measured ``X(1)``. The usual
linearisation divides every point by ``X(1)``, so the noise of a single
measurement is copied into all the others; fitting it jointly lets every
level carry its own weight.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

#: Below this, a coherency coefficient is indistinguishable from zero for any
#: concurrency a SimLab pool can reach: its peak would sit past ``N = 10⁴``.
KAPPA_NEGLIGIBLE = 1e-8

#: The search range for κ, in log space. ``1`` puts the peak at ``N ≈ 1``,
#: which is already a system that cannot run two things at once.
_KAPPA_LOG_RANGE = (-8.0, 0.0)

#: Fit outcomes, published so a caller can tell *why* there is no curve.
FIT_OK = "ok"
#: Only two distinct concurrency levels: λ and σ are fitted with κ = 0,
#: because three parameters cannot be identified from two abscissae.
FIT_NO_COHERENCY = "amdahl_only"
#: Fewer than two distinct levels, or no positive measurement at all.
FIT_INSUFFICIENT = "insufficient_data"


@dataclass(frozen=True)
class UslFit:
    """A fitted USL curve and what it implies."""

    status: str
    lam: float
    sigma: float
    kappa: float
    #: Coefficient of determination over the points that were fitted.
    r2: Optional[float]
    rmse: Optional[float]
    n_points: int
    n_levels: int
    #: Where throughput peaks, when κ > 0. Real-valued — the integer
    #: concurrency worth running at is ``n_peak_int``.
    n_peak: Optional[float]
    n_peak_int: Optional[int]
    x_peak: Optional[float]
    #: The limit throughput approaches when there is no peak (κ = 0, σ > 0).
    x_asymptote: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


def throughput(n, lam: float, sigma: float, kappa: float):
    """``X(N)`` for a scalar or an array of concurrency levels."""
    n = np.asarray(n, dtype=float)
    return lam * n / (1.0 + sigma * (n - 1.0) + kappa * n * (n - 1.0))


def peak(sigma: float, kappa: float) -> Optional[float]:
    """``N* = sqrt((1 − σ) / κ)``, or ``None`` when the curve never turns.

    No peak below ``KAPPA_NEGLIGIBLE``: the formula would answer with a
    concurrency nobody will run, and reporting it would present a numerical
    artefact as a property of the infrastructure.
    """
    if kappa <= KAPPA_NEGLIGIBLE or sigma >= 1.0:
        return None
    return float(np.sqrt((1.0 - sigma) / kappa))


def best_integer_peak(lam: float, sigma: float, kappa: float) -> Optional[int]:
    """The integer concurrency with the highest modelled throughput.

    ``N*`` is real-valued; what an operator configures is a number of
    containers, and the better of its two integer neighbours is not always
    the nearest one.
    """
    n_star = peak(sigma, kappa)
    if n_star is None:
        return None
    candidates = sorted({max(1, int(np.floor(n_star))), max(1, int(np.ceil(n_star)))})
    values = throughput(candidates, lam, sigma, kappa)
    return int(candidates[int(np.argmax(values))])


def _best_lambda(g: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares λ for every candidate (σ, κ) at once.

    ``g`` has shape ``(candidates, points)``: the model with λ = 1. Minimising
    ``Σ (x − λ g)²`` over λ gives ``Σ x g / Σ g²``.
    """
    denominator = np.sum(g * g, axis=1)
    return np.where(denominator > 0, (g @ x) / np.where(denominator > 0, denominator, 1.0), 0.0)


def _search(n: np.ndarray, x: np.ndarray, sigmas: np.ndarray, kappas: np.ndarray):
    """Exhaustive search over a grid; returns ``(sse, λ, σ, κ)`` of the best."""
    s, k = np.meshgrid(sigmas, kappas, indexing="ij")
    s, k = s.ravel(), k.ravel()
    g = n[None, :] / (1.0 + s[:, None] * (n[None, :] - 1.0)
                      + k[:, None] * n[None, :] * (n[None, :] - 1.0))
    lam = np.maximum(_best_lambda(g, x), 0.0)
    sse = np.sum((x[None, :] - lam[:, None] * g) ** 2, axis=1)
    best = int(np.argmin(sse))
    return float(sse[best]), float(lam[best]), float(s[best]), float(k[best])


def fit(levels: Sequence[float], values: Sequence[float], *, rounds: int = 5) -> UslFit:
    """Fit ``λ, σ, κ`` to measured ``(N, X)`` pairs.

    Repeated levels are expected and welcome — each repetition is a point, so
    a level measured three times weighs three times as much as one measured
    once, which is what its evidence is worth.

    Points with a non-positive or non-finite throughput are dropped: they are
    failed measurements, not observations of a system with no throughput.
    """
    n_all = np.asarray(list(levels), dtype=float)
    x_all = np.asarray(list(values), dtype=float)
    if n_all.shape != x_all.shape:
        raise ValueError("levels and values must have the same length")
    keep = np.isfinite(n_all) & np.isfinite(x_all) & (n_all >= 1) & (x_all > 0)
    n, x = n_all[keep], x_all[keep]
    distinct = int(np.unique(n).size)

    if distinct < 2:
        return UslFit(FIT_INSUFFICIENT, 0.0, 0.0, 0.0, None, None,
                      int(n.size), distinct, None, None, None, None)

    # Two abscissae identify two parameters. κ is the one that needs the
    # curvature, so it is the one held at zero.
    fit_kappa = distinct >= 3
    status = FIT_OK if fit_kappa else FIT_NO_COHERENCY

    sigmas = np.linspace(0.0, 1.0, 101)
    kappas = (np.concatenate(([0.0], np.logspace(*_KAPPA_LOG_RANGE, 161)))
              if fit_kappa else np.array([0.0]))
    sse, lam, sigma, kappa = _search(n, x, sigmas, kappas)

    # Local refinement: a finer grid around the best point, shrinking each
    # round. κ is refined in log space where it is positive, because its
    # plausible values span eight orders of magnitude.
    sigma_step = 0.01
    log_step = (_KAPPA_LOG_RANGE[1] - _KAPPA_LOG_RANGE[0]) / 160
    for _ in range(rounds):
        sigmas = np.clip(np.linspace(sigma - sigma_step, sigma + sigma_step, 21), 0.0, 1.0)
        if fit_kappa:
            if kappa > 0:
                centre = np.log10(kappa)
                kappas = np.concatenate(([0.0], 10.0 ** np.linspace(
                    centre - log_step, centre + log_step, 21)))
            else:
                kappas = np.concatenate(([0.0], np.logspace(
                    _KAPPA_LOG_RANGE[0], _KAPPA_LOG_RANGE[0] + log_step, 11)))
        candidate = _search(n, x, np.unique(sigmas), np.unique(kappas))
        if candidate[0] <= sse:
            sse, lam, sigma, kappa = candidate
        sigma_step /= 5.0
        log_step /= 5.0

    if kappa <= KAPPA_NEGLIGIBLE:
        kappa = 0.0

    residual = x - throughput(n, lam, sigma, kappa)
    sst = float(np.sum((x - x.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / sst if sst > 0 else None
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    n_star = peak(sigma, kappa)
    n_star_int = best_integer_peak(lam, sigma, kappa)
    x_star = float(throughput(n_star_int, lam, sigma, kappa)) if n_star_int else None
    asymptote = lam / sigma if (kappa == 0.0 and sigma > 0.0) else None

    return UslFit(status, float(lam), float(sigma), float(kappa), r2, rmse,
                  int(n.size), distinct, n_star, n_star_int, x_star, asymptote)


def curve(fitted: UslFit, n_max: int, *, ideal: bool = True) -> list[dict]:
    """The modelled throughput at every integer ``N`` from 1 to ``n_max``.

    ``ideal`` adds ``λ N`` — linear scaling — so a reader sees how far below
    it the infrastructure sits, which a single curve cannot show.
    """
    if fitted.status == FIT_INSUFFICIENT or n_max < 1:
        return []
    ns = np.arange(1, int(n_max) + 1)
    modelled = throughput(ns, fitted.lam, fitted.sigma, fitted.kappa)
    out = []
    for n_value, x_value in zip(ns.tolist(), modelled.tolist()):
        row = {"n": int(n_value), "x": float(x_value)}
        if ideal:
            row["ideal"] = float(fitted.lam * n_value)
        out.append(row)
    return out


def display_range(levels: Iterable[float], fitted: UslFit) -> int:
    """How far to draw the curve so its shape is visible.

    Past the largest measured level — a model is worth drawing where nothing
    was measured, as long as the chart says so — and far enough to show the
    peak and some of the decline when there is one.
    """
    measured = max((int(n) for n in levels), default=1)
    upper = max(2 * measured, 4)
    if fitted.n_peak is not None:
        upper = max(upper, int(np.ceil(1.6 * fitted.n_peak)))
    # A coherency so large the peak is below 2 would otherwise draw a curve
    # that is mostly decline; a cap keeps the chart about the tested range.
    return int(min(upper, max(4 * measured, 16)))
