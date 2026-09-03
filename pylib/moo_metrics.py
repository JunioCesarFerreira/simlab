"""Convergence quality indicators: GD, IGD and IGD+ against a reference front.

Single source of truth for the indicators the platform reports, so the live
endpoint (``GET /experiments/{id}/hv-gd``) and the offline analysis CLIs can
never drift apart on a definition.

Everything here works in **minimisation space**: maximised objectives must be
negated by the caller before they get in. That is the same convention the
hypervolume path already uses, and it lets IGD+ (whose ``d+`` is orientation
sensitive) be applied without a per-axis orientation argument.

Definitions
-----------
For an approximation set ``A`` and a reference front ``R``, with ``d(a, r)``
the Euclidean distance:

* ``GD(A, R)  = (1/|A|) · Σ_{a∈A} min_{r∈R} d(a, r)``   — how close the
  population got to the reference (convergence).
* ``IGD(A, R) = (1/|R|) · Σ_{r∈R} min_{a∈A} d(r, a)``   — how well the
  population covers the reference (convergence *and* spread).
* ``IGD+`` replaces ``d`` with the Pareto-compliant ``d+`` of Ishibuchi et al.
  (2015), which only counts the components where ``a`` is worse than ``r``.
  Plain IGD is not even weakly Pareto compliant, so IGD+ is reported alongside
  it as the compliant reading of the same comparison.

The ``p = 1`` (arithmetic mean) form is used for both GD and IGD, matching
``moocore``, ``pymoo`` and jMetal. Note the asymmetry that makes reporting both
worthwhile: a single point sitting exactly on ``R`` has ``GD = 0`` but a large
IGD, because it covers almost none of the reference.

Normalisation
-------------
Raw distances are dominated by whichever objective has the largest magnitude —
a latency in milliseconds and a throughput in kbps are not commensurable. So by
default both sets are mapped through the affine transform derived from the
*reference* front's ideal and nadir points::

    z' = (z - ideal_R) / (nadir_R - ideal_R)

The reference is fixed for a whole experiment, so the transform is too: the
resulting curves stay comparable generation to generation, which a
population-derived normalisation would not be. Hypervolume is unaffected — it
keeps its own reference point in raw units.

References
----------
* Van Veldhuizen & Lamont (1998) — Generational Distance.
* Coello & Sierra (2004) — Inverted Generational Distance.
* Ishibuchi, Masuda, Tanigaki & Nojima (2015) — IGD+.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import moocore

# Objectives at or above this magnitude mark an infeasible ("penalised")
# individual rather than a measurement. Mirrors the REST API's
# ``_PENALTY_THRESHOLD`` and the frontend's ``PENALTY_THRESHOLD``.
PENALTY_THRESHOLD = 1.0e8


def is_penalized(objectives: Sequence[float], threshold: float = PENALTY_THRESHOLD) -> bool:
    """True if any objective is at or beyond the infeasibility marker."""
    return any(abs(float(v)) >= threshold for v in objectives)


def sanitize_reference_front(
    rows: Iterable[Sequence[float]],
    *,
    penalty_threshold: float = PENALTY_THRESHOLD,
) -> np.ndarray:
    """Turn raw stored rows into a usable reference front (minimisation space).

    Drops penalised rows, deduplicates, and keeps only the non-dominated ones.

    All three steps are load-bearing for IGD, and the first one especially:
    IGD averages over the *reference*, so a single penalised row at 1e9 drags
    the mean to 1e9 and destroys the indicator. GD hides the same contamination
    — it minimises over the reference, so a penalised row never wins the
    minimum — which is exactly why the problem can sit unnoticed until IGD is
    reported. A dominated row is a subtler version of the same bias: IGD would
    charge the population for failing to cover a point that is not on the front
    at all.

    Returns an ``(k, m)`` array, possibly with ``k == 0`` when nothing survives.
    """
    kept: list[list[float]] = []
    seen: set[tuple] = set()
    for row in rows:
        vals = [float(v) for v in row]
        if is_penalized(vals, penalty_threshold):
            continue
        key = tuple(vals)
        if key in seen:
            continue
        seen.add(key)
        kept.append(vals)

    if not kept:
        return np.empty((0, 0), dtype=float)

    arr = np.asarray(kept, dtype=float)
    return np.asarray(moocore.filter_dominated(arr), dtype=float)


def normalization_bounds(reference_front: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ideal point and per-axis scale derived from ``reference_front``.

    An axis on which the reference front is degenerate (every point equal, as
    on a single-point front) gets a unit scale, so the transform stays a pure
    translation there instead of dividing by zero.
    """
    ref = np.asarray(reference_front, dtype=float)
    ideal = ref.min(axis=0)
    scale = ref.max(axis=0) - ideal
    scale = np.where(scale > 0, scale, 1.0)
    return ideal, scale


def normalize(points: np.ndarray, ideal: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply the affine transform from :func:`normalization_bounds`."""
    return (np.asarray(points, dtype=float) - ideal) / scale


def _prepare(
    front: np.ndarray,
    reference: np.ndarray,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate both sets and, when asked, map them into normalised space."""
    a = np.asarray(front, dtype=float)
    r = np.asarray(reference, dtype=float)
    if a.size == 0 or r.size == 0:
        return None
    if a.ndim != 2 or r.ndim != 2 or a.shape[1] != r.shape[1]:
        raise ValueError(
            f"front {a.shape} and reference {r.shape} must be 2-D with matching width"
        )
    if not normalized:
        return a, r
    ideal, scale = normalization_bounds(r)
    return normalize(a, ideal, scale), normalize(r, ideal, scale)


def gd(front: np.ndarray, reference: np.ndarray, *, normalized: bool = True) -> float | None:
    """Generational Distance of ``front`` to ``reference``. ``None`` if either is empty."""
    prepared = _prepare(front, reference, normalized)
    if prepared is None:
        return None
    a, r = prepared
    # moocore.igd(data, ref) averages over ``ref`` the distance to ``data``,
    # so GD is that same primitive with the roles swapped.
    return float(moocore.igd(r, ref=a))


def igd(front: np.ndarray, reference: np.ndarray, *, normalized: bool = True) -> float | None:
    """Inverted Generational Distance of ``front`` to ``reference``."""
    prepared = _prepare(front, reference, normalized)
    if prepared is None:
        return None
    a, r = prepared
    return float(moocore.igd(a, ref=r))


def igd_plus(front: np.ndarray, reference: np.ndarray, *, normalized: bool = True) -> float | None:
    """Pareto-compliant IGD+ of ``front`` to ``reference`` (Ishibuchi et al., 2015)."""
    prepared = _prepare(front, reference, normalized)
    if prepared is None:
        return None
    a, r = prepared
    return float(moocore.igd_plus(a, ref=r))
