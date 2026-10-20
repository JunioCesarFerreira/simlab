"""Convergence quality indicators: GD, IGD and IGD+ against a reference front.

Standalone mirror of the canonical ``pylib.moo_metrics``, which is what the live
``GET /experiments/{id}/hv-gd`` endpoint uses. The copy exists on purpose — the
CLIs in this directory must run with no ``pylib`` on the path — so
``tests/test_metrics_parity.py``, not a shared import, is what keeps the two
from drifting apart. That mirrors what ``lib/true_fronts.py`` already does for
``pylib.benchmarks``.

Any change here must be made in ``pylib/moo_metrics.py`` as well. See that
module for the full definitions and references; the short version:

* ``GD(A, R)``  — mean over ``A`` of the distance to the nearest point of ``R``.
* ``IGD(A, R)`` — mean over ``R`` of the distance to the nearest point of ``A``.
* ``IGD+``      — the Pareto-compliant variant (Ishibuchi et al., 2015).

Everything works in minimisation space, and distances are normalised by the
reference front's ideal-nadir range by default.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import moocore

PENALTY_THRESHOLD = 1.0e8


def is_penalized(objectives: Sequence[float], threshold: float = PENALTY_THRESHOLD) -> bool:
    """True if any objective is at or beyond the infeasibility marker."""
    return any(abs(float(v)) >= threshold for v in objectives)


def sanitize_reference_front(
    rows: Iterable[Sequence[float]],
    *,
    penalty_threshold: float = PENALTY_THRESHOLD,
) -> np.ndarray:
    """Drop penalised rows, deduplicate, keep only the non-dominated ones.

    IGD averages over the reference, so a single penalised row at 1e9 would
    drag the mean to 1e9. GD hides the same contamination because it minimises
    over the reference instead.
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

    A degenerate axis gets a unit scale, so the transform stays a pure
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


def analytical_scale(ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """Per-axis normalisation range from an ideal/nadir pair, zeros guarded."""
    scale = np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float)
    return np.where(scale > 0, scale, 1.0)


def _prepare(
    front: np.ndarray,
    reference: np.ndarray,
    normalized: bool,
    bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate both sets and, when asked, map them into normalised space.

    *bounds* is an explicit ``(ideal, nadir)`` pair — the benchmark's THEORETICAL
    range. Prefer it when it is known: deriving the range from the sampled
    reference front instead makes the indicator depend on how that sample
    happened to be drawn, and on a sparse sample the extremes fall short of the
    real ones. Without *bounds* the reference's own extremes are used, which is
    all a run with no analytical front has.
    """
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
    if bounds is None:
        ideal, scale = normalization_bounds(r)
    else:
        ideal = np.asarray(bounds[0], dtype=float)
        scale = analytical_scale(ideal, bounds[1])
    return normalize(a, ideal, scale), normalize(r, ideal, scale)


def gd(
    front: np.ndarray,
    reference: np.ndarray,
    *,
    normalized: bool = True,
    bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
) -> float | None:
    """Generational Distance of ``front`` to ``reference``. ``None`` if either is empty."""
    prepared = _prepare(front, reference, normalized, bounds)
    if prepared is None:
        return None
    a, r = prepared
    # moocore.igd(data, ref) averages over ``ref`` the distance to ``data``,
    # so GD is that same primitive with the roles swapped.
    return float(moocore.igd(r, ref=a))



def igd(
    front: np.ndarray,
    reference: np.ndarray,
    *,
    normalized: bool = True,
    bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
) -> float | None:
    """Inverted Generational Distance of ``front`` to ``reference``."""
    prepared = _prepare(front, reference, normalized, bounds)
    if prepared is None:
        return None
    a, r = prepared
    return float(moocore.igd(a, ref=r))



def igd_plus(
    front: np.ndarray,
    reference: np.ndarray,
    *,
    normalized: bool = True,
    bounds: "tuple[np.ndarray, np.ndarray] | None" = None,
) -> float | None:
    """Pareto-compliant IGD+ of ``front`` to ``reference`` (Ishibuchi et al., 2015)."""
    prepared = _prepare(front, reference, normalized, bounds)
    if prepared is None:
        return None
    a, r = prepared
    return float(moocore.igd_plus(a, ref=r))


def gd_analytical(distances: np.ndarray, *, scale: float = 1.0) -> float | None:
    """GD from exact per-point distances — see :func:`pylib.benchmarks.front_distance`.

    Same definition as :func:`gd`: the arithmetic mean of each point's distance
    to the front. It differs only in where the distances come from — a closed
    form rather than a nearest-neighbour search over a sampled reference, which
    removes that sample's discretisation floor (0.19 at M=6, and still 0.05 with
    200 000 reference points).

    *scale* maps the raw distances into normalised space and must be the UNIFORM
    ideal-nadir range: the closed forms are Euclidean in raw space, so only an
    isotropic scaling carries through exactly. All three benchmarks have one
    (DTLZ2 and ZDT1 unit, SCH1 four); a caller facing an anisotropic range must
    fall back to the sampled reference.
    """
    d = np.asarray(distances, dtype=float)
    if d.size == 0:
        return None
    if not scale > 0:
        raise ValueError(f"scale must be positive, got {scale}.")
    return float(np.mean(d) / scale)
