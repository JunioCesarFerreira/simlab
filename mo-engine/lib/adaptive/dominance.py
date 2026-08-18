"""Objective-sense conversion and margin-aware dominance tests.

Every decision the adaptive heuristic takes is expressed in **minimization
space**.  The conversion is driven exclusively by the experiment's
``transform_config`` / ``objectives[].goal`` list — nothing here assumes that
latency is minimised or that throughput is maximised, and the helpers work for
any number of objectives.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: Sign applied to an objective so that "smaller is better" always holds.
GOAL_MIN = 1
GOAL_MAX = -1


def goal_signs(goals: Sequence[str]) -> list[int]:
    """Map ``["min", "max", ...]`` onto ``[+1, -1, ...]``.

    Raises ``ValueError`` on an unknown goal so a typo in the experiment
    document fails loudly instead of silently inverting an objective.
    """
    signs: list[int] = []
    for goal in goals:
        normalized = str(goal).strip().lower()
        if normalized == "min":
            signs.append(GOAL_MIN)
        elif normalized == "max":
            signs.append(GOAL_MAX)
        else:
            raise ValueError(f"Unsupported objective goal: {goal!r} (expected 'min' or 'max').")
    return signs


def to_minimization(values: Sequence[float], signs: Sequence[int]) -> list[float]:
    """Convert original-space objectives into minimization space."""
    if len(values) != len(signs):
        raise ValueError(
            f"Objective/goal length mismatch: {len(values)} values vs {len(signs)} goals."
        )
    return [float(v) * int(s) for v, s in zip(values, signs)]


def to_original(values: Sequence[float], signs: Sequence[int]) -> list[float]:
    """Inverse of :func:`to_minimization` (the signs are involutive)."""
    return to_minimization(values, signs)


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Standard Pareto dominance in minimization space (``a`` dominates ``b``)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return bool(np.all(a_arr <= b_arr) and np.any(a_arr < b_arr))


def dominates_clearly(
    a: Sequence[float],
    b: Sequence[float],
    margin: Sequence[float] | float = 0.0,
) -> bool:
    """``a`` beats ``b`` by at least ``margin`` in **every** objective.

    This is the conservative test the decision policy uses: an individual is
    only discarded from simulation when a known, really-evaluated solution is
    clearly better than its most optimistic estimate, never when the two merely
    tie.  The strict-improvement clause is what rules ties out — with
    ``margin = 0`` the test degrades exactly to :func:`dominates`.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    m_arr = np.broadcast_to(np.asarray(margin, dtype=float), a_arr.shape)
    return bool(np.all(a_arr + m_arr <= b_arr) and np.any(a_arr < b_arr))


def dominated_by_any(
    point: Sequence[float],
    reference: Sequence[Sequence[float]],
    margin: Sequence[float] | float = 0.0,
) -> bool:
    """Whether any vector in ``reference`` clearly dominates ``point``."""
    for candidate in reference:
        if dominates_clearly(candidate, point, margin):
            return True
    return False


def non_dominated(points: Sequence[Sequence[float]]) -> list[int]:
    """Indices of the non-dominated subset of ``points`` (minimization)."""
    keep: list[int] = []
    for i, p in enumerate(points):
        if not any(dominates(q, p) for j, q in enumerate(points) if j != i):
            keep.append(i)
    return keep


def objective_ranges(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Per-objective spread of ``points``; zero spreads become ``1.0``.

    Used to express thresholds (uncertainty, dominance margin) as fractions of
    the observed objective scale, so the same configuration works for latency
    in milliseconds and for energy in millijoules.
    """
    if not points:
        return np.ones(0, dtype=float)
    arr = np.asarray(points, dtype=float)
    spread = arr.max(axis=0) - arr.min(axis=0)
    spread[spread <= 0.0] = 1.0
    return spread


def lower_bound(mean: Sequence[float], sigma: Sequence[float], kappa: float) -> np.ndarray:
    """Optimistic view ``L(x) = mean - kappa * sigma`` (minimization)."""
    return np.asarray(mean, dtype=float) - float(kappa) * np.asarray(sigma, dtype=float)


def upper_bound(mean: Sequence[float], sigma: Sequence[float], kappa: float) -> np.ndarray:
    """Pessimistic view ``U(x) = mean + kappa * sigma`` (minimization)."""
    return np.asarray(mean, dtype=float) + float(kappa) * np.asarray(sigma, dtype=float)
