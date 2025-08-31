from typing import Sequence, TypeVar
from math import isfinite, inf

T = TypeVar("T")  # individual type (e.g., list[float], custom dataclass, etc.)
ObjectiveVec = Sequence[float]


def crowding_distance(front: Sequence[int], objectives: Sequence[ObjectiveVec]) -> list[float]:
    """
    NSGA-II crowding distance for a given Pareto front.

    Computes the crowding distance per Deb et al.:
    - Boundary solutions (w.r.t. each objective) receive +inf.
    - Interior solutions sum normalized gaps to neighbors along each objective.
    Distances are returned aligned with the *order of indices in `front`*.
    """
    n = len(front)
    if n == 0:
        return []
    if n == 1:
        return [inf]
    # For two points, both are boundaries for any sorted projection.
    if n == 2:
        return [inf, inf]

    m = len(objectives[0])
    # Distances are kept in the index-space of `front` (0..n-1)
    dist = [0.0] * n

    for obj_idx in range(m):
        # Values of this objective on the front
        vals = [objectives[idx][obj_idx] for idx in front]

        # Stable sort: indices of `front` ordered by objective value
        order = sorted(range(n), key=lambda i: vals[i])

        # Mark boundaries as infinite
        dist[order[0]] = inf
        dist[order[-1]] = inf

        vmin, vmax = vals[order[0]], vals[order[-1]]
        denom = vmax - vmin

        if not isfinite(denom) or denom == 0.0:
            # Degenerate objective (all equal or invalid) -> contributes nothing
            continue

        # Add normalized crowding contribution
        for k in range(1, n - 1):
            left_v = vals[order[k - 1]]
            right_v = vals[order[k + 1]]
            dist[order[k]] += (right_v - left_v) / denom

    return dist


def select_next_population(
    fronts: Sequence[Sequence[int]],
    objectives: Sequence[ObjectiveVec],
    population: Sequence[T],
    pop_size: int,
) -> list[T]:
    """
    NSGA-II elitist environmental selection.

    Takes the ranked fronts and fills the next population:
    - Add whole fronts while there is room.
    - For the last partial front, pick by descending crowding distance.
    Returns the selected individuals (not the indices).
    """
    selected_idx: list[int] = []

    for front in fronts:
        if len(selected_idx) + len(front) <= pop_size:
            selected_idx.extend(front)
        else:
            # Tie-break within the partial front using crowding distance
            dist = crowding_distance(front, objectives)  # aligned to order in `front`
            # Sort indices of `front` by distance (desc), keep most diverse first
            order = sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
            remaining = pop_size - len(selected_idx)
            selected_idx.extend(front[i] for i in order[:remaining])
            break

    return [population[i] for i in selected_idx]