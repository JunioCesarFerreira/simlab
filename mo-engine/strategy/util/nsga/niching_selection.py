from typing import Sequence, TypeVar
from collections import defaultdict
import random
import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")  # individual type (e.g., list[float], custom dataclass, etc.)
ObjectiveVec = Sequence[float]


def generate_reference_points(M: int, p: int) -> NDArray[np.float64]:
    """
    Generate Das & Dennis (1998) simplex-lattice reference points for NSGA-III.

    Parameters
    ----------
    M : int
        Number of objectives (dimension of the reference points).
    p : int
        Divisions per objective (granularity). The number of points equals
        C(M + p - 1, p).

    Returns
    -------
    np.ndarray (shape: [num_points, M], dtype float64)
        Barycentric coordinates on the unit simplex that sum to 1.0.
    """
    if M <= 0 or p <= 0:
        raise ValueError("M and p must be positive integers.")

    points: list[list[float]] = []

    def _rec(left: int, depth: int, current: list[float]) -> None:
        # At the last dimension, assign the remainder to close the simplex (sum == 1).
        if depth == M - 1:
            current.append(left / p)
            points.append(current.copy())
            current.pop()
            return

        # Distribute integer mass 'left' across remaining dimensions.
        for i in range(left + 1):
            current.append(i / p)
            _rec(left - i, depth + 1, current)
            current.pop()

    _rec(left=p, depth=0, current=[])
    # Convert to array; rows sum numerically to ~1.0
    return np.asarray(points, dtype=np.float64)


def environmental_selection(
    population: Sequence[T],
    objectives: Sequence[ObjectiveVec],
    fronts: Sequence[Sequence[int]],
    reference_points: NDArray[np.float64],
    pop_size: int,
) -> list[T]:
    """
    NSGA-III environmental selection (simplified).

    Fills the next generation by:
    1) Adding whole fronts while capacity remains.
    2) For the partial last front, selecting solutions via niching against
       the provided reference points.

    Returns the selected individuals (not indices).
    """
    if pop_size <= 0:
        return []

    next_idx: list[int] = []
    for front in fronts:
        if len(next_idx) + len(front) <= pop_size:
            next_idx.extend(front)
        else:
            n_needed = pop_size - len(next_idx)
            if n_needed > 0:
                chosen = niching_selection(front, objectives, reference_points, n_needed)
                next_idx.extend(chosen)
            break

    return [population[i] for i in next_idx]


def niching_selection(
    front: Sequence[int],
    objectives: Sequence[ObjectiveVec],
    reference_points: NDArray[np.float64],
    N: int,
) -> list[int]:
    """
    NSGA-III niching selection (simplified Euclidean association).

    Steps:
    - Compute the ideal point on the given front and normalize objectives by
      subtracting the ideal and dividing by per-objective range.
    - Associate each solution to its nearest reference point (Euclidean distance
      in the normalized space).
    - Iteratively pick solutions from the least crowded niches; within a niche,
      prefer the one closest to the reference point (smallest distance).

    Returns
    -------
    list[int]
        Indices (from the original population) of selected solutions, up to N.
    """
    if N <= 0 or not front:
        return []

    # If front size is already small, just return all (guard).
    if N >= len(front):
        return list(front)

    objs = np.asarray([objectives[i] for i in front], dtype=float)
    M = objs.shape[1]
    if reference_points.ndim != 2 or reference_points.shape[1] != M:
        raise ValueError("reference_points must have shape [K, M] matching objectives dimension M.")

    # Ideal point and range normalization (avoid division by zero).
    ideal = np.min(objs, axis=0)
    shifted = objs - ideal
    ranges = np.max(shifted, axis=0)
    ranges[ranges == 0.0] = 1.0
    norm_objs = shifted / ranges

    # Associate each candidate to nearest reference point.
    associations: list[tuple[int, int, float]] = []  # (idx_in_pop, ref_idx, distance)
    for idx_in_pop, vec in zip(front, norm_objs):
        dists = np.linalg.norm(reference_points - vec, axis=1)
        j = int(np.argmin(dists))
        associations.append((idx_in_pop, j, float(dists[j])))

    # Group by reference point.
    ref_to_candidates: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for pid, ridx, d in associations:
        ref_to_candidates[ridx].append((pid, d))

    # Niching loop: pick from least crowded niches first.
    selected: list[int] = []
    selected_flag: dict[int, bool] = {i: False for i in front}
    niche_count: dict[int, int] = {r: 0 for r in range(reference_points.shape[0])}

    while len(selected) < N:
        # Find minimum occupancy across niches.
        min_occ = min(niche_count.values())
        # Consider all reference points with that occupancy (randomized order avoids bias).
        least_crowded = [r for r, c in niche_count.items() if c == min_occ]
        random.shuffle(least_crowded)

        picked_this_round = False
        for r in least_crowded:
            candidates = [(pid, d) for (pid, d) in ref_to_candidates.get(r, []) if not selected_flag[pid]]
            if candidates:
                # Choose closest to the reference point within this niche.
                candidates.sort(key=lambda x: x[1])
                chosen_pid = candidates[0][0]
                selected.append(chosen_pid)
                selected_flag[chosen_pid] = True
                niche_count[r] += 1
                picked_this_round = True
                if len(selected) >= N:
                    break

        if not picked_this_round:
            # If all least-crowded niches are empty, pick uniformly from remaining.
            remaining = [pid for pid, flag in selected_flag.items() if not flag]
            if not remaining:
                break
            chosen_pid = random.choice(remaining)
            selected.append(chosen_pid)
            selected_flag[chosen_pid] = True
            # (Optionally, increment the niche of its association if you keep it.)

    return selected[:N]
