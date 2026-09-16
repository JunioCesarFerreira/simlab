from typing import Sequence, TypeVar
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
    rng: random.Random
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
                chosen = niching_selection(
                    front, objectives, reference_points, n_needed, rng, accepted=next_idx
                )
                next_idx.extend(chosen)
            break

    return [population[i] for i in next_idx]


def _find_extreme_points(F: NDArray, ideal: NDArray) -> NDArray:
    """The M points of *F* that best represent each objective axis.

    Achievement scalarizing function with a near-axis weight vector (1 on the
    objective, 1e6 on the others): its minimiser is the point that reaches
    furthest along that axis without being extreme on the rest. Deb & Jain
    (2014), Algorithm 2.
    """
    translated = F - ideal
    weights = np.eye(F.shape[1])
    weights[weights == 0.0] = 1e6
    asf = np.max(translated * weights[:, np.newaxis, :], axis=2)
    return F[np.argmin(asf, axis=1), :]


def _find_intercepts(extreme_points: NDArray, ideal: NDArray, worst: NDArray) -> NDArray:
    """Where the hyperplane through the extreme points cuts each objective axis.

    Returns the intercepts in ABSOLUTE coordinates, so the normalisation range
    is always ``intercepts - ideal``.

    Falls back to the worst observed value per objective whenever the hyperplane
    is degenerate: linearly dependent extreme points, a zero component in the
    solution, an intercept at or below the ideal point, or one past the worst
    observed value. Those cases are not exotic — a population collapsed onto a
    face of the simplex hits them routinely early in a run — and without the
    guards the normalisation explodes.
    """
    b = np.ones(extreme_points.shape[1])
    A = extreme_points - ideal
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return worst
    if np.count_nonzero(x) != len(x):
        return worst
    intercepts = 1.0 / x + ideal
    if (
        not np.allclose(A @ x, b)
        or np.any(intercepts - ideal <= 1e-6)
        or np.any(intercepts > worst)
    ):
        return worst
    return intercepts


def niching_selection(
    front: Sequence[int],
    objectives: Sequence[ObjectiveVec],
    reference_points: NDArray[np.float64],
    N: int,
    rng: random.Random,
    accepted: Sequence[int] = (),
) -> list[int]:
    """NSGA-III niching: pick N of *front* to complete the next population.

    Deb & Jain (2014), Algorithms 1-4, following the reference implementation in
    ``deap.tools.selNSGA3``.

    *accepted* holds the indices already taken from the earlier, complete
    fronts. It is not optional context — the algorithm is defined over
    ``St = accepted ∪ front``:

    * the ideal point, the extreme points and the intercepts describe the
      population being assembled, not the one front being truncated;
    * niche occupancy starts from the individuals already accepted, so a
      reference direction that is already crowded does not win the next pick.

    Passing nothing reduces it to niching over the front alone, which is only
    correct when the front IS the whole selection.

    Returns indices into *objectives*, taken from *front*.
    """
    if N <= 0 or not front:
        return []

    front = list(front)
    accepted = list(accepted)
    if N >= len(front):
        return front

    pool = accepted + front
    F = np.asarray([objectives[i] for i in pool], dtype=float)
    M = F.shape[1]
    if reference_points.ndim != 2 or reference_points.shape[1] != M:
        raise ValueError("reference_points must have shape [K, M] matching objectives dimension M.")

    ideal = np.min(F, axis=0)
    worst = np.max(F, axis=0)
    intercepts = _find_intercepts(_find_extreme_points(F, ideal), ideal, worst)
    niches, distances = associate_to_niches(F, reference_points, ideal, intercepts)

    niche_count = np.zeros(reference_points.shape[0], dtype=np.int64)
    for niche in niches[: len(accepted)]:
        niche_count[niche] += 1

    # Candidates are the truncated front's slice of the pool.
    candidate_niches = niches[len(accepted):]
    candidate_distances = distances[len(accepted):]
    available = np.ones(len(front), dtype=bool)

    selected: list[int] = []
    while len(selected) < N:
        # Only niches that still hold an available candidate compete for the
        # minimum occupancy. Leaving the empty ones in used to let them own the
        # minimum forever, which pushed the loop into a uniform random pick over
        # everything left — a pick that did not even update the occupancy.
        open_niches = np.unique(candidate_niches[available])
        if open_niches.size == 0:
            break
        min_count = niche_count[open_niches].min()
        least_crowded = [int(r) for r in open_niches if niche_count[r] == min_count]
        rng.shuffle(least_crowded)

        for niche in least_crowded[: N - len(selected)]:
            members = np.flatnonzero((candidate_niches == niche) & available)
            if niche_count[niche] == 0:
                # Seed an empty niche with its closest candidate: that is what
                # pulls the population towards an unrepresented direction.
                pick = int(members[np.argmin(candidate_distances[members])])
            else:
                # An occupied niche takes a random member instead. Always taking
                # the closest would keep stacking the same spot on a direction
                # that is already represented.
                pick = int(rng.choice(members.tolist()))
            available[pick] = False
            niche_count[niche] += 1
            selected.append(front[pick])
            if len(selected) >= N:
                break

    return selected[:N]


def associate_to_niches(
    F: NDArray,
    reference_points: NDArray,
    ideal: NDArray,
    intercepts: NDArray,
) -> tuple[NDArray, NDArray]:
    """Associate each row of *F* with its nearest reference direction.

    Distance is PERPENDICULAR to the direction — the ray from the origin through
    the reference point — not Euclidean to the reference point itself. A
    solution far out along a direction is perfectly aligned with it however
    distant that reference point is; measuring to the point instead hands the
    solution to whichever reference happens to sit nearby, which scrambles the
    niches precisely where the front is sparse. Deb & Jain (2014), Algorithm 3.

    *ideal* and *intercepts* come from the whole selection pool, so a solution's
    association does not depend on which front it arrived in.

    Returns
    -------
    (niche index per row, perpendicular distance to that niche)
    """
    F = np.asarray(F, dtype=np.float64)
    H = np.asarray(reference_points, dtype=np.float64)

    if F.ndim != 2 or H.ndim != 2:
        raise ValueError("F and reference_points must be 2D arrays")
    if F.shape[1] != H.shape[1]:
        raise ValueError(
            f"Dimension mismatch: F has M={F.shape[1]} but reference_points has M={H.shape[1]}"
        )
    if F.shape[0] == 0 or H.shape[0] == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    scale = np.asarray(intercepts, dtype=np.float64) - np.asarray(ideal, dtype=np.float64)
    scale = np.where(np.abs(scale) > np.finfo(float).eps, scale, np.finfo(float).eps)
    normalized = (F - ideal) / scale

    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms[norms <= 0.0] = np.finfo(float).eps
    unit = H / norms

    # Perpendicular component: f - (f·ĥ)ĥ, for every (solution, direction) pair.
    projection = (normalized @ unit.T)[:, :, np.newaxis] * unit[np.newaxis, :, :]
    d_perp = np.linalg.norm(normalized[:, np.newaxis, :] - projection, axis=2)

    niche_idx = np.argmin(d_perp, axis=1)
    niche_dist = d_perp[np.arange(F.shape[0]), niche_idx]
    return niche_idx.astype(int), niche_dist.astype(float)
