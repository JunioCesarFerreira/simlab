from typing import Sequence, TypeVar
import random

T = TypeVar("T")  

def roulette_selection(
    population: Sequence[T],
    scores: Sequence[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    """Fitness-proportionate selection (requires non-negative weights).
    If scores are not strictly positive, we shift them.
    """
    # Shift to positive weights
    mn = min(scores)
    weights = [s - mn + 1e-12 for s in scores]
    s = sum(weights)
    if s <= 0:
        # Fallback to uniform if all weights ~ 0
        return [rng.randrange(len(population)) for _ in range(k)]
    probs = [w / s for w in weights]

    # CDF sampling
    cdf: list[float] = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    picks: list[int] = []
    for _ in range(k):
        u = rng.random()
        # binary search
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        picks.append(lo)
    return picks
