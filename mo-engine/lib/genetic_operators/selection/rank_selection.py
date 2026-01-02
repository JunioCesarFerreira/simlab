from typing import Sequence, TypeVar
import random

T = TypeVar("T")  

def rank_selection(
    population: Sequence[T],
    scores: Sequence[float],
    k: int,
    rng: random.Random,
) -> list[int]:
    """Linear rank-based selection: probability ∝ rank (1..n)."""
    order = sorted(range(len(population)), key=lambda i: scores[i])
    # lowest score -> rank 1; highest -> rank n
    ranks = {idx: r + 1 for r, idx in enumerate(order)}
    total = sum(ranks.values())
    probs = [ranks[i] / total for i in range(len(population))]

    # CDF sampling
    cdf: list[float] = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)
    picks: list[int] = []
    for _ in range(k):
        u = rng.random()
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        picks.append(lo)
    return picks
