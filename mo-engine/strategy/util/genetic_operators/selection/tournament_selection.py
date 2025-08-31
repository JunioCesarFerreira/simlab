from typing import Sequence, TypeVar
import random

T = TypeVar("T")

def tournament_selection(
    population: Sequence[T],
    scores: Sequence[float],
    k: int,
    rng: random.Random,
    tournament_size: int = 3,
) -> list[int]:
    """Pick k indices using t-way tournament (higher score wins)."""
    n = len(population)
    sel: list[int] = []
    for _ in range(k):
        cand_idx = [rng.randrange(n) for _ in range(tournament_size)]
        best = max(cand_idx, key=lambda i: scores[i])
        sel.append(best)
    return sel