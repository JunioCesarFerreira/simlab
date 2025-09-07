from typing import Sequence, TypeVar
import random
import numpy as np

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

def tournament_selection_nsga(
    population: list[list[float]], 
    individual_ranks: dict[int, int]
    ) -> list[list[float]]:
        i1, i2 = random.sample(range(len(population)), 2)
        rank1: int = individual_ranks[i1]
        rank2: int = individual_ranks[i2]
        if rank1 < rank2:
            return population[i1]
        elif rank2 < rank1:
            return population[i2]
        else:
            return population[random.choice([i1, i2])]