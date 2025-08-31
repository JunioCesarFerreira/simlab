from typing import Sequence, TypeVar
import random

T = TypeVar("T") 
 
def uniform_crossover(a: Sequence, b: Sequence, rng: random.Random, p_swap: float = 0.5) -> tuple[list, list]:
    """Uniform crossover: per-gene swapping with prob p_swap."""
    n = min(len(a), len(b))
    ca, cb = list(a), list(b)
    for i in range(n):
        if rng.random() < p_swap:
            ca[i], cb[i] = cb[i], ca[i]
    return ca, cb