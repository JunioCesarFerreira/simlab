from typing import Sequence, TypeVar
import random

T = TypeVar("T")  

def two_point_crossover(a: Sequence, b: Sequence, rng: random.Random) -> tuple[list, list]:
    """Two-point crossover for indexable sequences."""
    n = min(len(a), len(b))
    if n < 2:
        return list(a), list(b)
    i, j = sorted((rng.randrange(1, n), rng.randrange(1, n)))
    return (list(a[:i]) + list(b[i:j]) + list(a[j:]),
            list(b[:i]) + list(a[i:j]) + list(b[j:]))