from typing import Sequence, TypeVar
import random

T = TypeVar("T")  
    
def one_point_crossover(a: Sequence, b: Sequence, rng: random.Random) -> tuple[list, list]:
    """One-point crossover for indexable sequences."""
    n = min(len(a), len(b))
    if n < 2:
        return list(a), list(b)
    c = rng.randrange(1, n)  # cut in [1, n-1]
    return list(a[:c]) + list(b[c:]), list(b[:c]) + list(a[c:])