from typing import Sequence, TypeVar, Optional
import random

T = TypeVar("T")  

def gaussian_mutation(
    x: Sequence[float],
    rng: random.Random,
    sigma: float = 0.1,
    per_gene_prob: float = 0.1,
    bounds: Optional[Sequence[tuple[float, float]]] = None,
) -> list[float]:
    """Add Gaussian noise to each gene w.p. per_gene_prob; clip to bounds if provided."""
    y = list(x)
    for i in range(len(y)):
        if rng.random() < per_gene_prob:
            y[i] = y[i] + rng.gauss(0.0, sigma)
            if bounds is not None:
                lo, hi = bounds[i]
                y[i] = max(lo, min(hi, y[i]))
    return y