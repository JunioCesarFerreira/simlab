from typing import Sequence, Optional, Callable
import math
import random

# Type aliases reused from your GA
RealVec = Sequence[float]
Bounds = Sequence[tuple[float, float]]
CrossoverFn = Callable[[RealVec, RealVec, random.Random], tuple[list[float], list[float]]]

# ---------- SBX: Simulated Binary Crossover ----------

def make_sbx_crossover(
    *,
    eta: float = 15.0,
    bounds: Optional[Bounds] = None,
) -> CrossoverFn:
    """
    Factory for Simulated Binary Crossover (SBX), Deb & Agrawal (1995).

    Parameters
    ----------
    eta : float
        Distribution index (η_c). Larger -> children closer to parents.
        Typical values: 10 ~ 30. Default 15.0.
    bounds : Optional[Bounds]
        Per-gene bounds [(lo, hi), ...] used for constructing children and clipping.
        If None, the operator still works but cannot respect domain limits.

    Returns
    -------
    CrossoverFn
        A function (p1, p2, rng) -> (c1, c2), suitable for your GA.
    """
    if eta <= 0:
        raise ValueError("eta must be > 0 for SBX.")

    def sbx(p1: RealVec, p2: RealVec, rng: random.Random) -> tuple[list[float], list[float]]:
        n = min(len(p1), len(p2))
        if n == 0:
            return list(p1), list(p2)

        c1 = list(p1)
        c2 = list(p2)

        for i in range(n):
            x1, x2 = p1[i], p2[i]

            # If genes are (almost) identical, nothing to mix
            if abs(x1 - x2) < 1e-14:
                c1[i], c2[i] = x1, x2
                continue

            # Optional bounds
            if bounds is not None:
                xl, xu = bounds[i]
            else:
                xl, xu = -math.inf, math.inf

            # Ensure y1 <= y2 for formula
            if x1 > x2:
                y1, y2 = x2, x1
                swap = True
            else:
                y1, y2 = x1, x2
                swap = False

            rand = rng.random()
            beta = 1.0 + 2.0 * (y1 - xl) / (y2 - y1) if math.isfinite(xl) else 1.0
            alpha = 2.0 - pow(beta, -(eta + 1.0)) if beta > 0 else 1.0

            if rand <= 1.0 / alpha:
                betaq = pow(rand * alpha, 1.0 / (eta + 1.0))
            else:
                betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0))

            child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
            child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            # Clip to bounds if provided
            child1 = min(max(child1, xl), xu)
            child2 = min(max(child2, xl), xu)

            # Randomly swap assignment to increase diversity
            if rng.random() < 0.5:
                child1, child2 = child2, child1

            # Restore original order of parents
            if swap:
                c1[i], c2[i] = child2, child1
            else:
                c1[i], c2[i] = child1, child2

        return c1, c2

    return sbx
