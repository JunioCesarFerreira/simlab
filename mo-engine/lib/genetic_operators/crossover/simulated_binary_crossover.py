import math
import random


def _spread_factor(beta: float, rand: float, eta: float) -> float:
    """SBX spread factor betaq for one child, from its own distance to the bound.

    *beta* encodes how much room that child has before it would leave the box;
    the larger it is, the closer the distribution gets to unbounded SBX.
    """
    alpha = 2.0 - pow(beta, -(eta + 1.0)) if beta > 0 else 1.0
    if rand <= 1.0 / alpha:
        return pow(rand * alpha, 1.0 / (eta + 1.0))
    return pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0))


def sbx(p1: float, p2: float, 
        rng: random.Random,
        eta: float, 
        bounds: tuple[float,float]
        ) -> tuple[float, float]:    
    if eta <= 0:
        raise ValueError("eta must be > 0 for SBX.")
    
    x1, x2 = p1, p2

    # If genes are (almost) identical, nothing to mix
    if abs(x1 - x2) < 1e-14:
        c1, c2 = x1, x2
        return c1, c2

    # Optional bounds
    if bounds is not None:
        xl, xu = bounds
    else:
        xl, xu = -math.inf, math.inf

    # Ensure y1 <= y2 for formula
    if x1 > x2:
        y1, y2 = x2, x1
        swap = True
    else:
        y1, y2 = x1, x2
        swap = False

    # One draw, two spread factors. Each child is pushed away from the parents
    # in the opposite direction, so each needs the distance to the bound it is
    # heading towards: the lower child to xl, the upper child to xu. Reusing the
    # lower child's factor for both — as this did — biases the upper child by
    # however asymmetric the parents sit inside the box, and clipping afterwards
    # does not restore the distribution, it only stacks probability mass on the
    # bound. Matches DEAP's cxSimulatedBinaryBounded.
    rand = rng.random()
    beta_low = 1.0 + 2.0 * (y1 - xl) / (y2 - y1) if math.isfinite(xl) else 1.0
    beta_high = 1.0 + 2.0 * (xu - y2) / (y2 - y1) if math.isfinite(xu) else 1.0

    child1 = 0.5 * ((y1 + y2) - _spread_factor(beta_low, rand, eta) * (y2 - y1))
    child2 = 0.5 * ((y1 + y2) + _spread_factor(beta_high, rand, eta) * (y2 - y1))

    # Clip to bounds if provided
    child1 = min(max(child1, xl), xu)
    child2 = min(max(child2, xl), xu)

    # Randomly swap assignment to increase diversity
    if rng.random() < 0.5:
        child1, child2 = child2, child1

    # Restore original order of parents
    if swap:
        c1, c2 = child2, child1
    else:
        c1, c2 = child1, child2

    return c1, c2