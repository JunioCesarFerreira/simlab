import math
import random

def poly_mut(x: float, 
    rng: random.Random, 
    eta: float,
    bound: tuple[float, float]
    ) -> float:
    if eta <= 0:
        raise ValueError("eta must be > 0 for Polynomial Mutation.")
    yl, yu = bound

    # Safety checks
    if not math.isfinite(yl) or not math.isfinite(yu) or yu <= yl:
        # Fallback: very small Gaussian perturbation
        return x + rng.gauss(0.0, 1e-2)

    # Normalize distances
    delta1 = (x - yl) / (yu - yl)
    delta2 = (yu - x) / (yu - yl)

    r = rng.random()
    mut_pow = 1.0 / (eta + 1.0)

    if r <= 0.5:
        xy = 1.0 - delta1
        val = 2.0 * r + (1.0 - 2.0 * r) * pow(xy, eta + 1.0)
        deltaq = pow(val, mut_pow) - 1.0
    else:
        xy = 1.0 - delta2
        val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * pow(xy, eta + 1.0)
        deltaq = 1.0 - pow(val, mut_pow)

    x_new = x + deltaq * (yu - yl)

    # Projection to bounds
    return min(max(x_new, yl), yu)