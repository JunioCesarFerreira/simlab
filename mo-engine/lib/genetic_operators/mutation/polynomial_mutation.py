from typing import Sequence, Optional, Callable
import math
import random

# Type aliases reused from your GA
RealVec = Sequence[float]
Bounds = Sequence[tuple[float, float]]
MutationFn  = Callable[[RealVec, random.Random], list[float]]

# ---------- Polynomial Mutation (real-coded) ----------

def make_polynomial_mutation(
    *,
    eta: float = 20.0,
    per_gene_prob: Optional[float] = None,
    bounds: Optional[Bounds] = None,
) -> MutationFn:
    """
    Factory for Polynomial Mutation, Deb (1996).

    Parameters
    ----------
    eta : float
        Distribution index (η_m). Larger -> smaller perturbations.
        Typical values: 15 ~ 25. Default 20.0.
    per_gene_prob : Optional[float]
        Probability to mutate each gene. If None, defaults to 1/n (n = len(x)).
    bounds : Optional[Bounds]
        Per-gene bounds [(lo, hi), ...] for valid domain and scaling.
        If None, mutation still works but without clipping.

    Returns
    -------
    MutationFn
        A function (x, rng) -> y, suitable for your GA.
    """
    if eta <= 0:
        raise ValueError("eta must be > 0 for Polynomial Mutation.")

    def poly_mut(x: RealVec, rng: random.Random) -> list[float]:
        n = len(x)
        if n == 0:
            return list(x)
        p = (1.0 / n) if per_gene_prob is None else per_gene_prob
        if not (0.0 <= p <= 1.0):
            raise ValueError("per_gene_prob must be in [0, 1].")

        y = list(x)
        mut_pow = 1.0 / (eta + 1.0)

        for i in range(n):
            if rng.random() >= p:
                continue

            xi = y[i]
            if bounds is not None:
                yl, yu = bounds[i]
            else:
                yl, yu = -math.inf, math.inf

            if not math.isfinite(yl) or not math.isfinite(yu) or yu <= yl:
                # Without valid bounds, apply an unbounded small perturbation (fallback)
                # (You can customize this fallback if you prefer strict bounded domains.)
                delta = rng.gauss(0.0, 0.01)  # tiny step as a safeguard
                y[i] = xi + delta
                continue

            # Normalize distances to bounds
            delta1 = (xi - yl) / (yu - yl)
            delta2 = (yu - xi) / (yu - yl)

            r = rng.random()
            if r <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * r + (1.0 - 2.0 * r) * pow(xy, eta + 1.0)
                deltaq = pow(val, mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * pow(xy, eta + 1.0)
                deltaq = 1.0 - pow(val, mut_pow)

            xi_new = xi + deltaq * (yu - yl)
            # Clip to bounds
            y[i] = min(max(xi_new, yl), yu)

        return y

    return poly_mut