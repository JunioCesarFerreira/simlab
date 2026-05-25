"""
NSGA-III on DTLZ2 using the SimLab native implementation
(``mo-engine/lib/nsga/``).  No MongoDB, no master-node — just the same
non-dominated sorting and niching code the platform runs.

Returns the final population's objective matrix.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# Allow `import lib.nsga.*` and `import lib.genetic_operators.*` from mo-engine.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mo-engine"))

from lib.nsga.fast_nondominated_sort import fast_nondominated_sort   # noqa: E402
from lib.nsga.niching_selection import (                              # noqa: E402
    environmental_selection,
    generate_reference_points,
)
from lib.genetic_operators.crossover.simulated_binary_crossover import sbx  # noqa: E402
from lib.genetic_operators.mutation.polynomial_mutation import poly_mut     # noqa: E402

from dtlz2 import evaluate as dtlz2_evaluate, n_variables as dtlz2_n_variables


def _sbx_vector(
    p1: list[float],
    p2: list[float],
    rng: random.Random,
    eta: float,
    bounds: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    """Apply gene-wise SBX with crossover probability 0.5 per gene."""
    lo, hi = bounds
    c1, c2 = list(p1), list(p2)
    for i in range(len(p1)):
        if rng.random() < 0.5:
            c1[i], c2[i] = sbx(p1[i], p2[i], rng=rng, eta=eta, bounds=(lo[i], hi[i]))
    return c1, c2


def _poly_mut_vector(
    x: list[float],
    rng: random.Random,
    eta: float,
    bounds: tuple[list[float], list[float]],
    per_gene_prob: float,
) -> list[float]:
    lo, hi = bounds
    out = list(x)
    for i in range(len(x)):
        if rng.random() < per_gene_prob:
            out[i] = poly_mut(out[i], rng=rng, eta=eta, bound=(lo[i], hi[i]))
    return out


def _tournament(pop_indices: list[int], ranks: dict[int, int], rng: random.Random) -> int:
    a, b = rng.sample(pop_indices, 2)
    return a if ranks[a] <= ranks[b] else b


def _compute_ranks(fronts: list[list[int]]) -> dict[int, int]:
    ranks: dict[int, int] = {}
    for r, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = r
    return ranks


def run_native(
    M: int,
    k: int,
    pop_size: int = 92,
    n_generations: int = 400,
    divisions: int = 12,
    eta_cx: float = 20.0,
    eta_mt: float = 25.0,
    prob_cx: float = 0.9,
    prob_mt: float | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Run the native NSGA-III on DTLZ2(M, k) and return the final population's
    objective matrix of shape (pop_size, M).
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n_var = dtlz2_n_variables(M, k)
    if prob_mt is None:
        prob_mt = 1.0 / max(1, n_var)

    bounds = ([0.0] * n_var, [1.0] * n_var)
    ref_points = generate_reference_points(M, divisions)

    population: list[list[float]] = [
        np_rng.random(n_var).tolist() for _ in range(pop_size)
    ]
    objectives = [dtlz2_evaluate(ind, M) for ind in population]

    for _gen in range(n_generations):
        fronts = fast_nondominated_sort(objectives)
        ranks = _compute_ranks(fronts)
        pop_idx_list = list(range(len(population)))

        offspring: list[list[float]] = []
        while len(offspring) < pop_size:
            i = _tournament(pop_idx_list, ranks, rng)
            j = _tournament(pop_idx_list, ranks, rng)
            p1, p2 = population[i], population[j]
            if rng.random() < prob_cx:
                c1, c2 = _sbx_vector(p1, p2, rng, eta_cx, bounds)
            else:
                c1, c2 = list(p1), list(p2)
            c1 = _poly_mut_vector(c1, rng, eta_mt, bounds, prob_mt)
            c2 = _poly_mut_vector(c2, rng, eta_mt, bounds, prob_mt)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        combined = population + offspring
        combined_obj = objectives + [dtlz2_evaluate(ind, M) for ind in offspring]
        combined_fronts = fast_nondominated_sort(combined_obj)
        population = environmental_selection(
            combined, combined_obj, combined_fronts, ref_points, pop_size, rng,
        )
        objectives = [dtlz2_evaluate(ind, M) for ind in population]

    return np.asarray(objectives, dtype=float)


if __name__ == "__main__":
    front = run_native(M=3, k=2, pop_size=92, n_generations=200, seed=42)
    print(f"native NSGA-III: final population {front.shape}")
