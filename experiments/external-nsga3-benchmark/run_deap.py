"""
NSGA-III on DTLZ2 via DEAP (``deap.tools.selNSGA3``).

This implementation is provided strictly for the benchmark comparison in
Table 3 of the SimLab paper — DEAP is NOT a runtime dependency of the
platform.

References:
  * Fortin et al., "DEAP: Evolutionary Algorithms Made Easy", JMLR 2012.
  * Deb & Jain, NSGA-III, IEEE TEVC 2014.
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from deap import base, creator, tools
from deap.tools import selNSGA3, uniform_reference_points

from dtlz2 import evaluate as dtlz2_evaluate, n_variables as dtlz2_n_variables


def _build_deap_toolbox(M: int, n_var: int, eta_cx: float, eta_mt: float,
                       prob_mt: float):
    # Defensive: avoid re-creating the same class on repeated calls
    if not hasattr(creator, "FitnessMinMulti"):
        creator.create("FitnessMinMulti", base.Fitness, weights=(-1.0,) * M)
    else:  # remove previous FitnessMin if dimensions differ
        if len(creator.FitnessMinMulti.weights) != M:
            del creator.FitnessMinMulti
            creator.create("FitnessMinMulti", base.Fitness, weights=(-1.0,) * M)

    if hasattr(creator, "Individual"):
        del creator.Individual
    creator.create("Individual", list, fitness=creator.FitnessMinMulti)

    tb = base.Toolbox()
    tb.register("attr_float", random.random)
    tb.register("individual", tools.initRepeat, creator.Individual,
                tb.attr_float, n=n_var)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: tuple(dtlz2_evaluate(list(ind), M)))
    tb.register("mate", tools.cxSimulatedBinaryBounded,
                low=0.0, up=1.0, eta=eta_cx)
    tb.register("mutate", tools.mutPolynomialBounded,
                low=0.0, up=1.0, eta=eta_mt, indpb=prob_mt)
    return tb


def run_deap(
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
    random.seed(seed)
    np.random.seed(seed)

    n_var = dtlz2_n_variables(M, k)
    if prob_mt is None:
        prob_mt = 1.0 / max(1, n_var)

    tb = _build_deap_toolbox(M, n_var, eta_cx, eta_mt, prob_mt)
    ref_points = uniform_reference_points(nobj=M, p=divisions)

    pop = tb.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = tb.evaluate(ind)

    for _gen in range(n_generations):
        offspring = [tb.clone(ind) for ind in pop]
        # Variation: in-place mating + mutation
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < prob_cx:
                tb.mate(offspring[i], offspring[i + 1])
            tb.mutate(offspring[i])
            tb.mutate(offspring[i + 1])
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        # Evaluate invalid fitnesses
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = tb.evaluate(ind)
        # NSGA-III environmental selection
        pop = selNSGA3(pop + offspring, pop_size, ref_points)

    return np.asarray([list(ind.fitness.values) for ind in pop], dtype=float)


if __name__ == "__main__":
    front = run_deap(M=3, k=2, pop_size=92, n_generations=200, seed=42)
    print(f"DEAP NSGA-III: final population {front.shape}")
