"""
NSGA-III on DTLZ2 via pymoo (``pymoo.algorithms.moo.nsga3``).

This implementation is provided strictly for the benchmark comparison in
Table 3 of the SimLab paper — it is independent of the platform's
``nsga3_pymoo`` strategy, which uses only pymoo's environmental selection.

References:
  * Blank & Deb, "pymoo: Multi-objective Optimization in Python", IEEE
    Access 2020.
  * Deb & Jain, NSGA-III, IEEE TEVC 2014.
"""
from __future__ import annotations

import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.ref_dirs import get_reference_directions

from dtlz2 import evaluate as dtlz2_evaluate, n_variables as dtlz2_n_variables


class DTLZ2Problem(ElementwiseProblem):
    """Wraps the local DTLZ2 implementation so all runners share the exact
    same evaluation (matching ``pylib/benchmarks.py``)."""

    def __init__(self, n_var: int, n_obj: int):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self._n_obj = n_obj

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.asarray(dtlz2_evaluate(list(x), self._n_obj), dtype=float)


def _by_rank(pop, P, **kwargs):
    """Binary tournament on non-domination rank, the first drawn on a tie:
    the mating selection of run_native and run_deap. pymoo's default for
    NSGA3 is a random pick on an unconstrained problem."""
    rank = pop.get("rank")
    a, b = P[:, 0], P[:, 1]
    return np.where(rank[b] < rank[a], b, a)


def build_algorithm(
    M: int,
    pop_size: int,
    divisions: int,
    eta_cx: float,
    eta_mt: float,
    prob_cx: float,
    prob_mt: float,
) -> NSGA3:
    """pymoo's NSGA-III with the variation the other two runners use."""
    ref_dirs = get_reference_directions("das-dennis", M, n_partitions=divisions)
    return NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        selection=TournamentSelection(func_comp=_by_rank),
        # prob: per pair; prob_var: per variable, as the 0.5 of run_native.
        crossover=SBX(eta=eta_cx, prob=prob_cx, prob_var=0.5),
        # prob is per individual: the 1/n per variable is prob_var.
        mutation=PM(eta=eta_mt, prob=1.0, prob_var=prob_mt),
        # The other two keep duplicates.
        eliminate_duplicates=False,
    )


def run_pymoo(
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
    """The final population's objective matrix, of shape (pop_size, M)."""
    n_var = dtlz2_n_variables(M, k)
    if prob_mt is None:
        prob_mt = 1.0 / max(1, n_var)

    algo = build_algorithm(M, pop_size, divisions, eta_cx, eta_mt, prob_cx, prob_mt)
    res = minimize(
        DTLZ2Problem(n_var, M),
        algo,
        ("n_gen", n_generations),
        seed=seed,
        verbose=False,
    )
    # Not res.F: that is the part of the first front nearest the reference
    # directions, taken before survival, so it can hold more points than the
    # population. The other runners return the population.
    return np.asarray(res.pop.get("F"), dtype=float)


if __name__ == "__main__":
    front = run_pymoo(M=3, k=2, pop_size=92, n_generations=200, seed=42)
    print(f"pymoo NSGA-III: final population {front.shape}")
