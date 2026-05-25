"""
NSGA-III on DTLZ2 via pymoo (``pymoo.algorithms.moo.nsga3``).

This implementation is provided strictly for the benchmark comparison in
Table 3 of the SimLab paper — pymoo is NOT a runtime dependency of the
platform.

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
from pymoo.util.ref_dirs import get_reference_directions

from dtlz2 import evaluate as dtlz2_evaluate, n_variables as dtlz2_n_variables


class DTLZ2Problem(ElementwiseProblem):
    """Wraps the local DTLZ2 implementation so all runners share the exact
    same evaluation (matching ``master-node/lib/synthetic_data.py``)."""

    def __init__(self, n_var: int, n_obj: int):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self._n_obj = n_obj

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.asarray(dtlz2_evaluate(list(x), self._n_obj), dtype=float)


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
    n_var = dtlz2_n_variables(M, k)
    if prob_mt is None:
        prob_mt = 1.0 / max(1, n_var)

    ref_dirs = get_reference_directions("das-dennis", M, n_partitions=divisions)
    algo = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        crossover=SBX(eta=eta_cx, prob=prob_cx),
        mutation=PM(eta=eta_mt, prob=prob_mt),
    )
    res = minimize(
        DTLZ2Problem(n_var, M),
        algo,
        ("n_gen", n_generations),
        seed=seed,
        verbose=False,
    )
    return np.asarray(res.F, dtype=float)


if __name__ == "__main__":
    front = run_pymoo(M=3, k=2, pop_size=92, n_generations=200, seed=42)
    print(f"pymoo NSGA-III: final population {front.shape}")
