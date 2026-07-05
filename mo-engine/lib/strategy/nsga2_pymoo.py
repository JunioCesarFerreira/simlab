"""NSGA-II strategy with environmental selection delegated to pymoo.

Inherits all MongoDB / simulation / genome-cache infrastructure from
NSGA2LoopStrategy and overrides only _select_next_parents(), using:
  - pymoo's RankAndCrowdingSurvival (from NSGA-II algorithm)

pymoo is an optional dependency. All imports are deferred so that this module
can be imported safely in environments without pymoo installed; only
instantiation will fail with a clear ImportError.
"""

import numpy as np

from .nsga2 import NSGA2LoopStrategy


class NSGA2PymooStrategy(NSGA2LoopStrategy):
    """NSGA-II using pymoo's RankAndCrowdingSurvival for environmental selection.

    Equivalent to NSGA2LoopStrategy for all SimLab integration concerns
    (Change Streams, MongoDB persistence, Genome Cache, ProblemAdapter
    crossover/mutation). Only the environmental-selection step
    (_select_next_parents) is replaced by pymoo's rank-and-crowding survival.

    Requires: pymoo >= 0.6.0.
    """

    def __init__(self, experiment: dict, mongo) -> None:
        super().__init__(experiment, mongo)
        # Deferred imports: pymoo is optional.
        from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
        from pymoo.core.problem import Problem

        n_obj = len(self._objective_keys)
        self._pymoo_survival = RankAndCrowding()
        # Minimal dummy problem — RankAndCrowdingSurvival only needs n_obj.
        # Cached here to avoid per-generation object allocation.
        self._pymoo_problem = Problem(n_var=1, n_obj=n_obj)

    # ------------------------------------------------------------------
    # Override: environmental selection via pymoo
    # ------------------------------------------------------------------

    def _select_next_parents(
        self,
        R_population: list,
        R_objectives: "list[list[float]]",
    ) -> "list | None":
        """Select self._pop_size parents using pymoo's RankAndCrowdingSurvival."""
        from pymoo.core.population import Population  # deferred: pymoo is optional

        F = np.array(R_objectives, dtype=float)

        # Build a pymoo Population from the objective matrix.
        pop = Population.new(F=F)
        # Tag each individual with its original index for the round-trip mapping.
        for i, ind in enumerate(pop):
            ind.set("simlab_idx", i)

        # pymoo NSGA-II environmental selection (NDS + crowding distance).
        survived = self._pymoo_survival.do(
            self._pymoo_problem, pop, n_survive=self._pop_size
        )

        return [R_population[ind.get("simlab_idx")] for ind in survived]
