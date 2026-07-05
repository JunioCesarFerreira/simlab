"""NSGA-II strategy with environmental selection delegated to DEAP's selNSGA2.

Inherits all MongoDB / simulation / genome-cache infrastructure from
NSGA2LoopStrategy and overrides only _select_next_parents(), replacing the
native fast_nondominated_sort + select_next_population pair with DEAP's
tools.selNSGA2 (non-dominated sorting + crowding distance tie-breaking).

DEAP is an optional dependency. All imports are deferred so that this module
can be imported safely in environments without DEAP installed; only
instantiation will fail with a clear ImportError.
"""

from .nsga2 import NSGA2LoopStrategy


def _ensure_deap_classes_nsga2(n_obj: int) -> tuple[str, str]:
    """Register DEAP creator classes for *n_obj* objectives if not already done.

    Class names encode the algorithm (nsga2) and n_obj to coexist with the
    nsga3_deap classes in the same process. Returns (fitness_name, ind_name).
    """
    from deap import creator, base  # deferred: DEAP is optional

    fitness_name = f"FitnessSimlabNsga2Deap{n_obj}"
    ind_name = f"IndSimlabNsga2Deap{n_obj}"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,) * n_obj)
        creator.create(ind_name, list, fitness=getattr(creator, fitness_name))
    return fitness_name, ind_name


class NSGA2DeapStrategy(NSGA2LoopStrategy):
    """NSGA-II using DEAP's selNSGA2 for environmental selection.

    Equivalent to NSGA2LoopStrategy for all SimLab integration concerns
    (Change Streams, MongoDB persistence, Genome Cache, ProblemAdapter
    crossover/mutation). Only the environmental-selection step
    (_select_next_parents) is replaced by DEAP's non-dominated sorting +
    crowding distance.

    Requires: deap >= 1.3.0.
    """

    def __init__(self, experiment: dict, mongo) -> None:
        super().__init__(experiment, mongo)
        n_obj = len(self._objective_keys)
        _, self._deap_ind_class = _ensure_deap_classes_nsga2(n_obj)

    # ------------------------------------------------------------------
    # Override: environmental selection via DEAP
    # ------------------------------------------------------------------

    def _select_next_parents(
        self,
        R_population: list,
        R_objectives: "list[list[float]]",
    ) -> "list | None":
        """Select self._pop_size parents using DEAP's selNSGA2.

        Objectives are already in minimization space. DEAP's selNSGA2 treats
        fitness.values as "higher is better", so we negate via weights=(-1,…)
        to align with minimization — the double negation keeps correct order.
        """
        from deap import creator, tools  # deferred: DEAP is optional

        IndClass = getattr(creator, self._deap_ind_class)

        deap_inds = []
        for i, obj in enumerate(R_objectives):
            ind = IndClass(list(obj))
            ind.fitness.values = tuple(float(v) for v in obj)
            ind._orig_idx = i
            deap_inds.append(ind)

        selected = tools.selNSGA2(deap_inds, self._pop_size)

        return [R_population[ind._orig_idx] for ind in selected]
