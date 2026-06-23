"""NSGA-III strategy with environmental selection delegated to DEAP's selNSGA3.

Inherits all MongoDB / simulation / genome-cache infrastructure from
NSGA3LoopStrategy and overrides only _select_next_parents(), replacing the
native fast_nondominated_sort + niching_selection pair with DEAP's
tools.sortNondominated + tools.selNSGA3.

Article reference: Table 3 — nsga3_deap_func (HV/GD/IGD/Coverage on DTLZ2).

DEAP is an optional dependency. All imports are deferred so that this module
can be imported safely in environments without DEAP installed; only
instantiation will fail with a clear ImportError.
"""

from .nsga3 import NSGA3LoopStrategy


def _ensure_deap_classes(n_obj: int) -> tuple[str, str]:
    """Register DEAP creator classes for *n_obj* objectives if not already done.

    Class names encode n_obj so that different objective counts can coexist
    in the same process (e.g. two sequential experiments with 2 and 3 objectives
    respectively). Returns (fitness_class_name, individual_class_name).
    """
    from deap import creator, base  # deferred: DEAP is optional

    fitness_name = f"FitnessSimlabDeap{n_obj}"
    ind_name = f"IndSimlabDeap{n_obj}"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0,) * n_obj)
        creator.create(ind_name, list, fitness=getattr(creator, fitness_name))
    return fitness_name, ind_name


class NSGA3DeapStrategy(NSGA3LoopStrategy):
    """NSGA-III using DEAP's selNSGA3 for environmental selection.

    Equivalent to NSGA3LoopStrategy for all SimLab integration concerns
    (Change Streams, MongoDB persistence, Genome Cache, ProblemAdapter
    crossover/mutation). Only the environmental-selection step
    (_select_next_parents) is replaced by DEAP's reference-point niching.

    Requires: deap >= 1.3.0.
    """

    def __init__(self, experiment: dict, mongo) -> None:
        super().__init__(experiment, mongo)
        from deap import tools  # deferred: DEAP is optional

        n_obj = len(self._objective_keys)
        _, self._deap_ind_class = _ensure_deap_classes(n_obj)
        # Pre-compute reference points once; same granularity as native impl.
        self._deap_ref_points = tools.uniform_reference_points(
            nobj=n_obj, p=self._divisions
        )

    # ------------------------------------------------------------------
    # Override: environmental selection via DEAP
    # ------------------------------------------------------------------

    def _select_next_parents(
        self,
        R_population: list,
        R_objectives: "list[list[float]]",
    ) -> "list | None":
        """Select self._pop_size parents using DEAP's NSGA-III selNSGA3.

        Objectives are already in minimization space (multiplied by goal sign
        in _extract_objectives_to_minimization). DEAP's selNSGA3 treats
        fitness.values as "higher is better", so we negate via weights=(-1,…)
        to align with minimization — the double negation keeps correct order.
        """
        from deap import creator, tools  # deferred: DEAP is optional

        IndClass = getattr(creator, self._deap_ind_class)

        # Wrap each objective vector as a DEAP individual.
        # _orig_idx tracks position in R_population for the round-trip mapping.
        deap_inds = []
        for i, obj in enumerate(R_objectives):
            ind = IndClass(list(obj))
            ind.fitness.values = tuple(float(v) for v in obj)
            ind._orig_idx = i
            deap_inds.append(ind)

        # DEAP's selNSGA3: non-dominated sorting + niching with ref points.
        selected = tools.selNSGA3(deap_inds, self._pop_size, self._deap_ref_points)

        return [R_population[ind._orig_idx] for ind in selected]
