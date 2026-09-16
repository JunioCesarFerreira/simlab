"""NSGA-III strategy with environmental selection delegated to pymoo.

Inherits all MongoDB / simulation / genome-cache infrastructure from
NSGA3LoopStrategy and overrides only _select_next_parents(), using:
  - pymoo's ReferenceDirectionSurvival for niching (from NSGA3 algorithm)
  - das-dennis reference directions (same partition count as native impl)

Article reference: Table 3 — nsga3_pymoo_func (HV/GD/IGD/Coverage on DTLZ2).

pymoo is an optional dependency. All imports are deferred so that this module
can be imported safely in environments without pymoo installed; only
instantiation will fail with a clear ImportError.
"""

import logging

import numpy as np

from pylib.db.models.generation import Generation

from .nsga3 import NSGA3LoopStrategy
from .library_rng import derive_generator

logger = logging.getLogger(__name__)
_NORMALIZATION_FIELDS = ("ideal_point", "worst_point", "nadir_point", "extreme_points")


class NSGA3PymooStrategy(NSGA3LoopStrategy):
    """NSGA-III using pymoo's ReferenceDirectionSurvival for environmental selection.

    Equivalent to NSGA3LoopStrategy for all SimLab integration concerns
    (Change Streams, MongoDB persistence, Genome Cache, ProblemAdapter
    crossover/mutation). Only the environmental-selection step
    (_select_next_parents) is replaced by pymoo's reference-direction survival.

    Requires: pymoo >= 0.6.0.
    """

    def __init__(self, experiment: dict, mongo) -> None:
        super().__init__(experiment, mongo)
        # Deferred imports: pymoo is optional.
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
        from pymoo.core.problem import Problem

        n_obj = len(self._objective_keys)
        # das-dennis reference directions with same granularity as native impl.
        ref_dirs = get_reference_directions(
            "das-dennis", n_obj, n_partitions=self._divisions
        )
        self._pymoo_survival = ReferenceDirectionSurvival(ref_dirs)
        # Minimal dummy problem — ReferenceDirectionSurvival only needs n_obj.
        # Cached here to avoid per-generation object allocation.
        self._pymoo_problem = Problem(n_var=1, n_obj=n_obj)

    def _dump_selection_state(self) -> dict:
        """Checkpoint the hyperplane history alongside the RNG at enqueue.

        ReferenceDirectionSurvival retains ideal/worst and extreme points from
        earlier selections. Recomputing them from the current population loses
        information and can change niche assignments even with identical RNG.
        Before the first selection, None represents the initial normalization.
        """
        norm = self._pymoo_survival.norm
        return {
            "backend": "nsga3_pymoo",
            "version": 1,
            "normalization": None if norm.extreme_points is None else {
                name: getattr(norm, name).tolist() for name in _NORMALIZATION_FIELDS
            },
        }

    def _restore_selection_state(self, generation: Generation) -> None:
        from pymoo.algorithms.moo.nsga3 import HyperplaneNormalization

        snapshot = generation.get("selection_state")
        n_obj = len(self._objective_keys)
        norm = HyperplaneNormalization(n_obj)
        if snapshot is None:
            # Generations 0 and 1 precede environmental selection, so their
            # initial normalization is recoverable even from a legacy record.
            if int(generation["index"]) > 1:
                logger.warning(
                    "[NSGA-III/pymoo] Generation %s has no normalization checkpoint; "
                    "resuming with fresh normalization. The trajectory may differ "
                    "from the uninterrupted run.", generation["index"],
                )
        else:
            try:
                if snapshot["backend"] != "nsga3_pymoo" or snapshot["version"] != 1:
                    raise ValueError("unsupported backend or checkpoint version")
                values = snapshot["normalization"]
                if values is None and int(generation["index"]) > 1:
                    raise ValueError("missing normalization after environmental selection")
                if values is not None:
                    for name in _NORMALIZATION_FIELDS:
                        value = np.asarray(values[name], dtype=float)
                        shape = (n_obj, n_obj) if name == "extreme_points" else (n_obj,)
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError(f"invalid {name}")
                        setattr(norm, name, value.copy())
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("Cannot resume: invalid pymoo normalization checkpoint") from exc
        self._pymoo_survival.norm = norm

    # ------------------------------------------------------------------
    # Override: environmental selection via pymoo
    # ------------------------------------------------------------------

    def _select_next_parents(
        self,
        R_population: list,
        R_objectives: "list[list[float]]",
    ) -> "list | None":
        """Select self._pop_size parents using pymoo's ReferenceDirectionSurvival."""
        from pymoo.core.population import Population  # deferred: pymoo is optional

        F = np.array(R_objectives, dtype=float)

        # Build a pymoo Population from the objective matrix.
        pop = Population.new(F=F)
        # Tag each individual with its original index for the round-trip mapping.
        for i, ind in enumerate(pop):
            ind.set("simlab_idx", i)

        # pymoo NSGA-III environmental selection (NDS + ref-dir niching).
        survived = self._pymoo_survival.do(
            self._pymoo_problem,
            pop,
            n_survive=self._pop_size,
            random_state=derive_generator(self._ga_rng),
        )

        return [R_population[ind.get("simlab_idx")] for ind in survived]
