from .base import (
    Constraint,
    MilpBuilder,
    MilpIR,
    SolveResult,
    SolverBackend,
    SOLUTION_STATUSES,
    STATUS_ERROR,
    STATUS_INFEASIBLE,
    STATUS_NO_SOLUTION,
    STATUS_OPTIMAL,
    STATUS_TIME_LIMIT,
)
from .gurobi_backend import GurobiBackend
from .highs_backend import HighsBackend

BACKEND_REGISTRY: dict[str, type[SolverBackend]] = {
    GurobiBackend.name: GurobiBackend,
    HighsBackend.name: HighsBackend,
}


def resolve_backend(name: str, allow_fallback: bool = False) -> SolverBackend:
    """
    Instantiate a solver backend by name.

    With allow_fallback=True, an unavailable primary backend (e.g. Gurobi
    without a license) falls back to any other available backend instead of
    raising.
    """
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(f"Unknown solver backend '{name}'. Known: {known}")
    if cls.is_available():
        return cls()
    if allow_fallback:
        for other in BACKEND_REGISTRY.values():
            if other is not cls and other.is_available():
                return other()
    raise RuntimeError(
        f"Solver backend '{name}' is not available "
        "(package missing or license not usable)."
    )
