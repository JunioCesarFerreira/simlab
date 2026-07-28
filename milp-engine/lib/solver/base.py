"""
Solver-agnostic MILP intermediate representation (IR) and backend interface.

Models (lib/models/*) build a MilpIR; backends (gurobi_backend, highs_backend)
translate the IR into their native API and return a unified SolveResult.
Keeping the IR independent of any solver allows the same model to run under
Gurobi (licensed) or HiGHS (open source) and makes model construction testable
without a solver installed.
"""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional

VarType = Literal["B", "C"]  # binary | continuous

Sense = Literal["<=", ">=", "=="]

# Unified solve statuses
STATUS_OPTIMAL = "OPTIMAL"            # proven optimal (or within mip_gap)
STATUS_TIME_LIMIT = "TIME_LIMIT"      # time limit hit, incumbent available
STATUS_NO_SOLUTION = "NO_SOLUTION"    # time limit hit, no incumbent
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_ERROR = "ERROR"

# Statuses that carry a usable variable assignment
SOLUTION_STATUSES = (STATUS_OPTIMAL, STATUS_TIME_LIMIT)


@dataclass(frozen=True, slots=True)
class Constraint:
    coeffs: tuple[tuple[int, float], ...]  # (var_index, coefficient)
    sense: Sense
    rhs: float


@dataclass(slots=True)
class MilpIR:
    var_names: list[str]
    var_types: list[VarType]
    var_lb: list[float]
    var_ub: list[float]
    objective: list[tuple[int, float]]  # sparse coefficients, always MINIMIZE
    constraints: list[Constraint]

    @property
    def n_vars(self) -> int:
        return len(self.var_names)

    @property
    def n_constraints(self) -> int:
        return len(self.constraints)


class MilpBuilder:
    """Incremental construction helper for MilpIR."""

    def __init__(self) -> None:
        self._names: list[str] = []
        self._types: list[VarType] = []
        self._lb: list[float] = []
        self._ub: list[float] = []
        self._obj: list[tuple[int, float]] = []
        self._constraints: list[Constraint] = []

    def add_var(
        self,
        name: str,
        vtype: VarType = "C",
        lb: float = 0.0,
        ub: float = math.inf,
    ) -> int:
        if vtype == "B":
            lb, ub = 0.0, 1.0
        self._names.append(name)
        self._types.append(vtype)
        self._lb.append(lb)
        self._ub.append(ub)
        return len(self._names) - 1

    def add_constr(
        self, coeffs: list[tuple[int, float]], sense: Sense, rhs: float
    ) -> None:
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"Invalid constraint sense: {sense!r}")
        self._constraints.append(Constraint(tuple(coeffs), sense, rhs))

    def add_objective_term(self, var_index: int, coefficient: float) -> None:
        self._obj.append((var_index, coefficient))

    def build(self) -> MilpIR:
        # Merge repeated objective indices so backends can assume one entry per var
        merged: dict[int, float] = {}
        for idx, coef in self._obj:
            merged[idx] = merged.get(idx, 0.0) + coef
        return MilpIR(
            var_names=self._names,
            var_types=self._types,
            var_lb=self._lb,
            var_ub=self._ub,
            objective=sorted(merged.items()),
            constraints=self._constraints,
        )


@dataclass(slots=True)
class SolveResult:
    status: str
    values: Optional[list[float]] = None
    obj_value: Optional[float] = None
    mip_gap: Optional[float] = None
    runtime_s: float = 0.0
    message: str = ""

    @property
    def has_solution(self) -> bool:
        return self.status in SOLUTION_STATUSES and self.values is not None


class SolverBackend(ABC):
    """A MILP solver capable of solving a MilpIR."""

    name: str = "abstract"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """True when the backend can actually solve (package importable and,
        for Gurobi, a usable license — full or size-limited trial)."""

    @abstractmethod
    def solve(
        self,
        ir: MilpIR,
        time_limit_s: Optional[float] = None,
        mip_gap: Optional[float] = None,
    ) -> SolveResult:
        """Solve the IR (minimization). Never raises for solver-side outcomes;
        maps them to SolveResult.status instead."""
