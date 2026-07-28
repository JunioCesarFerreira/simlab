"""
MILP model interface.

A MilpModel turns (problem draft, parameter values) into a solver-agnostic
MilpIR plus the mapping needed to decode a solution back into a SimLab
chromosome mask.

GENOTYPE ORDER CONVENTION: mask[i] refers to problem["candidates"][i] — the
same convention as ChromosomeP2/P3 in the mo-engine. Models must build
``y_indices`` in candidates-list order, never sorted by any node name.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """Schema of one model parameter, consumed by the sweep builder GUI."""
    name: str
    description: str
    default: float
    sweepable: bool = True


@dataclass(slots=True)
class BuiltModel:
    """A model instance ready to solve."""
    ir: "MilpIR"  # noqa: F821 (import kept light; see lib.solver.base)
    # Variable index of the installation decision y_j for each candidate,
    # in problem["candidates"] order.
    y_indices: list[int]
    meta: dict[str, Any] = field(default_factory=dict)

    def decode_mask(self, values: list[float]) -> list[int]:
        return [1 if values[i] > 0.5 else 0 for i in self.y_indices]


class MilpModel(ABC):
    key: ClassVar[str]
    problem_key: ClassVar[str]      # SimLab problem name this model applies to
    title: ClassVar[str]
    description: ClassVar[str] = ""
    parameters: ClassVar[tuple[ParamSpec, ...]] = ()
    solver_defaults: ClassVar[dict[str, float]] = {
        "time_limit_s": 300.0,
        "mip_gap": 0.01,
    }

    @abstractmethod
    def build(self, problem: Mapping[str, Any], params: Mapping[str, float]) -> BuiltModel:
        """Build the MILP for the given problem instance and parameter values.

        ``params`` may omit any parameter; defaults from ``parameters`` apply.
        Raises ValueError on malformed problem documents."""

    def resolve_params(self, params: Mapping[str, float]) -> dict[str, float]:
        """Merge user params over defaults, rejecting unknown names."""
        known = {p.name: p.default for p in self.parameters}
        unknown = set(params) - set(known)
        if unknown:
            raise ValueError(
                f"Unknown parameters for model '{self.key}': {sorted(unknown)}"
            )
        known.update({k: float(v) for k, v in params.items()})
        return known
