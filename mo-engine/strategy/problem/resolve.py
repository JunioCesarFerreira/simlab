from typing import Any, Mapping

from .registry import PROBLEM_REGISTRY
from .problem_adapter import ProblemAdapter


def resolve_problem_key(problem: Mapping[str, Any]) -> str:
    """
    Resolve the problem key from a problem document.

    Primary convention:
      - problem["name"] where name ∈ {"problem1", "problem2", "problem3", "problem4"}

    Backward-compatible aliases (if you still have older stored documents):
      - problem["problem_type"]
      - problem["type"]
      - problem["problem"]["type"]   (nested shape)
      - problem["parameters"]["problem_type"] (nested shape)

    Raises:
      ValueError if no supported field is found or if the value is empty.
    """
    # Preferred field: TypedDict HomogeneousProblem.name
    name = problem.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()

    raise ValueError(
        "Unable to resolve problem key. Expected a non-empty key in one of: problem['name']"
    )


def build_adapter(problem: Mapping[str, Any]) -> ProblemAdapter:
    """
    Instantiate the correct ProblemAdapter based on the resolved problem key.
    """
    key = resolve_problem_key(problem)

    adapter_cls = PROBLEM_REGISTRY.get(key)
    if adapter_cls is None:
        known = ", ".join(sorted(PROBLEM_REGISTRY.keys()))
        raise ValueError(f"Unknown problem name='{key}'. Known: {known}")

    return adapter_cls(problem)
