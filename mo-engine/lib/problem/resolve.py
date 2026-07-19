import random
import logging

from typing import Any, Mapping
from typing import Type

from .adapter import ProblemAdapter
from .p0_synthetic import Problem0SyntheticAdapter
from .p1_continuous_mobility import Problem1ContinuousMobilityAdapter
from .p2_discrete_mobility import Problem2DiscreteMobilityAdapter
from .p3_target_coverage import Problem3TargetCoverageAdapter
from .p4_mobile_sink_collection import Problem4MobileSinkCollectionAdapter

log = logging.getLogger(__name__)

# Problem key -> adapter class
PROBLEM_REGISTRY: dict[str, Type[ProblemAdapter]] = {
    "problem0": Problem0SyntheticAdapter,
    "problem1": Problem1ContinuousMobilityAdapter,
    "problem2": Problem2DiscreteMobilityAdapter,
    "problem3": Problem3TargetCoverageAdapter,
    "problem4": Problem4MobileSinkCollectionAdapter,
}

# GA parameter keys consumed by the strategies themselves (lib/strategy/*),
# regardless of the problem. Problem-specific keys are declared per adapter
# in CONSUMED_GA_KEYS. selection_method is accepted-but-fixed: every strategy
# currently uses tournament selection.
STRATEGY_GA_KEYS = frozenset({
    "population_size", "number_of_generations", "random_seed",
    "divisions", "prob_cx", "prob_mt", "selection_method",
})


def _warn_ignored_ga_keys(key: str, adapter_cls: Type[ProblemAdapter], ga_parameter: Mapping[str, Any]) -> None:
    ignored = set(ga_parameter) - STRATEGY_GA_KEYS - adapter_cls.CONSUMED_GA_KEYS
    if ignored:
        log.warning(
            "[%s] GA parameter(s) ignored by this problem: %s",
            key, ", ".join(sorted(ignored)),
        )


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


def build_test_adapter(problem: Mapping[str, Any]) -> ProblemAdapter:
    """
    Instantiate the correct ProblemAdapter based on the resolved problem key.
    """
    key = resolve_problem_key(problem)

    adapter_cls = PROBLEM_REGISTRY.get(key)
    if adapter_cls is None:
        known = ", ".join(sorted(PROBLEM_REGISTRY.keys()))
        raise ValueError(f"Unknown problem name='{key}'. Known: {known}")

    adptr = adapter_cls(problem)
          
    return adptr
  
  
def build_adapter(
    problem: Mapping[str, Any], 
    ga_parameter: Mapping[str, Any], 
    rng: random.Random
    ) -> ProblemAdapter:
    """
    Instantiate the correct ProblemAdapter based on the resolved problem key.
    """
    key = resolve_problem_key(problem)

    adapter_cls = PROBLEM_REGISTRY.get(key)
    if adapter_cls is None:
        known = ", ".join(sorted(PROBLEM_REGISTRY.keys()))
        raise ValueError(f"Unknown problem name='{key}'. Known: {known}")

    adptr = adapter_cls(problem)

    if len(ga_parameter) > 0:
      _warn_ignored_ga_keys(key, adapter_cls, ga_parameter)
      adptr.set_ga_operator_configs(rng, ga_parameter)

    return adptr
