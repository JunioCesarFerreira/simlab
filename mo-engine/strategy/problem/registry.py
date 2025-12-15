from typing import Type

from .problem_adapter import ProblemAdapter
from .adapters import (
    Problem1ContinuousMobilityAdapter,
    Problem2DiscreteMobilityAdapter,
    Problem3TargetCoverageAdapter,
    Problem4MobileSinkCollectionAdapter,
)

# Problem key -> adapter class
PROBLEM_REGISTRY: dict[str, Type[ProblemAdapter]] = {
    "problem1": Problem1ContinuousMobilityAdapter,
    "problem2": Problem2DiscreteMobilityAdapter,
    "problem3": Problem3TargetCoverageAdapter,
    "problem4": Problem4MobileSinkCollectionAdapter,
}
