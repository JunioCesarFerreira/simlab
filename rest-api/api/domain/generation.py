from typing import TypedDict, Any, Optional, NotRequired
from datetime import datetime


class CompletenessDto(TypedDict, total=False):
    """What an objective vector rests on.

    Published because an objective alone cannot be read: a mean over one
    replica and a mean over five look identical, and the difference is the
    difference between a measurement and an anecdote. ``state`` is the short
    answer, the counts are the evidence, and ``objective_used`` says whether
    the configured policy let this evaluation reach the algorithm.

    Absent on an individual evaluated before this was recorded — which is
    itself information, and better than a fabricated "complete".
    """
    state: str
    expected: int
    observed: int
    failed: int
    pending: int
    absent: int
    expected_seeds: list[int]
    observed_seeds: list[int]
    missing_seeds: list[int]
    failed_seeds: list[int]
    per_metric: dict[str, int]
    attempts: int
    objective_used: bool
    policy: str
    #: Carried over from an earlier evaluation of the same genome, with that
    #: evaluation's decision intact. A reused individual has no simulations in
    #: its own generation to count, which is why the record travels with it.
    reused: bool
    source: str
    #: Why there were no replicas, for the states that never had any.
    reason: str


class IndividualDto(TypedDict):
    id: str
    individual_id: str
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: str
    simulations_ids: NotRequired[list[str]]
    #: ``None`` when the evaluation predates the record, not when it is
    #: complete. The two are different states and the GUI says which.
    completeness: NotRequired[Optional[CompletenessDto]]


class GenerationDto(TypedDict):
    id: str
    experiment_id: str
    index: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    population: list[IndividualDto]
