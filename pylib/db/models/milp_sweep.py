from datetime import datetime
from typing import Any, NotRequired, Optional, TypedDict
from bson import ObjectId


class MilpSweepProgress(TypedDict):
    total_combos: int
    done: int                 # combinations processed (solved + failed)
    solved: int               # combinations with a usable solution
    infeasible: int
    unique_genotypes: int


class MilpSweepCheckpoint(TypedDict):
    next_index: int           # first combination index not yet processed
    genotypes: list[str]      # genotypes already seen (dedup state)


class MilpSweep(TypedDict):
    """
    One MILP parameter sweep: solve a model over a parameter grid against a
    problem instance, then hand the unique topologies off to a batch
    experiment. Consumed by the milp-engine via change streams.
    """
    name: str
    model_key: str                          # key in MILP_MODEL_SPECS
    problem: dict[str, Any]                 # exported problem JSON (same shape as experiment parameters.problem)
    problem_id: NotRequired[Optional[ObjectId]]  # provenance: problems collection draft
    parameter_grid: dict[str, list[float]]
    fixed_parameters: dict[str, float]
    solver: dict[str, Any]                  # {backend, time_limit_s, mip_gap, allow_fallback}
    batch_options: dict[str, Any]           # passed through to the generated batch experiment
    status: str                             # EnumStatus
    system_message: NotRequired[Optional[str]]
    cancel_requested: NotRequired[bool]
    created_time: datetime
    start_time: NotRequired[Optional[datetime]]
    end_time: NotRequired[Optional[datetime]]
    progress: NotRequired[MilpSweepProgress]
    checkpoint: NotRequired[MilpSweepCheckpoint]
    solutions: NotRequired[list[dict[str, Any]]]  # one entry per grid combination (SolveRecord)
    experiment_id: NotRequired[Optional[ObjectId]]
    campaign_id: NotRequired[Optional[ObjectId]]
