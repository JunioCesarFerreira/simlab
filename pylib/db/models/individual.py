from typing import NotRequired, TypedDict, Any
from bson import ObjectId


class Individual(TypedDict):
    experiment_id: ObjectId
    generation_id: ObjectId
    individual_id: str           # hash of chromosome
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: ObjectId
    # Provenance of `objectives`. Absent on documents written before the
    # adaptive strategies existed, which are all ground truth by construction:
    # readers must treat a missing value as "simulated".
    evaluation_source: NotRequired[str]  # "simulated" | "cache" | "penalty" | "estimated"
