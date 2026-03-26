from typing import TypedDict, Any
from bson import ObjectId


class Individual(TypedDict):
    experiment_id: ObjectId
    generation_id: ObjectId
    individual_id: str           # hash of chromosome
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: ObjectId
