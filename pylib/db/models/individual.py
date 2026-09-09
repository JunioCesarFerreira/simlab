from typing import TypedDict, NotRequired, Any
from bson import ObjectId


class Individual(TypedDict):
    experiment_id: ObjectId
    generation_id: ObjectId
    individual_id: str           # hash of chromosome
    # Position within the generation's population, as the engine produced it.
    # Documents come back from Mongo in no guaranteed order, and a resumed run
    # must rebuild the population in the SAME order or the mating tournament
    # draws different indices and the search diverges. Absent on individuals
    # written before the field existed; readers fall back to the hash order.
    index: NotRequired[int]
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: ObjectId
