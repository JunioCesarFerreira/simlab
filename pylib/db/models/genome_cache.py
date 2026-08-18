from typing import NotRequired, TypedDict, Optional, Any
from bson import ObjectId


class GenomeCache(TypedDict):
    experiment_id: ObjectId
    genome_hash: str
    chromosome: dict[str, Any]
    objectives: Optional[list[float]]
    # Only ground truth is ever cached (simulation, exact replay or the
    # analytical fast path). Estimated objectives never reach this collection.
    evaluation_source: NotRequired[str]
