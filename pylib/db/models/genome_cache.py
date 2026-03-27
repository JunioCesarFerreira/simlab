from typing import TypedDict, Optional, Any
from bson import ObjectId


class GenomeCache(TypedDict):
    experiment_id: ObjectId
    genome_hash: str
    chromosome: dict[str, Any]
    objectives: Optional[list[float]]
