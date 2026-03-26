from typing import TypedDict, Any, Optional, NotRequired
from datetime import datetime


class IndividualDto(TypedDict):
    id: str
    individual_id: str
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: str
    simulations_ids: NotRequired[list[str]]


class GenerationDto(TypedDict):
    id: str
    experiment_id: str
    index: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    population: list[IndividualDto]
