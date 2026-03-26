from typing import Optional, NotRequired
from datetime import datetime
from typing import TypedDict, Any

from pylib.db.models import Parameters
from api.domain.generation import GenerationDto


class MetricItemDto(TypedDict):
    name: str
    kind: str
    column: str
    q: NotRequired[float]
    scale: NotRequired[float]


class DataConversionConfigDto(TypedDict):
    node_col: str
    time_col: str
    metrics: list[MetricItemDto]


class ParetoFrontItemDto(TypedDict):
    chromosome: dict[str, Any]
    objectives: dict[str, float]


class ExperimentDto(TypedDict):
    id: Optional[str]
    name: str
    status: Optional[str]
    system_message: Optional[str]
    created_time: Optional[datetime]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    parameters: Parameters
    source_repository_options: dict[str, str]
    data_conversion_config: DataConversionConfigDto
    pareto_front: Optional[list[ParetoFrontItemDto]]
    generations: NotRequired[list[GenerationDto]]
    analysis_files: NotRequired[dict[str, str]]


class ExperimentInfoDto(TypedDict):
    id: Optional[str]
    name: str
    system_message: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
