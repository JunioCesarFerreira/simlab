from typing import TypedDict, Any, Optional, NotRequired
from datetime import datetime

from pylib.config.simulator import SimulationConfig
from pylib.db.models import Parameters

# ---------------------------------------------------------------------------------------------------------
# API DTO
# dto uses str for ObjectId fields
# ---------------------------------------------------------------------------------------------------------

class SimulationDto(TypedDict):
    id: str
    experiment_id: str
    generation_id: str
    individual_id: str
    status: str
    system_message: str
    random_seed: int
    start_time: datetime
    end_time: datetime
    parameters: SimulationConfig
    pos_file_id: str
    csc_file_id: str
    source_repository_id: str
    log_cooja_id: str
    runtime_log_id: str
    csv_log_id: str
    network_metrics: dict[str, float]

class IndividualDto(TypedDict):
    id: str
    individual_id: str
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: str

class GenerationDto(TypedDict):
    id: str
    experiment_id: str
    index: int
    status: str
    start_time: datetime
    end_time: datetime
    population: list[IndividualDto]

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
    created_time: datetime | None
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    parameters: Parameters
    source_repository_options: dict[str, str]
    data_conversion_config: DataConversionConfigDto
    pareto_front: Optional[list[ParetoFrontItemDto]]
    analysis_files: NotRequired[dict[str, str]]

# ---------------------------------------------------------------------------------------------------------
# Additional DTOs
class ExperimentInfoDto(TypedDict):
    id: Optional[str]
    name: str
    system_message: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
