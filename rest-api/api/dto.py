from typing import TypedDict, Any, Optional, NotRequired
from datetime import datetime

from pylib.dto.simulator import SimulationConfig
from pylib.dto.database import Parameters

# ---------------------------------------------------------------------------------------------------------
# API DTO
# dto uses str for ObjectId fields
# ---------------------------------------------------------------------------------------------------------

class SimulationDto(TypedDict):
    id: str
    experiment_id: str
    status: str
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
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: str
    simulations_ids: list[str]

class GenerationDto(TypedDict):
    index: int
    population: list[IndividualDto]

class BatchDto(TypedDict):
    id: str
    status: str
    start_time: datetime
    end_time: datetime
    simulations_ids: list[str]

class MetricItemDto(TypedDict):
    name: str
    kind: str
    column: str
    q: NotRequired[float] = None
    scale: NotRequired[float] = None

class DataConversionConfigDto(TypedDict):
    node_col: str
    time_col: str
    metrics: list[MetricItemDto]

class ParetoFrontItemDto(TypedDict):
    chromosome: dict[str, Any]
    objectives: dict[str, float]

class ExperimentDto(TypedDict):
    id: Optional[str] = None
    name: str
    status: Optional[str] = 'Building'
    system_message: Optional[str]
    created_time: datetime | None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parameters: Parameters
    generations: list[GenerationDto]
    source_repository_options: dict[str, str]
    data_conversion_config: DataConversionConfigDto
    pareto_front: Optional[list[ParetoFrontItemDto]] = None
    analysis_files: NotRequired[dict[str, str]] = None

# ---------------------------------------------------------------------------------------------------------
# Additional DTOs
class ExperimentInfoDto(TypedDict):
    id: Optional[str] = None
    name: str
    system_message: Optional[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

