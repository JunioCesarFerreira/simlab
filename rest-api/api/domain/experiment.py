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


class RuntimeMetricsArtifactDto(TypedDict, total=False):
    storage: str
    file_id: str
    filename: str
    content_type: str
    compression: str
    size_bytes: int
    sha256: str
    schema_version: int


class RuntimeMetricsDto(TypedDict, total=False):
    """Computational telemetry summary of an experiment run.

    Only the summary and the GridFS artifact reference — the full time series
    are served on demand by GET /experiments/{id}/runtime-metrics.
    """
    status: str
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    collection_finished_at: Optional[datetime]
    summary: dict[str, Any]
    artifact: RuntimeMetricsArtifactDto
    error: str


class FirmwareFileDto(TypedDict, total=False):
    file_name: str
    file_id: str          # GridFS copy owned by the experiment
    origin_file_id: str   # file in the shared source repository
    size_bytes: int
    sha256: str


class FirmwareRepositorySnapshotDto(TypedDict, total=False):
    option_keys: list[str]
    source_repository_id: str
    name: str
    description: str
    files: list[FirmwareFileDto]
    missing_files: list[FirmwareFileDto]


class FirmwareSnapshotDto(TypedDict, total=False):
    """Immutable record of the firmware an experiment was executed with."""
    status: str
    captured_at: Optional[datetime]
    schema_version: int
    repositories: list[FirmwareRepositorySnapshotDto]
    reason: str
    error: str


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
    runtime_metrics: NotRequired[Optional[RuntimeMetricsDto]]
    firmware_snapshot: NotRequired[Optional[FirmwareSnapshotDto]]


class ExperimentFullDto(TypedDict):
    """Experiment with all generations and individuals fully embedded."""
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
    analysis_files: NotRequired[dict[str, str]]
    runtime_metrics: NotRequired[Optional[RuntimeMetricsDto]]
    firmware_snapshot: NotRequired[Optional[FirmwareSnapshotDto]]
    generations: list[GenerationDto]


class ExperimentInfoDto(TypedDict, total=False):
    id: Optional[str]
    name: str
    # Present in GET /experiments/ responses; omitted by /by-status/{status},
    # where the status is implied by the route.
    status: Optional[str]
    system_message: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    is_synthetic: bool
    synthetic_bench: Optional[str]
