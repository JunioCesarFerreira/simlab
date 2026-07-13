from typing import TypedDict, Any, Optional
from datetime import datetime
from bson import ObjectId


class MetricItem(TypedDict):
    name: str
    kind: str
    column: str
    # Optional parameters (validated according to the type)
    q: Optional[float]      # required if kind == QUANTILE (0<q<=1)
    scale: Optional[float]  # required if kind == INVERSE_MEDIAN (scale>0)


class ObjetiveItem(TypedDict):
    metric_name: str
    goal: str  # "min" or "max"


class DataConversionConfig(TypedDict):
    node_col: str
    time_col: str
    metrics: list[MetricItem]


class SyntheticConfig(TypedDict, total=False):
    """Per-experiment synthetic benchmark configuration.

    When ``enabled`` is True the master-node evaluates a classical benchmark
    function (DTLZ2, ZDT1 or SCH1) instead of running a Cooja simulation.
    All fields are optional; missing values fall back to the master-node
    environment variables (ENABLE_DATA_SYNTHETIC, BENCH, NOISE_STD).
    """
    enabled: bool
    bench: str    # "DTLZ2" | "ZDT1" | "SCH1"
    noise_std: float


class Parameters(TypedDict):
    strategy: str
    algorithm: dict[str, Any]  # Algorithm parameters defined in config/algorithm.py
    simulation: dict[str, Any]
    problem: dict[str, Any]    # Problem parameters defined in config/problems.py
    objectives: list[ObjetiveItem]


class ParetoFrontItem(TypedDict):
    chromosome: dict[str, Any]
    objectives: dict[str, float]


class RuntimeMetricsArtifact(TypedDict):
    """Reference to the immutable raw-telemetry artifact stored in GridFS."""
    storage: str        # "gridfs"
    file_id: ObjectId
    filename: str
    content_type: str   # parquet or gzip (CSV fallback)
    compression: str
    size_bytes: int
    sha256: str
    schema_version: int


class RuntimeMetrics(TypedDict, total=False):
    """Summary + artifact reference for the computational telemetry of a run.

    The full time series live only in the GridFS artifact; this block stays
    small on purpose. New per-metric summaries may be added alongside ``cpu``
    and ``memory`` without breaking existing documents.
    """
    status: str  # "collecting" | "completed" | "no_data" | "failed"
    started_at: datetime
    finished_at: datetime
    collection_finished_at: datetime
    collection: dict[str, Any]   # source, prometheus_url, query_step, ...
    artifact: RuntimeMetricsArtifact
    summary: dict[str, Any]      # duration_seconds, cpu{...}, memory{...}
    error: str


class Experiment(TypedDict):
    id: str
    name: str
    status: str
    system_message: str
    created_time: datetime
    start_time: datetime
    end_time: datetime
    parameters: Parameters
    source_repository_options: dict[str, ObjectId]
    data_conversion_config: DataConversionConfig
    pareto_front: Optional[list[ParetoFrontItem]]
    analysis_files: dict[str, ObjectId]
    runtime_metrics: Optional[RuntimeMetrics]
