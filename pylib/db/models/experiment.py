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
