from typing import TypedDict, Optional, NotRequired
from datetime import datetime

from pylib.config.simulator import SimulationConfig


class SimulationDto(TypedDict):
    id: str
    experiment_id: str
    generation_id: str
    individual_id: str
    status: str
    system_message: str
    random_seed: int
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    parameters: SimulationConfig
    pos_file_id: str
    csc_file_id: str
    source_repository_id: str
    log_cooja_id: str
    runtime_log_id: str
    csv_log_id: str
    network_metrics: dict[str, float]
    # Populated by the master-node after a run (RPL DODAG formation + tree);
    # optional on create so existing simulation payloads remain valid.
    dodag: NotRequired[Optional[dict]]
