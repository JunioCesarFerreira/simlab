from typing import Optional, TypedDict
from datetime import datetime
from bson import ObjectId

from pylib.config.simulator import SimulationConfig


class Simulation(TypedDict):
    id: str
    experiment_id: ObjectId
    generation_id: ObjectId      
    individual_id: str           
    status: str
    system_message: str
    random_seed: int
    start_time: datetime
    end_time: datetime
    parameters: SimulationConfig
    pos_file_id: ObjectId
    csc_file_id: ObjectId
    source_repository_id: ObjectId
    log_cooja_id: ObjectId
    runtime_log_id: ObjectId
    csv_log_id: ObjectId
    network_metrics: dict[str, float]
    dodag: Optional[dict]
    # Placement, written at reservation time by the master-node's dispatcher.
    # Host and resource names are denormalized so a finished simulation still
    # says where it ran after the registration is edited or removed.
    assigned_host_id: ObjectId
    assigned_host_name: str
    assigned_resource_id: ObjectId
    assigned_resource_name: str
    assigned_container_id: Optional[str]
    attempt_id: str
    assigned_at: datetime
    remote_pid: Optional[int]
