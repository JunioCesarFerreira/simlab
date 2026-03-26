from typing import TypedDict
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
