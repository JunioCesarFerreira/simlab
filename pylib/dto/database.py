from typing import TypedDict, Any, Optional
from datetime import datetime
from bson import ObjectId
 
from .simulator import SimulationConfig 

#---------------------------------------------------------------------------------------------------------    
# Database Structure -------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

class SourceFile(TypedDict):
    id: str
    file_name: str

class SourceRepository(TypedDict):
    id: str
    name : str
    description: str
    source_files: list[SourceFile]

class Simulation(TypedDict):
    id: str
    experiment_id: ObjectId
    batch_id: ObjectId
    status: str
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
    topology_picture_id: ObjectId
    network_metrics: dict[str,float]

class Individual(TypedDict):
    chromosome: dict[str, Any]
    objectives: list[float]
    topology_picture_id: ObjectId
    simulations_ids: list[ObjectId]

class Generation(TypedDict):
    index: int
    population: list[Individual]

class Batch(TypedDict):
    _id: ObjectId
    status: str
    start_time: datetime
    end_time: datetime
    simulations_ids: list[ObjectId]

class MetricItem(TypedDict):
    name: str
    kind: str
    column: str
    # Optional parameters (validated according to the type)
    q: Optional[float] = None         # required if kind == QUANTILE (0<q<=1)
    scale: Optional[float] = None     # required if kind == INVERSE_MEDIAN (scale>0)

class ObjetiveItem(TypedDict):
    metric_name: str
    goal: str # "min" or "max"
    
class DataConversionConfig(TypedDict):
    node_col: str
    time_col: str
    metrics: list[MetricItem]
      
class Parameters(TypedDict):
    strategy: str
    algorithm: dict[str, Any] # Algorithm parameters defined in dto/algorithms.py
    simulation: dict[str, Any] # This is not Simulation or SimulationConfig because it holds only the simulation parameters
    problem: dict[str, Any] # Problem parameters defined in dto/problems.py
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
    generations: list[Generation]
    source_repository_options: dict[str, ObjectId]
    data_conversion_config: DataConversionConfig
    pareto_front: Optional[list[ParetoFrontItem]] = None
    analysis_files: dict[str, ObjectId]