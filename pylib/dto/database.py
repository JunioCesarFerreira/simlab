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
    generation_id: ObjectId
    status: str
    start_time: datetime
    end_time: datetime
    parameters: SimulationConfig
    pos_file_id: ObjectId
    csc_file_id: ObjectId
    log_cooja_id: ObjectId
    runtime_log_id: ObjectId
    csv_log_id: ObjectId
    topology_picture_id: ObjectId
    objectives: dict[str,float]
    metrics: dict[str,float]
    
class Generation(TypedDict):
    id: str
    index: int
    experiment_id: ObjectId
    status: str
    start_time: datetime
    end_time: datetime
    simulations_ids: list[ObjectId]

class ObjectiveItem(TypedDict):
    name: str
    kind: str
    column: str
    goal: str
    # Optional parameters (validated according to the type)
    q: Optional[float] = None         # required if kind == QUANTILE (0<q<=1)
    scale: Optional[float] = None     # required if kind == INVERSE_MEDIAN (scale>0)

class MetricItem(TypedDict):
    name: str
    kind: str
    column: str
    # Optional parameters (validated according to the type)
    q: Optional[float] = None         # required if kind == QUANTILE (0<q<=1)

class TransformConfig(TypedDict):
    node_col: str
    time_col: str
    objectives: list[ObjectiveItem]
    metrics: list[MetricItem]
    
class Parameters(TypedDict):
    strategy: str
    algorithm: dict[str, Any] # Algorithm parameters defined in dto/algorithms.py
    simulation: dict[str, Any] # This is not Simulation or SimulationConfig because it holds only the simulation parameters
    problem: dict[str, Any] # Problem parameters defined in dto/problems.py
    
class ParetoFrontItem(TypedDict):
    simulation_id: ObjectId
    chromosome: dict[str, Any]
    objectives: dict[str, float]    

class Experiment(TypedDict):
    id: str
    name: str
    status: str
    created_time: datetime
    start_time: datetime
    end_time: datetime
    parameters: Parameters
    generations: list[ObjectId]
    source_repository_id: str
    transform_config: TransformConfig
    pareto_front: Optional[list[ParetoFrontItem]] = None