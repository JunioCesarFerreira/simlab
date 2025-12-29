from typing import TypedDict, Any, Optional, NotRequired
from datetime import datetime
from bson import ObjectId
 
from pylib.dto.simulator import SimulationConfig 
from pylib.dto.database import (
    Simulation, 
    Generation, 
    Experiment, 
    Parameters, 
    SourceRepository
)
    
#---------------------------------------------------------------------------------------------------------
# API DTO ------------------------------------------------------------------------------------------------
# dto uses str for ObjectId fields
#---------------------------------------------------------------------------------------------------------

class SimulationDto(TypedDict):
    id: str
    experiment_id: str
    generation_id: str
    status: str
    start_time: datetime
    end_time: datetime
    parameters: SimulationConfig
    pos_file_id: str
    csc_file_id: str
    log_cooja_id: str
    runtime_log_id: str
    csv_log_id: str
    topology_picture_id: str
    objectives: dict[str,float]
    metrics: dict[str,float]
    
class GenerationDto(TypedDict):
    id: str
    index: int
    experiment_id: str
    status: str
    start_time: datetime
    end_time: datetime
    simulations_ids: list[str]

class ObjectiveItemDto(TypedDict):
    name: str
    kind: str
    column: str
    goal: str
    # Optional parameters (validated according to the type)
    q: NotRequired[float] = None         # required if kind == QUANTILE (0<q<=1)
    scale: NotRequired[float] = None     # required if kind == INVERSE_MEDIAN (scale>0)

class MetricItemDto(TypedDict):
    name: str
    kind: str
    column: str
    # Optional parameters (validated according to the type)
    q: NotRequired[float] = None         # required if kind == QUANTILE (0<q<=1)

class TransformConfigDto(TypedDict):
    node_col: str
    time_col: str
    objectives: list[ObjectiveItemDto]
    metrics: list[MetricItemDto]

class ExperimentDto(TypedDict):
    id: Optional[str] = None
    name: str
    status: Optional[str] = 'Building'
    created_time: datetime | None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    parameters: Parameters
    generations: list[str]
    source_repository_id: str
    transform_config: TransformConfigDto
    pareto_front: Optional[Any] = None
    
#---------------------------------------------------------------------------------------------------------
# Converters Mongo ↔ DTO
from copy import deepcopy

# --- helpers --------------------------------------------------------

def _oid_to_str(x) -> str:
    """Converte ObjectId -> str; se já for str, retorna; se vazio/None, retorna ''."""
    if x is None:
        return ""
    if isinstance(x, ObjectId):
        return str(x)
    return str(x)

def _str_to_oid(x: Optional[str]) -> Optional[ObjectId]:
    """Converte str -> ObjectId; se string vazia/None, retorna None (útil para campos opcionais)."""
    if not x:
        return None
    return ObjectId(x)

def _pop_id_fields(doc: dict) -> tuple[Optional[str], dict]:
    """
    Extrai id de doc['_id'] (ou 'id' fallback) e retorna (id_str, doc_sem_ids).
    Não muta o dicionário original.
    """
    d = dict(doc)
    _id = d.pop("_id", None)
    id_fallback = d.pop("id", None) if _id is None else None
    id_str = _oid_to_str(_id if _id is not None else id_fallback)
    return id_str, d

def _list_oid_to_str(lst: list) -> list[str]:
    return [_oid_to_str(x) for x in (lst or [])]

def _list_str_to_oid(lst: list[str]) -> list[ObjectId]:
    return [ObjectId(x) for x in (lst or []) if x]

def _ensure_datetime(x):
    # FastAPI/Pydantic podem serializar datetime nativamente.
    # Este método é para caso no futuro queira converter string ISO.
    return x

# --- Simulation -----------------------------------------------------

def simulation_from_mongo(doc: dict) -> SimulationDto:
    """
    Converte um documento Mongo (Simulation) em SimulationDto.
    Aceita '_id' e converte todos os ObjectIds para str.
    """
    if not doc:
        raise ValueError("simulation_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    return {
        "id": id_str,
        "experiment_id": _oid_to_str(d.get("experiment_id")),
        "generation_id": _oid_to_str(d.get("generation_id")),
        "status": d.get("status", ""),
        "start_time": _ensure_datetime(d.get("start_time")),
        "end_time": _ensure_datetime(d.get("end_time")),
        "parameters": d.get("parameters", {}),  # tipo: SimulationConfig
        "pos_file_id": _oid_to_str(d.get("pos_file_id")),
        "csc_file_id": _oid_to_str(d.get("csc_file_id")),
        "log_cooja_id": _oid_to_str(d.get("log_cooja_id")),
        "runtime_log_id": _oid_to_str(d.get("runtime_log_id")),
        "csv_log_id": _oid_to_str(d.get("csv_log_id")),
        "topology_picture_id": _oid_to_str(d.get("topology_picture_id")),
        "objectives": d.get("objectives", {}),
        "metrics": d.get("metrics", {}),
    }

def simulation_to_mongo(dto: SimulationDto) -> Simulation:
    """
    Converte SimulationDto -> estrutura Mongo (Simulation).
    - 'id' (str) vira '_id' (ObjectId) se presente.
    - Campos *file_id (str) viram ObjectId quando não vazios.
    """
    if not dto:
        raise ValueError("simulation_to_mongo: dto vazio")

    d = deepcopy(dto)

    sim: dict[str, Any] = {}
    if d.get("id"):
        sim["_id"] = _str_to_oid(d["id"])

    sim["experiment_id"] = _str_to_oid(d["experiment_id"])
    sim["generation_id"] = _str_to_oid(d["generation_id"])
    sim["status"] = d.get("status", "")
    sim["start_time"] = _ensure_datetime(d.get("start_time"))
    sim["end_time"] = _ensure_datetime(d.get("end_time"))
    sim["parameters"] = d.get("parameters", {})
    sim["objectives"] = d.get("objectives", {})
    sim["metrics"] = d.get("metrics", {})

    # IDs de arquivos
    for k in ("pos_file_id", "csc_file_id", "log_cooja_id", "runtime_log_id", "csv_log_id", "topology_picture_id"):
        oid = _str_to_oid(d.get(k))
        if oid is not None:
            sim[k] = oid
        else:
            sim[k] = None

    return sim  # type: ignore[return-value]

# --- Generation -----------------------------------------------------

def generation_from_mongo(doc: dict) -> GenerationDto:
    """
    Converte documento Mongo (Generation) -> GenerationDto (ObjectIds como str).
    """
    if not doc:
        raise ValueError("generation_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    return {
        "id": id_str,
        "index": d.get("index", 0),
        "experiment_id": _oid_to_str(d.get("experiment_id")),
        "status": d.get("status", ""),
        "start_time": _ensure_datetime(d.get("start_time")),
        "end_time": _ensure_datetime(d.get("end_time")),
        "simulations_ids": _list_oid_to_str(d.get("simulations_ids", [])),
    }

def generation_to_mongo(dto: GenerationDto) -> Generation:
    """
    Converte GenerationDto -> estrutura Mongo (Generation).
    """
    if not dto:
        raise ValueError("generation_to_mongo: dto vazio")

    d = deepcopy(dto)
    gen: dict[str, Any] = {}

    if d.get("id"):
        gen["_id"] = _str_to_oid(d["id"])

    gen["index"] = d.get("index", 0)
    gen["experiment_id"] = _str_to_oid(d["experiment_id"])
    gen["status"] = d.get("status", "")
    gen["start_time"] = _ensure_datetime(d.get("start_time"))
    gen["end_time"] = _ensure_datetime(d.get("end_time"))
    gen["simulations_ids"] = _list_str_to_oid(d.get("simulations_ids", []))

    return gen  # type: ignore[return-value]

# --- Experiment -----------------------------------------------------

def experiment_from_mongo(doc: dict) -> ExperimentDto:
    """
    Converte documento Mongo (Experiment) -> ExperimentDto (ObjectIds como str).
    """
    if not doc:
        raise ValueError("experiment_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    return {
        "id": id_str,
        "name": d.get("name", ""),
        "status": d.get("status", "Building"),
        "created_time": _ensure_datetime(d.get("created_time")),
        "start_time": _ensure_datetime(d.get("start_time")),
        "end_time": _ensure_datetime(d.get("end_time")),
        "parameters": d.get("parameters", {}),
        "generations": _list_oid_to_str(d.get("generations", [])),
        "source_repository_id": _oid_to_str(d.get("source_repository_id") or d.get("source_repository_id".replace("_id","")) or d.get("source_repository_id_str")),
        "transform_config": d.get("transform_config", {}),
        "pareto_front": d.get("pareto_front", {}),
    }

def experiment_to_mongo(dto: ExperimentDto) -> Experiment:
    """
    Converte ExperimentDto -> estrutura Mongo (Experiment).
    - 'id' (str) vira '_id' (ObjectId) se presente.
    - 'generations' (list[str]) vira list[ObjectId].
    - 'source_repository_id' (str) permanece string SE for modelo atual,
      mas se você for migrar para ObjectId, troque por _str_to_oid aqui.
    """
    if not dto:
        raise ValueError("experiment_to_mongo: dto vazio")

    d = deepcopy(dto)
    exp: dict[str, Any] = {}

    if d.get("id"):
        exp["_id"] = _str_to_oid(d["id"])

    exp["name"] = d.get("name", "")
    exp["status"] = d.get("status", "Building") or "Building"
    exp["created_time"] = _ensure_datetime(d.get("created_time"))
    exp["start_time"] = _ensure_datetime(d.get("start_time"))
    exp["end_time"] = _ensure_datetime(d.get("end_time"))
    exp["parameters"] = d.get("parameters", {})
    exp["generations"] = _list_str_to_oid(d.get("generations", []))
    exp["source_repository_id"] = d.get("source_repository_id", "")
    exp["transform_config"] = d.get("transform_config", {}) or {}
    exp["pareto_front"] = d.get("pareto_front", {}) or {}
    
    return exp 

# --- SourceRepository (opcional) -----------------------------------

def source_repository_from_mongo(doc: dict) -> SourceRepository:
    """
    Converte documento Mongo (SourceRepository) -> SourceRepository TypedDict.
    Garante 'id' como str e 'source_files' com ids em str.
    """
    if not doc:
        raise ValueError("source_repository_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    # Normaliza lista de arquivos
    sf_list = []
    for sf in d.get("source_files", []) or []:
        sf_list.append({
            "id": _oid_to_str(sf.get("id") or sf.get("_id")),
            "file_name": sf.get("file_name", ""),
        })

    return {
        "id": id_str or d.get("id", ""),
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "source_files": sf_list,
    }

def source_repository_to_mongo(sr: SourceRepository) -> dict:
    """
    Converte SourceRepository TypedDict -> doc Mongo.
    Mantém 'source_files.id' como str (GridFS usa ObjectId; converta se desejar).
    """
    d = deepcopy(sr)
    out: dict[str, Any] = {}

    if d.get("id"):
        out["_id"] = _str_to_oid(d["id"])

    out["name"] = d.get("name", "")
    out["description"] = d.get("description", "")
    out["source_files"] = [{"id": f["id"], "file_name": f.get("file_name", "")} for f in d.get("source_files", [])]

    return out
