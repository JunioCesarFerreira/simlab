
# ---------------------------------------------------------------------------------------------------------
# Converters Mongo ↔ DTO
from copy import deepcopy
from typing import Optional, Any
from bson import ObjectId

from pylib.dto.database import (
    Simulation,
    Experiment,
    SourceRepository,
    Batch,
)
from api.dto import (
    IndividualDto,
    ParetoFrontItemDto, 
    GenerationDto, 
    SimulationDto, 
    BatchDto, 
    ExperimentDto
)


def _oid_to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, ObjectId):
        return str(x)
    return str(x)


def _str_to_oid(x: Optional[str]) -> Optional[ObjectId]:
    if not x:
        return None
    return ObjectId(x)


def _pop_id_fields(doc: dict) -> tuple[Optional[str], dict]:
    d = dict(doc)
    _id = d.pop("_id", None)
    id_fallback = d.pop("id", None) if _id is None else None
    id_str = _oid_to_str(_id if _id is not None else id_fallback)
    return id_str, d


def _list_oid_to_str(lst: list) -> list[str]:
    return [_oid_to_str(x) for x in (lst or [])]


def _list_str_to_oid(lst: list[str]) -> list[ObjectId]:
    return [ObjectId(x) for x in (lst or []) if x]


def _dict_oid_to_str(d: dict[str, ObjectId]) -> dict[str, str]:
    if not d:
        return {}
    return {k: str(v) for k, v in d.items()}


def _dict_str_to_oid(d: dict[str, str]) -> dict[str, ObjectId]:
    if not d:
        return {}
    return {k: ObjectId(v) for k, v in d.items() if v}


def _ensure_datetime(x):
    return x


# --- Pareto front (ajuste mínimo: remove simulation_id) -----------------------

def _pareto_front_from_mongo(pf: Optional[list[dict]]) -> Optional[list[ParetoFrontItemDto]]:
    if not pf:
        return None

    out: list[ParetoFrontItemDto] = []
    for item in pf:
        out.append({
            "chromosome": item.get("chromosome", {}),
            "objectives": item.get("objectives", {}),
        })
    return out


def _pareto_front_to_mongo(pf: Optional[list[ParetoFrontItemDto]]) -> Optional[list[dict]]:
    if not pf:
        return None

    out: list[dict] = []
    for item in pf:
        out.append({
            "chromosome": item.get("chromosome", {}),
            "objectives": item.get("objectives", {}),
        })
    return out


def _dict_analysis_files_to_str(d: dict[str, ObjectId]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (d or {}).items():
        out[k] = _oid_to_str(v)
    return out


def _dict_analysis_files_to_oid(d: dict[str, str]) -> dict[str, ObjectId]:
    out: dict[str, ObjectId] = {}
    for k, v in (d or {}).items():
        oid = _str_to_oid(v)
        if oid is not None:
            out[k] = oid
    return out


# --- Generations embedadas (mínimo necessário) --------------------------------

def _generations_from_mongo(gens: list[dict] | None) -> list[GenerationDto]:
    out: list[GenerationDto] = []
    for g in (gens or []):
        pop_out: list[IndividualDto] = []
        for ind in (g.get("population", []) or []):
            pop_out.append({
                "chromosome": ind.get("chromosome", {}),
                "objectives": ind.get("objectives", []),
                "topology_picture_id": _oid_to_str(ind.get("topology_picture_id")),
                "simulations_ids": _list_oid_to_str(ind.get("simulations_ids", [])),
            })

        out.append({
            "index": g.get("index", 0),
            "population": pop_out,
        })
    return out


def _generations_to_mongo(gens: list[GenerationDto] | None) -> list[dict]:
    out: list[dict] = []
    for g in (gens or []):
        pop_out: list[dict] = []
        for ind in (g.get("population", []) or []):
            pop_out.append({
                "chromosome": ind.get("chromosome", {}),
                "objectives": ind.get("objectives", []),
                "topology_picture_id": _str_to_oid(ind.get("topology_picture_id")),
                "simulations_ids": _list_str_to_oid(ind.get("simulations_ids", [])),
            })

        out.append({
            "index": g.get("index", 0),
            "population": pop_out,
        })
    return out


# --- Simulation -----------------------------------------------------

def simulation_from_mongo(doc: dict) -> SimulationDto:
    if not doc:
        raise ValueError("simulation_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    return {
        "id": id_str,
        "experiment_id": _oid_to_str(d.get("experiment_id")),
        "status": d.get("status", ""),
        "random_seed": d.get("random_seed", 0),
        "start_time": _ensure_datetime(d.get("start_time")),
        "end_time": _ensure_datetime(d.get("end_time")),
        "parameters": d.get("parameters", {}),
        "pos_file_id": _oid_to_str(d.get("pos_file_id")),
        "csc_file_id": _oid_to_str(d.get("csc_file_id")),
        "source_repository_id": _oid_to_str(d.get("source_repository_id")),
        "log_cooja_id": _oid_to_str(d.get("log_cooja_id")),
        "runtime_log_id": _oid_to_str(d.get("runtime_log_id")),
        "csv_log_id": _oid_to_str(d.get("csv_log_id")),
        "network_metrics": d.get("network_metrics", {}),
    }


def simulation_to_mongo(dto: SimulationDto) -> Simulation:
    if not dto:
        raise ValueError("simulation_to_mongo: dto vazio")

    d = deepcopy(dto)
    sim: dict[str, Any] = {}

    if d.get("id"):
        sim["_id"] = _str_to_oid(d["id"])

    sim["status"] = d.get("status", "")
    sim["random_seed"] = d.get("random_seed", 0)
    sim["start_time"] = _ensure_datetime(d.get("start_time"))
    sim["end_time"] = _ensure_datetime(d.get("end_time"))
    sim["parameters"] = d.get("parameters", {})
    sim["network_metrics"] = d.get("network_metrics", {})

    oid_fields = (
        "experiment_id",
        "pos_file_id",
        "csc_file_id",
        "source_repository_id",
        "log_cooja_id",
        "runtime_log_id",
        "csv_log_id",
    )
    for k in oid_fields:
        sim[k] = _str_to_oid(d.get(k))

    return sim  # type: ignore[return-value]


# --- Batch ----------------------------------------------------------

def batch_from_mongo(doc: dict) -> BatchDto:
    if not doc:
        raise ValueError("batch_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    return {
        "id": id_str,
        "status": d.get("status", ""),
        "start_time": _ensure_datetime(d.get("start_time")),
        "end_time": _ensure_datetime(d.get("end_time")),
        "simulations_ids": _list_oid_to_str(d.get("simulations_ids", [])),
    }


def batch_to_mongo(dto: BatchDto) -> Batch:
    if not dto:
        raise ValueError("batch_to_mongo: dto vazio")

    d = deepcopy(dto)
    b: dict[str, Any] = {}

    if d.get("id"):
        b["_id"] = _str_to_oid(d["id"])

    b["status"] = d.get("status", "")
    b["start_time"] = _ensure_datetime(d.get("start_time"))
    b["end_time"] = _ensure_datetime(d.get("end_time"))
    b["simulations_ids"] = _list_str_to_oid(d.get("simulations_ids", []))

    return b  # type: ignore[return-value]


# --- Experiment -----------------------------------------------------

def experiment_from_mongo(doc: dict) -> ExperimentDto:
    if not doc:
        raise ValueError("experiment_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

    return {
        "id": id_str,
        "name": d.get("name", ""),
        "status": d.get("status", "Building"),
        "system_message": d.get("system_message", ""),
        "created_time": _ensure_datetime(d.get("created_time")),
        "start_time": _ensure_datetime(d.get("start_time")),
        "end_time": _ensure_datetime(d.get("end_time")),
        "parameters": d.get("parameters", {}),
        "generations": _generations_from_mongo(d.get("generations", [])),
        "source_repository_options": _dict_oid_to_str(d.get("source_repository_options", {})),
        "data_conversion_config": d.get("data_conversion_config", {}),
        "pareto_front": _pareto_front_from_mongo(d.get("pareto_front")),
        "analysis_files": _dict_analysis_files_to_str(d.get("analysis_files", {})),
    }


def experiment_to_mongo(dto: ExperimentDto) -> Experiment:
    if not dto:
        raise ValueError("experiment_to_mongo: dto vazio")

    d = deepcopy(dto)
    exp: dict[str, Any] = {}

    if d.get("id"):
        exp["_id"] = _str_to_oid(d["id"])

    exp["name"] = d.get("name", "")
    exp["status"] = d.get("status", "Building") or "Building"
    exp["system_message"] = d.get("system_message", "") or ""
    exp["created_time"] = _ensure_datetime(d.get("created_time"))
    exp["start_time"] = _ensure_datetime(d.get("start_time"))
    exp["end_time"] = _ensure_datetime(d.get("end_time"))
    exp["parameters"] = d.get("parameters", {})
    exp["generations"] = _generations_to_mongo(d.get("generations", []))
    exp["source_repository_options"] = _dict_str_to_oid(d.get("source_repository_options", {}) or {})
    exp["data_conversion_config"] = d.get("data_conversion_config", {}) or {}
    exp["pareto_front"] = _pareto_front_to_mongo(d.get("pareto_front"))
    exp["analysis_files"] = _dict_analysis_files_to_oid(d.get("analysis_files", {}) or {})

    return exp  # type: ignore[return-value]


# --- SourceRepository (opcional) -----------------------------------

def source_repository_from_mongo(doc: dict) -> SourceRepository:
    if not doc:
        raise ValueError("source_repository_from_mongo: doc vazio")

    id_str, d = _pop_id_fields(doc)

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
    d = deepcopy(sr)
    out: dict[str, Any] = {}

    if d.get("id"):
        out["_id"] = _str_to_oid(d["id"])

    out["name"] = d.get("name", "")
    out["description"] = d.get("description", "")
    out["source_files"] = [
        {"id": f["id"], "file_name": f.get("file_name", "")}
        for f in d.get("source_files", []) or []
    ]

    return out
