from copy import deepcopy
from typing import Any, Optional
from bson import ObjectId

from pylib.db.models import Experiment
from api.domain.experiment import ExperimentDto, ExperimentFullDto, ExperimentInfoDto, ParetoFrontItemDto
from api.domain.generation import GenerationDto
from api.mappers.helpers import oid_to_str, str_to_oid, pop_id, dict_oid_to_str, dict_str_to_oid


def _pareto_from_mongo(pf: Optional[list[dict]]) -> Optional[list[ParetoFrontItemDto]]:
    if not pf:
        return None
    return [{"chromosome": item.get("chromosome", {}), "objectives": item.get("objectives", {})}
            for item in pf]


def _pareto_to_mongo(pf: Optional[list[ParetoFrontItemDto]]) -> Optional[list[dict]]:
    if not pf:
        return None
    return [{"chromosome": item.get("chromosome", {}), "objectives": item.get("objectives", {})}
            for item in pf]


def _analysis_files_to_str(d: dict) -> dict[str, str]:
    return {k: oid_to_str(v) for k, v in (d or {}).items()}


def _analysis_files_to_oid(d: dict) -> dict[str, ObjectId]:
    return {k: str_to_oid(v) for k, v in (d or {}).items() if str_to_oid(v) is not None}


def experiment_from_mongo(doc: dict) -> ExperimentDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "status": d.get("status", ""),
        "system_message": d.get("system_message", ""),
        "created_time": d.get("created_time"),
        "start_time": d.get("start_time"),
        "end_time": d.get("end_time"),
        "parameters": d.get("parameters", {}),
        "source_repository_options": dict_oid_to_str(d.get("source_repository_options", {})),
        "data_conversion_config": d.get("data_conversion_config", {}),
        "pareto_front": _pareto_from_mongo(d.get("pareto_front")),
        "analysis_files": _analysis_files_to_str(d.get("analysis_files", {})),
    }


def experiment_full_from_mongo(doc: dict, generations: list[GenerationDto]) -> ExperimentFullDto:
    base = experiment_from_mongo(doc)
    return {**base, "generations": generations}


def experiment_info_from_mongo(doc: dict) -> ExperimentInfoDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "system_message": d.get("system_message", ""),
        "start_time": d.get("start_time"),
        "end_time": d.get("end_time"),
    }


def experiment_to_mongo(dto: ExperimentDto) -> Experiment:
    d = deepcopy(dto)
    exp: dict[str, Any] = {}

    if d.get("id"):
        exp["_id"] = str_to_oid(d["id"])

    exp["name"] = d.get("name", "")
    exp["status"] = d.get("status", "") or ""
    exp["system_message"] = d.get("system_message", "") or ""
    exp["created_time"] = d.get("created_time")
    exp["start_time"] = d.get("start_time")
    exp["end_time"] = d.get("end_time")
    exp["parameters"] = d.get("parameters", {})
    exp["source_repository_options"] = dict_str_to_oid(d.get("source_repository_options") or {})
    exp["data_conversion_config"] = d.get("data_conversion_config") or {}
    exp["pareto_front"] = _pareto_to_mongo(d.get("pareto_front"))
    exp["analysis_files"] = _analysis_files_to_oid(d.get("analysis_files") or {})

    return exp  # type: ignore[return-value]
