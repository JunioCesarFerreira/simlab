from copy import deepcopy
from typing import Any

from pylib.db.models import Simulation
from api.domain.simulation import SimulationDto
from api.mappers.helpers import oid_to_str, str_to_oid, pop_id


def simulation_from_mongo(doc: dict) -> SimulationDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "experiment_id": oid_to_str(d.get("experiment_id")),
        "generation_id": oid_to_str(d.get("generation_id")),
        "individual_id": str(d.get("individual_id", "")),
        "status": d.get("status", ""),
        "system_message": d.get("system_message", ""),
        "random_seed": d.get("random_seed", 0),
        "start_time": d.get("start_time"),
        "end_time": d.get("end_time"),
        "parameters": d.get("parameters", {}),
        "pos_file_id": oid_to_str(d.get("pos_file_id")),
        "csc_file_id": oid_to_str(d.get("csc_file_id")),
        "source_repository_id": oid_to_str(d.get("source_repository_id")),
        "log_cooja_id": oid_to_str(d.get("log_cooja_id")),
        "runtime_log_id": oid_to_str(d.get("runtime_log_id")),
        "csv_log_id": oid_to_str(d.get("csv_log_id")),
        "network_metrics": d.get("network_metrics", {}),
    }


def simulation_to_mongo(dto: SimulationDto) -> Simulation:
    d = deepcopy(dto)
    sim: dict[str, Any] = {}

    if d.get("id"):
        sim["_id"] = str_to_oid(d["id"])

    sim["status"] = d.get("status", "")
    sim["individual_id"] = d.get("individual_id", "")
    sim["system_message"] = d.get("system_message", "")
    sim["random_seed"] = d.get("random_seed", 0)
    sim["start_time"] = d.get("start_time")
    sim["end_time"] = d.get("end_time")
    sim["parameters"] = d.get("parameters", {})
    sim["network_metrics"] = d.get("network_metrics", {})

    for k in ("experiment_id", "generation_id", "pos_file_id", "csc_file_id",
              "source_repository_id", "log_cooja_id", "runtime_log_id", "csv_log_id"):
        sim[k] = str_to_oid(d.get(k))

    return sim  # type: ignore[return-value]
