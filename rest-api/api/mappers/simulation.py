from copy import deepcopy
from typing import Any, Optional

from pylib.db.models import Simulation
from api.domain.simulation import PlacementDto, SimulationDto
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
        "dodag": d.get("dodag"),
        "placement": _placement_from_mongo(d),
    }


def _placement_from_mongo(d: dict) -> Optional[PlacementDto]:
    """Where this simulation ran, or ``None`` if it never got that far.

    The fields are flat on the document — the master-node writes them in one
    update at reservation time — and nested here, because they are one fact
    about one run and a client reading them wants them together.

    ``None`` rather than an object full of empties: a simulation that was
    never placed, or one from before the master-node recorded this, must be
    distinguishable from one that ran somewhere unknown.
    """
    host_name = str(d.get("assigned_host_name") or "")
    resource_name = str(d.get("assigned_resource_name") or "")
    if not (host_name or resource_name or d.get("assigned_resource_id")):
        return None
    return {
        "host_id": oid_to_str(d.get("assigned_host_id")) or None,
        "host_name": host_name,
        "resource_id": oid_to_str(d.get("assigned_resource_id")) or None,
        "resource_name": resource_name,
        "container_id": d.get("assigned_container_id") or None,
        "attempt_id": d.get("attempt_id") or None,
        "assigned_at": d.get("assigned_at"),
        "remote_pid": d.get("remote_pid"),
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
    if d.get("dodag") is not None:
        sim["dodag"] = d.get("dodag")

    for k in ("experiment_id", "generation_id", "pos_file_id", "csc_file_id",
              "source_repository_id", "log_cooja_id", "runtime_log_id", "csv_log_id"):
        sim[k] = str_to_oid(d.get(k))

    return sim  # type: ignore[return-value]
