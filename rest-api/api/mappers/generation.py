from typing import Optional

from api.domain.generation import GenerationDto, IndividualDto
from api.mappers.helpers import oid_to_str, pop_id


def individual_from_mongo(
    doc: dict,
    simulations_by_individual: Optional[dict[str, list[str]]] = None,
) -> IndividualDto:
    id_str, d = pop_id(doc)
    ind_id = str(d.get("individual_id", ""))
    out: IndividualDto = {
        "id": id_str,
        "individual_id": ind_id,
        "chromosome": d.get("chromosome", {}),
        "objectives": d.get("objectives", []),
        "topology_picture_id": oid_to_str(d.get("topology_picture_id")),
    }
    if simulations_by_individual is not None:
        out["simulations_ids"] = simulations_by_individual.get(ind_id, [])
    return out


def generation_from_mongo(
    doc: dict,
    individuals: list[dict],
    simulations_by_individual: Optional[dict[str, list[str]]] = None,
) -> GenerationDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "experiment_id": oid_to_str(d.get("experiment_id")),
        "index": d.get("index", 0),
        "status": d.get("status", ""),
        "start_time": d.get("start_time"),
        "end_time": d.get("end_time"),
        "population": [
            individual_from_mongo(ind, simulations_by_individual)
            for ind in (individuals or [])
        ],
    }
