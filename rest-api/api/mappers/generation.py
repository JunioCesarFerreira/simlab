from typing import Optional

from api.domain.generation import (
    CompletenessDto,
    GenerationDto,
    IndividualDto,
)
from api.mappers.helpers import oid_to_str, pop_id

#: The fields of a stored completeness record that are published. A fixed list
#: rather than the stored document, so a field added to the engine's record
#: cannot leak into the API contract unnoticed.
_COMPLETENESS_FIELDS = (
    "state", "expected", "observed", "failed", "pending", "absent",
    "expected_seeds", "observed_seeds", "missing_seeds", "failed_seeds",
    "per_metric", "attempts", "objective_used", "policy",
    "reused", "source", "reason",
)


def _completeness_from_mongo(stored: Optional[dict]) -> Optional[CompletenessDto]:
    """The evidence behind one objective vector, or ``None``.

    ``None`` means *not recorded*, which is what every individual evaluated
    before this existed will say. Returning an empty record instead would
    read as "nothing was expected and nothing observed", and a reader cannot
    distinguish a measurement from a missing one.
    """
    if not stored:
        return None
    out: CompletenessDto = {}
    for field in _COMPLETENESS_FIELDS:
        if field in stored:
            out[field] = stored[field]          # type: ignore[literal-required]
    return out or None


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
        # Published next to the objectives it belongs to. Read on its own, an
        # objective cannot say how many replicas it is a mean over.
        "completeness": _completeness_from_mongo(d.get("completeness")),
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
