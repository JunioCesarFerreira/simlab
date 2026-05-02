from copy import deepcopy
from typing import Any
from bson import ObjectId

from pylib.db.models import Campaign
from api.domain.campaign import CampaignDto, CampaignInfoDto, CampaignFullDto
from api.domain.experiment import ExperimentDto
from api.mappers.helpers import oid_to_str, str_to_oid, pop_id


def campaign_from_mongo(doc: dict) -> CampaignDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "created_time": d.get("created_time"),
        "experiment_ids": [oid_to_str(oid) for oid in d.get("experiment_ids", [])],
    }


def campaign_info_from_mongo(doc: dict) -> CampaignInfoDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "created_time": d.get("created_time"),
        "experiment_count": len(d.get("experiment_ids", [])),
    }


def campaign_full_from_mongo(doc: dict, experiments: list[ExperimentDto]) -> CampaignFullDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "created_time": d.get("created_time"),
        "experiments": experiments,
    }


def campaign_to_mongo(dto: CampaignDto) -> Campaign:
    d = deepcopy(dto)
    doc: dict[str, Any] = {}

    if d.get("id"):
        doc["_id"] = str_to_oid(d["id"])

    doc["name"] = d.get("name", "")
    doc["description"] = d.get("description", "") or ""
    doc["created_time"] = d.get("created_time")
    doc["experiment_ids"] = [
        ObjectId(eid) for eid in (d.get("experiment_ids") or []) if eid
    ]

    return doc  # type: ignore[return-value]
