from copy import deepcopy
from typing import Any

from api.domain.source import SourceRepositoryDto
from api.mappers.helpers import oid_to_str, str_to_oid, pop_id


def source_repository_from_mongo(doc: dict) -> SourceRepositoryDto:
    id_str, d = pop_id(doc)
    source_files = [
        {"id": oid_to_str(sf.get("id") or sf.get("_id")), "file_name": sf.get("file_name", "")}
        for sf in (d.get("source_files") or [])
    ]
    return {
        "id": id_str or d.get("id", ""),
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "source_files": source_files,
    }


def source_repository_to_mongo(dto: SourceRepositoryDto) -> dict:
    d = deepcopy(dto)
    out: dict[str, Any] = {}

    if d.get("id"):
        out["_id"] = str_to_oid(d["id"])

    out["name"] = d.get("name", "")
    out["description"] = d.get("description", "")
    out["source_files"] = [
        {"id": f["id"], "file_name": f.get("file_name", "")}
        for f in (d.get("source_files") or [])
    ]
    return out
