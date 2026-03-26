from typing import Optional
from bson import ObjectId


def oid_to_str(x) -> str:
    if x is None:
        return ""
    return str(x)


def str_to_oid(x: Optional[str]) -> Optional[ObjectId]:
    if not x:
        return None
    return ObjectId(x)


def pop_id(doc: dict) -> tuple[str, dict]:
    """Pops _id (or id) from a copy of doc and returns (id_str, rest)."""
    d = dict(doc)
    _id = d.pop("_id", None)
    id_fallback = d.pop("id", None) if _id is None else None
    return oid_to_str(_id if _id is not None else id_fallback), d


def dict_oid_to_str(d: dict) -> dict[str, str]:
    return {k: oid_to_str(v) for k, v in (d or {}).items()}


def dict_str_to_oid(d: dict) -> dict[str, ObjectId]:
    return {k: ObjectId(v) for k, v in (d or {}).items() if v}
