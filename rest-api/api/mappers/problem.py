from typing import Any

from api.domain.problem import ProblemInfoDto, ProblemDto


def _fmt_dt(dt) -> str:
    if dt is None:
        return ""
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def problem_to_info_dto(doc: dict[str, Any]) -> ProblemInfoDto:
    return ProblemInfoDto(
        id=str(doc["_id"]),
        name=doc.get("name", ""),
        created_time=_fmt_dt(doc.get("created_time")),
        updated_time=_fmt_dt(doc.get("updated_time")),
        has_background=bool(doc.get("background_image_id")),
    )


def problem_to_dto(doc: dict[str, Any]) -> ProblemDto:
    bg_id = doc.get("background_image_id")
    return ProblemDto(
        id=str(doc["_id"]),
        name=doc.get("name", ""),
        created_time=_fmt_dt(doc.get("created_time")),
        updated_time=_fmt_dt(doc.get("updated_time")),
        draft=doc.get("draft", {}),
        background_image_id=str(bg_id) if bg_id else None,
        image_world_bounds=doc.get("image_world_bounds"),
    )
