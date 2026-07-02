from typing import Optional, Any
from pydantic import BaseModel


class ProblemInfoDto(BaseModel):
    id: str
    name: str
    created_time: str
    updated_time: str
    has_background: bool


class ProblemDto(BaseModel):
    id: str
    name: str
    created_time: str
    updated_time: str
    draft: dict[str, Any]
    background_image_id: Optional[str]
    image_world_bounds: Optional[list[float]]


class ProblemCreateDto(BaseModel):
    name: str
    draft: dict[str, Any]
    image_world_bounds: Optional[list[float]] = None


class ProblemUpdateDto(BaseModel):
    name: Optional[str] = None
    draft: Optional[dict[str, Any]] = None
    image_world_bounds: Optional[list[float]] = None
