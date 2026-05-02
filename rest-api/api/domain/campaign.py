from typing import Optional, NotRequired
from datetime import datetime
from typing import TypedDict

from api.domain.experiment import ExperimentDto


class CampaignDto(TypedDict):
    id: NotRequired[Optional[str]]
    name: str
    description: NotRequired[str]
    created_time: NotRequired[Optional[datetime]]
    experiment_ids: NotRequired[list[str]]


class CampaignInfoDto(TypedDict):
    id: NotRequired[Optional[str]]
    name: str
    description: str
    created_time: NotRequired[Optional[datetime]]
    experiment_count: int


class CampaignFullDto(TypedDict):
    id: NotRequired[Optional[str]]
    name: str
    description: str
    created_time: NotRequired[Optional[datetime]]
    experiments: list[ExperimentDto]
