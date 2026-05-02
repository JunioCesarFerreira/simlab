from typing import Optional
from datetime import datetime
from typing import TypedDict

from api.domain.experiment import ExperimentDto


class CampaignDto(TypedDict):
    id: Optional[str]
    name: str
    description: str
    created_time: Optional[datetime]
    experiment_ids: list[str]


class CampaignInfoDto(TypedDict):
    id: Optional[str]
    name: str
    description: str
    created_time: Optional[datetime]
    experiment_count: int


class CampaignFullDto(TypedDict):
    id: Optional[str]
    name: str
    description: str
    created_time: Optional[datetime]
    experiments: list[ExperimentDto]
