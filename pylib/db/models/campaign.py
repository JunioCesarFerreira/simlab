from typing import TypedDict, Optional
from datetime import datetime
from bson import ObjectId


class Campaign(TypedDict):
    name: str
    description: str
    created_time: datetime
    experiment_ids: list[ObjectId]
