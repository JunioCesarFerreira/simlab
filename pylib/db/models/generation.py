from typing import TypedDict
from datetime import datetime
from bson import ObjectId


class Generation(TypedDict):
    _id: ObjectId
    experiment_id: ObjectId
    index: int
    status: str
    start_time: datetime
    end_time: datetime
