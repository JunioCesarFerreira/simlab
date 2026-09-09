from typing import TypedDict, NotRequired
from datetime import datetime
from bson import ObjectId


class Generation(TypedDict):
    _id: ObjectId
    experiment_id: ObjectId
    index: int
    status: str
    start_time: datetime
    end_time: datetime
    # Chromosome hashes kept by environmental selection at the end of this
    # generation — the population P_t the search actually carries forward, as
    # opposed to the offspring Q_t stored as this generation's individuals.
    # A survivor may come from an older generation, so these hashes must be
    # resolved against the whole experiment, not against this generation's
    # individual documents. Absent on generations written before the field
    # existed; consumers must fall back rather than assume an empty selection.
    survivors: NotRequired[list[str]]
