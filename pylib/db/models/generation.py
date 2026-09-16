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
    # individual documents. Recorded in selection order and WITH repeats: a
    # child that reproduces a surviving parent exactly can occupy two slots, and
    # a resume rebuilds the population from this list. Absent on generations
    # written before the field existed; consumers must fall back rather than
    # assume an empty selection.
    survivors: NotRequired[list[str]]
    # Snapshot of the engine's random generator taken when this generation was
    # enqueued — i.e. before any draw belonging to it. Every library seed
    # (DEAP, pymoo) is derived from that one generator. Stateful backends also
    # require selection_state to reproduce the uninterrupted trajectory.
    rng_state: NotRequired[dict]
    # Versioned BSON-safe backend state at enqueue, before this generation's
    # environmental selection. Absent in legacy checkpoints/stateless backends.
    selection_state: NotRequired[dict]
