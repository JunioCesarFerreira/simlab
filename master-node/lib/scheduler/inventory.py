"""Reading the inventory the placement decision is made on.

One aggregation per decision, not one query per host: the decision is taken
for every simulation, and a per-host fan-out would turn a single choice into N
round trips. The same document feeds the GUI's overview, so what an operator
sees is what the scheduler used.
"""
import logging
from typing import Any, Optional

from lib.scheduler.policy import HostLoad

log = logging.getLogger(__name__)


def snapshot_loads(host_repo: Any) -> list[HostLoad]:
    """Current per-host capacity and occupancy."""
    loads: list[HostLoad] = []
    for document in host_repo.inventory_snapshot():
        loads.append(HostLoad(
            host_id=document["_id"],
            name=str(document.get("name") or "?"),
            free=int(document.get("free") or 0),
            active=int(document.get("active") or 0),
            schedulable=bool(document.get("schedulable")),
            # Absent on a pool nothing has run on yet, and on a host whose
            # free containers have no measurement. `None` is the honest value
            # and the policies read it as unknown.
            fastest_free_sec=(
                float(document["fastest_free_sec"])
                if document.get("fastest_free_sec") is not None else None),
        ))
    return loads


def describe_loads(loads: list[HostLoad]) -> str:
    """A one-line summary for the log, so a placement can be explained later."""
    if not loads:
        return "no hosts registered"
    return " ".join(
        f"{load.name}[{load.active}a/{load.free}f"
        f"{'' if load.fastest_free_sec is None else f' {load.fastest_free_sec:.0f}s'}"
        f"{'' if load.schedulable else ' off'}]"
        for load in loads
    )


def total_capacity(loads: list[HostLoad]) -> tuple[int, int]:
    """``(free, active)`` across every schedulable host."""
    free = sum(load.free for load in loads if load.schedulable)
    active = sum(load.active for load in loads if load.schedulable)
    return free, active


def find_load(loads: list[HostLoad], host_id: Any) -> Optional[HostLoad]:
    return next((load for load in loads if load.host_id == host_id), None)
