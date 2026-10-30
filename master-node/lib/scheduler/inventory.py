"""Reading the inventory the placement decision is made on.

One aggregation per decision, not one query per host: the decision is taken
for every simulation, and a per-host fan-out would turn a single choice into N
round trips. The same document feeds the GUI's overview, so what an operator
sees is what the scheduler used.
"""
import logging
from dataclasses import replace
from datetime import datetime
from typing import Any, Optional

from lib.scheduler.policy import HostLoad
from pylib.db.models.infra import machine_of
from pylib.timeutil import elapsed_since

log = logging.getLogger(__name__)


def _seconds(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def snapshot_loads(host_repo: Any, *, now: Optional[datetime] = None) -> list[HostLoad]:
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
            machine=machine_of(document),
            free_durations=tuple(
                _seconds(d) for d in document.get("free_durations") or ()),
            occupied=tuple(
                (_seconds(w.get("mean_duration_sec")),
                 elapsed_since(w.get("started_at"), now=now))
                for w in document.get("occupied_workers") or ()),
        ))
    return loads


def apply_machine_limits(loads: list[HostLoad], limits: dict[str, int]) -> list[HostLoad]:
    """Mark the hosts of a machine at its limit as unable to take work.

    Counted over every host of the machine, schedulable or not: a host being
    drained still runs its simulations on the same CPUs. Applied before any
    policy decides, so the limit holds whichever one is configured.
    """
    if not limits:
        return loads
    occupied: dict[str, int] = {}
    for load in loads:
        occupied[load.machine] = occupied.get(load.machine, 0) + load.active
    return [
        replace(load, capped=True)
        if load.machine in limits and occupied[load.machine] >= int(limits[load.machine])
        else load
        for load in loads
    ]


def describe_loads(loads: list[HostLoad]) -> str:
    """A one-line summary for the log, so a placement can be explained later."""
    if not loads:
        return "no hosts registered"
    return " ".join(
        f"{load.name}[{load.active}a/{load.free}f"
        f"{'' if load.fastest_free_sec is None else f' {load.fastest_free_sec:.0f}s'}"
        f"{'' if load.schedulable else ' off'}"
        f"{' cap' if load.capped else ''}]"
        for load in loads
    )


def total_capacity(loads: list[HostLoad]) -> tuple[int, int]:
    """``(free, active)`` across every schedulable host."""
    free = sum(load.free for load in loads if load.schedulable)
    active = sum(load.active for load in loads if load.schedulable)
    return free, active


def find_load(loads: list[HostLoad], host_id: Any) -> Optional[HostLoad]:
    return next((load for load in loads if load.host_id == host_id), None)
