"""Placement: deciding where each simulation runs, and running it there.

``policy`` is the pure decision, ``inventory`` the snapshot it reads,
``dispatcher`` the single-threaded loop that applies it, and ``runner`` the
per-job execution the loop hands off to.
"""
from lib.scheduler.policy import (
    HostLoad,
    LeastLoadedHostPolicy,
    Policy,
    build_policy,
    select_least_loaded_host,
)
from lib.scheduler.inventory import (
    describe_loads,
    find_load,
    snapshot_loads,
    total_capacity,
)
from lib.scheduler.runner import (
    CollectionFailed,
    LostOwnership,
    RunFailed,
    RunReport,
    SimulationRunner,
    TerminationUnconfirmed,
)
from lib.scheduler.dispatcher import IDLE_TICK_SEC, Dispatcher

__all__ = [
    "HostLoad",
    "LeastLoadedHostPolicy",
    "Policy",
    "build_policy",
    "select_least_loaded_host",
    "describe_loads",
    "find_load",
    "snapshot_loads",
    "total_capacity",
    "CollectionFailed",
    "LostOwnership",
    "RunFailed",
    "RunReport",
    "SimulationRunner",
    "TerminationUnconfirmed",
    "Dispatcher",
    "IDLE_TICK_SEC",
]
