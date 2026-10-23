"""Placement: deciding where each simulation runs, and running it there.

``policy`` is the pure decision, ``completion`` the plan one policy makes of
when the waiting work would finish, ``inventory`` the snapshot it reads,
``dispatcher`` the single-threaded loop that applies it, and ``runner`` the
per-job execution the loop hands off to.
"""
from lib.scheduler.policy import (
    EarliestCompletionPolicy,
    HostLoad,
    LeastLoadedHostPolicy,
    Policy,
    build_policy,
    select_least_loaded_host,
)
from lib.scheduler.completion import CompletionPlan, plan_completion
from lib.scheduler.inventory import (
    apply_machine_limits,
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
    "CompletionPlan",
    "EarliestCompletionPolicy",
    "apply_machine_limits",
    "plan_completion",
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
