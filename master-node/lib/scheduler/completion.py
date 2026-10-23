"""Planning when the waiting work would finish, to decide whether to wait.

A generation closes when its last simulation does, so what a batch costs is
its **makespan**, and the decision that sets it is not "which free worker"
but "a free worker *now*, or a faster one when it frees up". The policies
before this one could only answer the first question.

The plan is list scheduling with earliest finish time, on paper: every
waiting job, in turn, goes to the worker that would finish it first — a free
one now, or a busy one once its current run ends — using each worker's
measured held time. For identical jobs on workers of different speeds this
greedy is optimal for the makespan, and the jobs in one queue are as close to
identical as the scheduler can know: it has no estimate per job, only per
worker. The plan is then used for one thing only: if it gives a job to a
worker that is free **now**, the head of the queue goes there; if it gives
none, the head waits.

Two inputs need care:

* **A busy worker's remaining time** is its expected held time minus how long
  it has been running. A run past ``OVERDUE_FACTOR`` times its expectation is
  not counted on at all: the estimate has failed for it, and waiting on a
  worker that may be stuck would stall the queue behind it.
* **A machine's limit** caps how many of its workers can run at once. Its
  occupied workers count first, because they hold the capacity already; only
  the fastest of its free workers fit in what is left.

A worker never measured is sampled before anything is planned, as the other
speed-aware policy does: the plan cannot reason about a duration it does not
have, and one run gives it one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lib.scheduler.policy import HostLoad, select_least_loaded_host

#: A run this many times past its expected held time is no longer counted on
#: to free its worker soon.
OVERDUE_FACTOR = 2.0


@dataclass(frozen=True)
class Slot:
    """One worker in the plan: when it can start a job, and how long one takes."""
    load: HostLoad
    duration: float
    ready_in: float
    free_now: bool


@dataclass(frozen=True)
class CompletionPlan:
    """What to do with the job at the head of the queue."""
    #: Where to place it now. ``None`` with ``deferred`` false means nothing
    #: can take it at all.
    host: Optional[HostLoad]
    #: A busy, faster worker finishes it sooner than any free one: wait.
    deferred: bool
    reason: str
    #: When the plan expects the waiting work to be done, in seconds from now.
    makespan_sec: Optional[float] = None


def _slots(loads: list[HostLoad], limits: dict[str, int]) -> list[Slot]:
    by_machine: dict[str, list[HostLoad]] = {}
    for load in loads:
        if load.schedulable:
            by_machine.setdefault(load.machine or str(load.host_id), []).append(load)

    slots: list[Slot] = []
    for machine, hosts in sorted(by_machine.items()):
        for load in hosts:
            for expected, elapsed in load.occupied:
                if expected is None or expected <= 0:
                    continue
                running = max(0.0, elapsed or 0.0)
                if running > OVERDUE_FACTOR * expected:
                    continue
                slots.append(Slot(load, float(expected),
                                  max(0.0, expected - running), False))

        free = sorted(
            ((load, float(d)) for load in hosts if load.eligible
             for d in load.free_durations[:load.free] if d is not None),
            key=lambda pair: (pair[1], str(pair[0].host_id)))
        limit = limits.get(machine)
        if limit is not None:
            occupied = sum(len(load.occupied) for load in hosts)
            free = free[:max(0, int(limit) - occupied)]
        slots.extend(Slot(load, duration, 0.0, True) for load, duration in free)
    return slots


def plan_completion(
    loads: list[HostLoad],
    waiting: int,
    *,
    limits: Optional[dict[str, int]] = None,
    rr_cursor: int = 0,
) -> CompletionPlan:
    """Decide whether the head of a queue of ``waiting`` jobs goes now, and where.

    ``loads`` should already have the machine limits applied (``capped``);
    ``limits`` is needed again to know how many of a machine's free workers
    fit under its limit.
    """
    limits = limits or {}
    eligible = [load for load in loads if load.eligible]
    if not eligible:
        return CompletionPlan(None, False, "no free worker")

    # Nothing to plan with: a worker with no measurement, or a snapshot that
    # does not carry the per-worker detail. Sampled, like the other
    # speed-aware policy does.
    unmeasured = [load for load in eligible
                  if not load.free_durations
                  or any(d is None for d in load.free_durations[:load.free])]
    if unmeasured:
        return CompletionPlan(
            select_least_loaded_host(unmeasured, rr_cursor), False,
            "sampling a worker that has no measurement yet")

    slots = _slots(loads, limits)
    if not any(slot.free_now for slot in slots):
        # Every free worker is on a machine at its limit.
        return CompletionPlan(None, False, "every free worker is on a machine at its limit")

    ready = [slot.ready_in for slot in slots]
    used_now: set[int] = set()
    first_finish: Optional[float] = None
    makespan = 0.0
    for _ in range(max(1, int(waiting))):
        best = min(
            range(len(slots)),
            key=lambda i: (ready[i] + slots[i].duration,
                           not (slots[i].free_now and ready[i] == 0.0),
                           slots[i].duration, i))
        if slots[best].free_now and ready[best] == 0.0:
            used_now.add(best)
        ready[best] += slots[best].duration
        makespan = max(makespan, ready[best])
        if first_finish is None:
            first_finish = ready[best]

    if used_now:
        chosen = min(used_now, key=lambda i: (slots[i].duration, str(slots[i].load.host_id)))
        return CompletionPlan(
            slots[chosen].load, False,
            f"finishes in ~{slots[chosen].duration:.0f} s on "
            f"{slots[chosen].load.name}", makespan)

    fastest_free = min(slot.duration for slot in slots if slot.free_now)
    return CompletionPlan(
        None, True,
        f"a busy, faster worker finishes it in ~{first_finish:.0f} s; the fastest "
        f"free one would take ~{fastest_free:.0f} s", makespan)
