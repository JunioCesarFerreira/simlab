"""Placement policy: which host gets the next simulation.

The unit of balancing is the **host**, and the metric is the **absolute count
of active placements** — not the utilization ratio. The difference is the whole
point of the feature, and it is easiest to see in the case the specification
gives: hosts with 8, 4 and 2 free containers, six simulations to run. By
absolute count the answer is 2/2/2. By utilization it would be 3/2/1, and the
machine with the most containers would quietly absorb most of every run.

A pure function over a snapshot, so every rule here is testable without a
database, a network or a container.
"""
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostLoad:
    """One host's capacity and occupancy, as the inventory snapshot reports it.

    ``free`` and ``active`` describe the containers; ``schedulable`` describes
    the host. They are kept apart so an unreachable host still reports the
    capacity it has — folding them together would make the GUI under-report
    containers that are merely out of contact.
    """
    host_id: Any
    name: str
    free: int
    active: int
    schedulable: bool
    #: Measured occupancy of the free container this host would hand over —
    #: ``None`` when none of them has been measured yet. Read as *unknown*,
    #: never as *infinitely slow*: a pool that has just been registered has
    #: no measurements at all, and treating that as the worst case would make
    #: a speed-aware policy refuse to use it.
    fastest_free_sec: Optional[float] = None

    @property
    def eligible(self) -> bool:
        return self.schedulable and self.free > 0


class Policy(Protocol):
    """Chooses a host. Implementations must be deterministic."""

    name: str
    #: Whether the dispatcher should also order *within* the chosen host by
    #: measured speed. Part of the policy because the two decisions are one
    #: decision — picking the host that finishes soonest and then the slowest
    #: container on it would undo half the choice.
    prefers_fastest: bool

    def select(self, loads: list[HostLoad], rr_cursor: int) -> Optional[HostLoad]: ...


def select_least_loaded_host(
    loads: Iterable[HostLoad], rr_cursor: int
) -> Optional[HostLoad]:
    """The host with the fewest active placements, ties broken by rotation.

    Returns ``None`` when nothing can take work, which is the dispatcher's cue
    to wait rather than to fail the simulation.

    Rotation on ties rather than, say, lowest id: with every host idle, every
    decision would otherwise pick the same one until its count rose, which
    turns a tie into a queue behind one machine.
    """
    candidates = [load for load in loads if load.eligible]
    if not candidates:
        return None

    fewest = min(load.active for load in candidates)
    tied = sorted(
        (load for load in candidates if load.active == fewest),
        # By id, not by name: a renamed host must not change where work goes.
        key=lambda load: str(load.host_id),
    )
    return tied[rr_cursor % len(tied)]


#: How much slower a host may be and still count as equivalent. Below this,
#: the measured difference is within the noise of a per-worker mean and
#: spreading the work is worth more than chasing it; above it, the slower host
#: delays the batch. 10% sits above the residual noise of a dozen samples
#: (a few per cent) and well below a real difference between machines
#: (1.22–1.25x measured on a three-host pool).
SPEED_TOLERANCE = 0.10


def select_fastest_then_least_loaded(
    loads: Iterable[HostLoad], rr_cursor: int
) -> Optional[HostLoad]:
    """Least-loaded, among the hosts that are not meaningfully slower.

    The rank restricts the candidates; the existing rule then decides inside
    that set, unchanged. Breaking the *tie* instead — the obvious design —
    was tried and does nothing: after one placement the fast host is strictly
    more loaded, so the count sends the next job to a slow host before any tie
    can occur. Four jobs on 8+2+2 free containers went 2/1/1 either way.

    **Under saturation this changes nothing.** With every container busy the
    dispatcher waits, and when one is released its host is the only one with
    ``free > 0`` — there is no choice to make. The throughput-proportional
    split that the absolute count produces is therefore untouched: measured on
    a three-host pool, 71.7/14.6/13.7 against a prediction of 71.5/14.4/14.1.

    **Under slack it is the whole point.** The tail of a generation, a
    population smaller than the pool, a run starting from idle: there the
    count spreads jobs one per host and the batch waits for the slowest. A
    generation closes only when its last simulation does, so that wait is paid
    once per generation.

    What this trades away, stated plainly: a homogeneous pool is unaffected —
    every host falls inside the tolerance, so the candidate set is everything
    and the behaviour is identical, including the 8/4/2 → 2/2/2 example. A
    pool with a genuinely slower host will leave it **idle** while a faster
    one has room. That is correct for finishing a batch soonest and wrong for
    spreading risk across machines, and it is why this is not the default.

    It cannot starve a host either way: a fast host with no free container is
    not eligible, so the slow ones become the only candidates the moment the
    fast ones fill — which, with any real queue, is immediately.
    """
    candidates = [load for load in loads if load.eligible]
    if not candidates:
        return None

    # A host nothing has measured cannot be ranked, and under slack sampling
    # it is free. Optimistic and self-limiting: one run gives it a number.
    unmeasured = [c for c in candidates if c.fastest_free_sec is None]
    if unmeasured:
        return select_least_loaded_host(unmeasured, rr_cursor)

    best = min(float(c.fastest_free_sec) for c in candidates)
    close = [c for c in candidates
             if float(c.fastest_free_sec) <= best * (1 + SPEED_TOLERANCE)]
    # The original rule, applied to the shortlist — so everything it
    # guarantees inside a set of comparable hosts still holds.
    return select_least_loaded_host(close, rr_cursor)


class LeastLoadedHostPolicy:
    """Least-loaded host by absolute count, ties by rotation.

    Deliberately behind an interface: capacity-weighted or CPU-aware variants
    are foreseen, and the specification asks for the initial policy to be
    replaceable without touching the dispatcher.
    """

    name = "least_loaded_host"
    prefers_fastest = False

    def select(self, loads: list[HostLoad], rr_cursor: int) -> Optional[HostLoad]:
        return select_least_loaded_host(loads, rr_cursor)


class FastestThenLeastLoadedPolicy:
    """Least-loaded among the hosts that are not meaningfully slower.

    Identical to :class:`LeastLoadedHostPolicy` under saturation and on a
    homogeneous pool. Different exactly where a batch loses time: slack on a
    pool whose hosts differ in speed.
    """

    name = "fastest_then_least_loaded"
    prefers_fastest = True

    def select(self, loads: list[HostLoad], rr_cursor: int) -> Optional[HostLoad]:
        return select_fastest_then_least_loaded(loads, rr_cursor)


_POLICIES: dict[str, type] = {
    LeastLoadedHostPolicy.name: LeastLoadedHostPolicy,
    FastestThenLeastLoadedPolicy.name: FastestThenLeastLoadedPolicy,
}


def build_policy(name: Optional[str] = None) -> Policy:
    """Resolve a policy by name, falling back to the default with a warning."""
    key = (name or LeastLoadedHostPolicy.name).strip()
    policy_type = _POLICIES.get(key)
    if policy_type is None:
        log.warning(
            "Unknown scheduling policy %r; using %s. Known: %s",
            key, LeastLoadedHostPolicy.name, ", ".join(sorted(_POLICIES)),
        )
        policy_type = LeastLoadedHostPolicy
    return policy_type()
