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

    @property
    def eligible(self) -> bool:
        return self.schedulable and self.free > 0


class Policy(Protocol):
    """Chooses a host. Implementations must be deterministic."""

    name: str

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


class LeastLoadedHostPolicy:
    """Least-loaded host by absolute count. The only policy for now.

    Deliberately behind an interface: capacity-weighted or CPU-aware variants
    are foreseen, and the specification asks for the initial policy to be
    replaceable without touching the dispatcher.
    """

    name = "least_loaded_host"

    def select(self, loads: list[HostLoad], rr_cursor: int) -> Optional[HostLoad]:
        return select_least_loaded_host(loads, rr_cursor)


_POLICIES: dict[str, type] = {
    LeastLoadedHostPolicy.name: LeastLoadedHostPolicy,
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
