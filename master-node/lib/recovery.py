"""Reconciling the pool with reality after the master-node restarts.

Before this feature a restart cost nothing: workers held no state, so the
containers were usable again the moment the process came back. Reservations
changed that — a container claimed by a process that is gone stays claimed —
so the equivalent has to be done explicitly, or the pool shrinks by one
container per restart until nothing is left.

Why this probes instead of assuming
-----------------------------------
The first version of this module released everything and failed every
simulation it found, on the grounds that Cooja ran attached to its SSH channel
and therefore died with SIGHUP when the master-node did. That was true, and it
stopped being true the moment the run was detached: a restart now leaves live
JVMs behind, and releasing their containers would both throw away work and
risk placing a second simulation onto a container that is already busy.

So each held container is asked a question before anything is decided, and the
three possible answers are kept apart:

===========================  ==========================  ==========================
Simulation                   Probe                       Action
===========================  ==========================  ==========================
already finished or failed   —                           release the container
non-terminal, no PID         —                           fail it, release
non-terminal                 the process is alive        **adopt**: leave it be
non-terminal                 the process is gone         fail it, release
non-terminal                 container unreachable       leave it be, fail nothing
===========================  ==========================  ==========================

The last row is the specification's requirement that losing contact is not the
same as the run failing. The fourth is why the probe exists at all.

Adoption hands the run to ``on_adopt``, which resumes *following* it — so a
simulation that outlived a restart has its result collected when it finishes
rather than being abandoned to its timeout. Without that callback adoption
still means "do not disturb", which is correct but wastes the run.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from bson import ObjectId

from pylib.db.models.enums import EnumStatus
from pylib.db.models.infra import operator_configured

log = logging.getLogger(__name__)

RECOVERY_MESSAGE = (
    "Recovered: the master-node restarted and this simulation was no longer running"
)

TERMINAL_STATUSES = (EnumStatus.DONE, EnumStatus.ERROR, EnumStatus.CANCELLED)

# A probe answers True (running), False (gone) or None (could not be reached).
LivenessProbe = Callable[[dict, dict, Optional[int]], Optional[bool]]
# Called with everything needed to resume following an adopted run.
AdoptionHandler = Callable[[dict, dict, dict, str, int], None]


@dataclass
class ReconcileReport:
    """What reconciliation decided, for the log and for tests."""
    released: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    failed_simulations: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.released) + len(self.adopted) + len(self.unreachable)

    def describe(self) -> str:
        return (
            f"{len(self.released)} released, {len(self.adopted)} adopted, "
            f"{len(self.unreachable)} unreachable"
        )


def reconcile_occupied_resources(
    mongo: Any,
    probe: Optional[LivenessProbe] = None,
    on_adopt: Optional[AdoptionHandler] = None,
) -> ReconcileReport:
    """Decide what to do with every container still held from a previous run.

    With no ``probe``, every non-terminal simulation is treated as gone. That
    is only correct where the run cannot have outlived the master-node — the
    legacy execution path — and callers on the detached path must supply one.
    """
    report = ReconcileReport()
    occupied = mongo.resource_repo.find_occupied()
    if not occupied:
        return report

    log.warning(
        "Found %d container(s) still held from a previous run; reconciling...",
        len(occupied),
    )
    for resource in occupied:
        try:
            _reconcile_one(mongo, resource, probe, on_adopt, report)
        except Exception:
            log.exception(
                "Failed to reconcile container %s", resource.get("name", "?"))
    log.warning("Reconciliation done: %s", report.describe())
    return report


def _reconcile_one(
    mongo: Any,
    resource: dict,
    probe: Optional[LivenessProbe],
    on_adopt: Optional[AdoptionHandler],
    report: ReconcileReport,
) -> None:
    name = str(resource.get("name", "?"))
    simulation = _load_simulation(mongo, resource.get("current_simulation_id"))
    # Read once: both the eligibility rule and the adoption need it.
    host = mongo.host_repo.get(resource.get("host_id"))

    # Nothing is running: the result was already recorded and only the release
    # was lost. Freeing the container cannot destroy anything.
    if simulation is None or simulation.get("status") in TERMINAL_STATUSES:
        if _release(
            mongo, resource, "recovered: no simulation was still running", host
        ):
            report.released.append(name)
        return

    sim_oid = _as_oid(simulation["_id"])
    alive = _ask_probe(probe, resource, simulation, name, host)

    if alive is True:
        # Still going. Releasing would invite a second simulation onto a busy
        # container, so the run is adopted and followed from where it is.
        log.warning(
            "Container %s is still running simulation %s (pid %s); adopting it",
            name, sim_oid, simulation.get("remote_pid"),
        )
        report.adopted.append(name)
        if on_adopt is None:
            log.warning(
                "Nothing is following %s, so its result will not be collected; "
                "it resolves through SIM_TIMEOUT_SEC", sim_oid,
            )
            return
        attempt_id = resource.get("attempt_id") or simulation.get("attempt_id")
        if host is None or not attempt_id:
            # Cannot resume without both. Still better to leave the run alone
            # than to free a container something is using.
            log.warning(
                "Cannot resume following %s: %s", sim_oid,
                "its host is gone" if host is None else "no attempt id was recorded",
            )
            return
        try:
            on_adopt(simulation, host, resource, str(attempt_id), int(simulation["remote_pid"]))
        except Exception:
            log.exception("Failed to resume following simulation %s", sim_oid)
        return

    if alive is None:
        # Unreachable says nothing about the run. Touching either the
        # container or the simulation here is exactly what the specification
        # forbids.
        log.warning(
            "Container %s could not be reached; leaving simulation %s as it is",
            name, sim_oid,
        )
        report.unreachable.append(name)
        return

    log.warning(
        "Simulation %s is no longer running on %s; marking it failed", sim_oid, name)
    _fail_simulation(mongo, simulation, sim_oid)
    report.failed_simulations.append(str(sim_oid))
    if _release(mongo, resource, "recovered on master-node restart", host):
        report.released.append(name)


def _ask_probe(
    probe: Optional[LivenessProbe],
    resource: dict,
    simulation: dict,
    name: str,
    host: Optional[dict],
) -> Optional[bool]:
    """Ask whether the recorded run is still going.

    No PID means it never got far enough to record one, so there is nothing
    to adopt — that is a definite "gone", not an unknown.
    """
    pid = simulation.get("remote_pid")
    if not pid:
        log.info("Container %s holds a simulation with no recorded PID", name)
        return False
    if probe is None:
        return False

    if host is None:
        log.warning("Host of container %s no longer exists", name)
        return False
    try:
        return probe(host, resource, int(pid))
    except Exception as error:
        # Cannot tell. Reported as unknown rather than dead, so an error in
        # the probe cannot destroy a running simulation.
        log.warning("Could not probe container %s: %s", name, error)
        return None


def _load_simulation(mongo: Any, simulation_id: Any) -> Optional[dict]:
    if not simulation_id:
        return None
    return mongo.simulation_repo.get(str(simulation_id))


def _release(mongo: Any, resource: dict, message: str, host: Optional[dict]) -> bool:
    """Hand the container back, by the same eligibility rule as everywhere.

    ``host`` may be ``None`` when its registration is gone; the rule then sees
    an empty host, which only means the host-level signals do not apply — the
    worker's own settings still do.
    """
    return bool(mongo.resource_repo.force_release(
        resource["_id"], message,
        container_state_counts=not operator_configured(host or {}, resource),
    ))


def _fail_simulation(mongo: Any, simulation: dict, sim_oid: ObjectId) -> None:
    # Unconditional: the probe said the process is gone, so no attempt owns
    # this simulation any more.
    mongo.simulation_repo.mark_error(sim_oid, RECOVERY_MESSAGE, attempt_id=None)
    generation_id = simulation.get("generation_id")
    if generation_id:
        _close_generation(mongo, generation_id)


def _close_generation(mongo: Any, generation_id: Any) -> None:
    try:
        mongo.generation_repo.close_if_finished(generation_id)
    except Exception:
        log.exception("Failed to close generation %s", generation_id)


def _as_oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)
