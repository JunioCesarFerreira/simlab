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

from lib.detached import RunState
from pylib.db.models.enums import EnumStatus
from pylib.db.models.infra import EnumResourceStatus, operator_configured
from pylib.timeutil import elapsed_since

log = logging.getLogger(__name__)

RECOVERY_MESSAGE = (
    "Recovered: the master-node restarted and this simulation was no longer running"
)

TERMINAL_STATUSES = (EnumStatus.DONE, EnumStatus.ERROR, EnumStatus.CANCELLED)

# A probe answers a ``RunState`` — alive, finished, what it left behind — or
# ``None`` when the container could not be reached. It was annotated as a
# boolean long after it stopped being one, which is the sort of drift that
# makes an annotation worse than none.
LivenessProbe = Callable[[dict, dict, Optional[int]], Optional[RunState]]
# Called with everything needed to resume following an adopted run.
AdoptionHandler = Callable[[dict, dict, dict, str, int], None]


@dataclass
class ReconcileReport:
    """What reconciliation decided, for the log and for tests."""
    released: list[str] = field(default_factory=list)
    adopted: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    #: Held, but a thread of this process is already watching them.
    followed: list[str] = field(default_factory=list)
    #: Reserved so recently that nothing can be concluded about them. Their
    #: TTL belongs to the stale-reservation sweep, not here.
    preparing: list[str] = field(default_factory=list)
    #: Held for something unresolved on the container. Only a pass that asks
    #: the container may let one of these go.
    held: list[str] = field(default_factory=list)
    failed_simulations: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.released) + len(self.adopted) + len(self.unreachable)

    def describe(self) -> str:
        parts = [
            f"{len(self.released)} released",
            f"{len(self.adopted)} adopted",
            f"{len(self.unreachable)} unreachable",
        ]
        if self.followed:
            parts.append(f"{len(self.followed)} already followed")
        if self.preparing:
            parts.append(f"{len(self.preparing)} still preparing")
        if self.held:
            parts.append(f"{len(self.held)} held")
        return ", ".join(parts)


def reconcile_occupied_resources(
    mongo: Any,
    probe: Optional[LivenessProbe] = None,
    on_adopt: Optional[AdoptionHandler] = None,
    *,
    is_followed: Optional[Callable[[Optional[str]], bool]] = None,
    reserved_grace_sec: float = 0.0,
    quiet: bool = False,
) -> ReconcileReport:
    """Decide what to do with every container still held that nobody watches.

    With no ``probe``, every non-terminal simulation is treated as gone, which
    is only correct where a run cannot outlive the master-node. Callers on the
    detached path must supply one.

    ``is_followed`` names the attempts a thread of this process is already
    watching, and they are skipped: their follower will collect the result and
    release the container, and reconciling underneath it would mean two
    followers for one attempt. Left out, nothing is skipped — which is right
    at startup, where no thread is following anything yet.

    That parameter is what lets this run *periodically* rather than only at
    startup. Running once meant a container whose host was unreachable at boot
    stayed ``busy`` for ever: the reconciliation never came back, the health
    cycle skips occupied containers, and the TTL sweep only covers reserved
    ones. ``quiet`` keeps the periodic pass from logging a headline when there
    is nothing to say.

    ``reserved_grace_sec`` is the second half of that. A container in
    ``reserved`` has had nothing started on it — there is no process to ask
    about and no evidence to weigh — so a periodic pass must not conclude
    anything from it: the attempt is very likely still assembling its inputs.
    That exact case freed a seconds-old reservation and failed its simulation
    (R04). Reservations younger than the grace are left to the stale-
    reservation sweep, which is the component that owns that timeout. Startup
    passes zero, because nothing there can belong to this process.
    """
    report = ReconcileReport()
    occupied = mongo.resource_repo.find_occupied()
    if is_followed is not None:
        unwatched = []
        for resource in occupied:
            attempt_id = resource.get("attempt_id")
            if is_followed(attempt_id):
                report.followed.append(str(resource.get("name", "?")))
                continue
            unwatched.append(resource)
        occupied = unwatched
    if reserved_grace_sec > 0:
        ready, preparing = _split_preparing(occupied, reserved_grace_sec)
        report.preparing.extend(preparing)
        occupied = ready
    if not occupied:
        return report

    announce = log.info if quiet else log.warning
    announce(
        "Found %d container(s) held with nobody following them; reconciling...",
        len(occupied),
    )
    for resource in occupied:
        try:
            _reconcile_one(
                mongo, resource, probe, on_adopt, report,
                reserved_grace_sec=reserved_grace_sec)
        except Exception:
            log.exception(
                "Failed to reconcile container %s", resource.get("name", "?"))
    announce("Reconciliation done: %s", report.describe())
    return report


def _split_preparing(
    occupied: list[dict], grace_sec: float
) -> tuple[list[dict], list[str]]:
    """Separate containers that are merely being prepared from the rest.

    ``reserved`` with a recent ``reserved_at`` and no PID: a placement in
    progress. Nothing about it can be concluded by probing, because nothing
    has been started.
    """
    ready: list[dict] = []
    preparing: list[str] = []
    for resource in occupied:
        if resource.get("status") != EnumResourceStatus.RESERVED.value:
            ready.append(resource)
            continue
        if resource.get("remote_pid"):
            # It did start, and the status write is simply behind.
            ready.append(resource)
            continue
        elapsed = elapsed_since(resource.get("reserved_at"))
        if elapsed is not None and 0 <= elapsed < grace_sec:
            preparing.append(str(resource.get("name", "?")))
            continue
        ready.append(resource)
    return ready, preparing


def _reconcile_one(
    mongo: Any,
    resource: dict,
    probe: Optional[LivenessProbe],
    on_adopt: Optional[AdoptionHandler],
    report: ReconcileReport,
    *,
    reserved_grace_sec: float = 0.0,
) -> None:
    name = str(resource.get("name", "?"))
    simulation = _load_simulation(mongo, resource.get("current_simulation_id"))
    # Read once: both the eligibility rule and the adoption need it.
    host = mongo.host_repo.get(resource.get("host_id"))

    # The simulation is over, or gone. That is *not* the same as "nothing is
    # running on the container": a quarantine whose write failed leaves a
    # terminal simulation and a worker that may still have a JVM on it, and
    # this branch used to free it with no probe and no guard — undoing the
    # protection exactly when the database came back.
    #
    # Two things now stop that. The write insists on the attempt that was
    # read, so a placement that landed in between is not erased; and the
    # repository refuses outright while `held_for` is set, which is how the
    # quarantine's decision outlives the process that made it.
    if simulation is None or simulation.get("status") in TERMINAL_STATUSES:
        if resource.get("held_for"):
            log.warning(
                "Container %s is held for %s; leaving it for the pass that "
                "asks the container", name,
                (resource.get("held_for") or {}).get("kind", "?"),
            )
            report.held.append(name)
            return
        if _release(
            mongo, resource, "recovered: no simulation was still running", host,
            expect_attempt_id=resource.get("attempt_id"),
        ):
            report.released.append(name)
        else:
            log.info(
                "Container %s was no longer held by attempt %s; leaving it",
                name, resource.get("attempt_id"),
            )
        return

    sim_oid = _as_oid(simulation["_id"])
    state = _ask_probe(probe, resource, simulation, name, host)

    if state is not None and (state.alive or state.collectable):
        # Two cases, one action. Still running: releasing would invite a
        # second simulation onto a busy container. Already finished, with its
        # testlog still there: the follower will see the exit code on its
        # first poll and collect the result. Both are *adopted* — the
        # difference is only how long the following lasts.
        #
        # The second case used to be a failure. A run that completed while
        # the master-node was down had its result thrown away and the next
        # job overwrote the evidence.
        log.warning(
            "Container %s %s simulation %s (pid %s); adopting it",
            name,
            "is still running" if state.alive else
            f"has a finished run (exit {state.exit_code}, "
            f"{state.testlog_bytes} bytes of log) for",
            sim_oid, simulation.get("remote_pid"),
        )
        report.adopted.append(name)
        if on_adopt is None:
            log.warning(
                "Nothing is following %s, so its result will not be collected; "
                "it resolves through SIM_TIMEOUT_SEC", sim_oid,
            )
            return
        attempt_id = resource.get("attempt_id") or simulation.get("attempt_id")
        # The same fallback the probe used: whichever document has the pid.
        pid = simulation.get("remote_pid") or resource.get("remote_pid")
        if host is None or not attempt_id or not pid:
            # Cannot resume without both. Still better to leave the run alone
            # than to free a container something is using.
            log.warning(
                "Cannot resume following %s: %s", sim_oid,
                "its host is gone" if host is None
                else "no attempt id was recorded" if not attempt_id
                else "no pid was recorded",
            )
            return
        try:
            on_adopt(simulation, host, resource, str(attempt_id), int(pid))
        except Exception:
            log.exception("Failed to resume following simulation %s", sim_oid)
        return

    if state is None:
        # Unreachable says nothing about the run. Touching either the
        # container or the simulation here is exactly what the specification
        # forbids.
        log.warning(
            "Container %s could not be reached; leaving simulation %s as it is",
            name, sim_oid,
        )
        report.unreachable.append(name)
        return

    # The container first, and only under the attempt that was read. Between
    # the query above and this write the dispatcher may have placed a new
    # simulation on it, or a runner thread may have promoted the reservation
    # and started a JVM — and freeing it then hands a container that is
    # running something to the next placement (R04). If the write is refused,
    # neither the container nor the simulation is touched.
    attempt_id = resource.get("attempt_id")
    if not _release(
        mongo, resource, "recovered on master-node restart", host,
        expect_attempt_id=attempt_id,
    ):
        log.info(
            "Container %s was no longer held by attempt %s; leaving it and "
            "its simulation alone", name, attempt_id,
        )
        return
    report.released.append(name)
    log.warning(
        "Simulation %s is no longer running on %s; marking it failed", sim_oid, name)
    if _fail_simulation(mongo, simulation, sim_oid, attempt_id):
        report.failed_simulations.append(str(sim_oid))


def _ask_probe(
    probe: Optional[LivenessProbe],
    resource: dict,
    simulation: dict,
    name: str,
    host: Optional[dict],
) -> Optional[RunState]:
    """Ask the container what became of the recorded run.

    ``None`` means "no answer" and the caller must leave everything alone. A
    ``RunState`` with nothing alive and nothing collectable is the definite
    "gone".

    No PID means it never got far enough to record one, so there is nothing
    to adopt and nothing to collect — a definite gone, not an unknown.
    """
    gone = RunState(alive=False, finished=False)
    # Two documents record the pid, and the resource is written first. A
    # failure between the two writes left the only copy on the resource, where
    # this never looked — so a live run was declared gone and its container
    # released (R05.3).
    pid = simulation.get("remote_pid") or resource.get("remote_pid")
    if not pid:
        log.info("Container %s holds a simulation with no recorded PID", name)
        return gone
    if not simulation.get("remote_pid"):
        log.warning(
            "Simulation %s has no pid of its own; using the one recorded on "
            "%s (%s)", simulation.get("_id"), name, pid,
        )
    if probe is None:
        return gone

    if host is None:
        log.warning("Host of container %s no longer exists", name)
        return gone
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


def _release(
    mongo: Any,
    resource: dict,
    message: str,
    host: Optional[dict],
    *,
    expect_attempt_id: Optional[str] = None,
) -> bool:
    """Hand the container back, by the same eligibility rule as everywhere.

    ``host`` may be ``None`` when its registration is gone; the rule then sees
    an empty host, which only means the host-level signals do not apply — the
    worker's own settings still do.

    ``expect_attempt_id`` makes the write conditional on the attempt that was
    read. Startup recovery can leave it out — the process that held the
    attempt died — but the periodic pass cannot, because a placement may have
    happened between its read and its write.
    """
    return bool(mongo.resource_repo.force_release(
        resource["_id"], message,
        container_state_counts=not operator_configured(host or {}, resource),
        # So a worker whose host's access changed while it was busy does not
        # come back as available on the strength of a probe that measured the
        # previous path.
        config_version=int((host or {}).get("config_version") or 0),
        expect_attempt_id=expect_attempt_id,
    ))


def _fail_simulation(
    mongo: Any,
    simulation: dict,
    sim_oid: ObjectId,
    attempt_id: Optional[str] = None,
) -> bool:
    """Fail a simulation whose process the probe found gone.

    Guarded by the attempt the *container* was holding, when there is one.
    Unconditional writes here are what let a recovery pass mark a failure
    against a placement that had already taken over; the release above has
    just confirmed that this attempt was still the holder, so the guard costs
    nothing and closes the window between the two writes.
    """
    failed = mongo.simulation_repo.mark_error(
        sim_oid, RECOVERY_MESSAGE, attempt_id=attempt_id)
    if not failed:
        log.info(
            "Simulation %s is owned by a newer attempt or already carries a "
            "result; not failing it", sim_oid,
        )
        return False
    generation_id = simulation.get("generation_id")
    if generation_id:
        _close_generation(mongo, generation_id)
    return True


def _close_generation(mongo: Any, generation_id: Any) -> None:
    try:
        mongo.generation_repo.close_if_finished(generation_id)
    except Exception:
        log.exception("Failed to close generation %s", generation_id)


def _as_oid(value: Any) -> ObjectId:
    return value if isinstance(value, ObjectId) else ObjectId(value)
