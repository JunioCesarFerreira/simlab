"""The health cycle: measuring the pool instead of assuming it.

Availability is decided at two levels, and a container needs both:

1. **Docker** — does the container exist, and is it running? Answered by
   ``lib.discovery``.
2. **SSH probe** — does it answer, and is there a usable Cooja build in it?
   Answered here.

Keeping them apart is what lets the GUI say *which* level failed. "The
container is stopped" and "the container is up but has no Cooja" need
different fixes, and a single boolean would hide the difference.

Rules this cycle obeys
----------------------
* **It never touches a container that is holding a simulation.** Enforced in
  the repository's query filters rather than here, so no code path can forget
  it. Losing contact with a host says nothing about the run on it.
* **A host goes offline only after repeated failure**, and comes back on the
  first success. Without that threshold one dropped packet would remove a
  host's whole capacity from the pool; without the asymmetry, recovery would
  be needlessly slow.
* **Docker being unavailable is not the host failing.** A host that already
  has resources is still checked over SSH, which is all the master-node ever
  did before this feature.
* **A container that answered recently is not asked again.** A probe is a
  two-hop SSH connection, and repeating it every thirty seconds for a worker
  that answered thirty seconds ago buys nothing. A pilot measured 3078
  transport opens for 344 simulations, and 41 in three minutes with nothing
  running, until the host's sshd began refusing connections. A worker that is
  in the pool and was last found *ready* within ``probe_ttl_sec`` is taken at
  its word; everything else is measured, so a failure is still noticed within
  one interval and recovery is unaffected. An operator pressing "check" gets
  a real measurement regardless.
* **The connections a cycle does need are shared.** Docker's inventory and
  every probe on one host ride a single hop-1 session (``lib.transport.pool``),
  which costs a host one handshake per cycle instead of one per worker plus
  one. Hop 2 stays per container: that is the connection a simulation runs on.
* **It replaces the optimistic defaults** that existed while nothing measured
  anything. A host now reaches ``online`` because a probe said so.
"""
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from lib.credentials import CredentialError, SecretOpener
from lib.discovery import discover_host, unavailable_report
from lib.transport import (
    HostKeyMismatch,
    JumpPool,
    PlanError,
    TransportError,
    build_transport,
)
from pylib.db.repositories.infra import HostWithdrawing
from pylib.timeutil import elapsed_since, utc_now
from pylib.db.models.infra import (
    EnumHostStatus,
    EnumHostType,
    EnumResourceStatus,
    # The eligibility rule lives in the shared library because the release
    # path and the API depend on the same answer. Re-exported here, where it
    # used to be defined, so this module stays the one place the health cycle
    # reads it from.
    operator_configured,
)

log = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 30.0
# Consecutive failures before a host leaves the pool. Three at the default
# interval is a minute and a half of real unreachability, which is long enough
# that a blip does not cost capacity and short enough to notice.
DEFAULT_FAILURE_THRESHOLD = 3
# How long a reservation may sit without the simulation starting. This covers
# only the window between claiming a container and the JVM reporting its PID,
# so it is minutes — unrelated to how long a simulation may legitimately run.
DEFAULT_RESERVATION_TTL_SEC = 600.0
# How long a successful probe speaks for the worker it measured. Ten intervals
# at the default: a healthy container is re-verified twice in ten minutes
# instead of twenty times, and a container that stops answering is still found
# within one interval, because only a *ready* result earns the silence.
DEFAULT_PROBE_TTL_SEC = 300.0

# How long a generation may stay open with nothing running in it before this
# closes it. Generously long: it has to outlast the enqueue of a whole
# generation, which inserts its simulations one at a time, and the cost of
# waiting is a delay in a repair rather than anything lost.
DEFAULT_GENERATION_GRACE_SEC = 300.0

STALE_RESERVATION_MESSAGE = (
    "Recovered: the container was reserved but the simulation never started"
)


@dataclass
class HostHealth:
    """The outcome of checking one host."""
    host_name: str
    status: EnumHostStatus
    message: str
    ready: list[str] = field(default_factory=list)
    unhealthy: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Ready, on the strength of a recent probe rather than a new one. A
    #: subset of ``ready``, kept apart so the log says what was measured.
    fresh: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [
            f"{len(self.ready)} ready",
            f"{len(self.unhealthy)} unhealthy",
            f"{len(self.skipped)} busy",
        ]
        if self.fresh:
            parts.append(f"{len(self.fresh)} unchanged")
        return f"{self.status.value}: " + ", ".join(parts)


class HealthMonitor:
    """Runs the check cycle, on a timer and on demand.

    Collaborators are injected rather than imported so the whole cycle can be
    exercised without a network: ``discover`` does level 1 and ``probe`` does
    level 2.
    """

    def __init__(
        self,
        mongo: Any,
        *,
        opener: Optional[SecretOpener] = None,
        interval_sec: float = DEFAULT_INTERVAL_SEC,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        reservation_ttl_sec: float = DEFAULT_RESERVATION_TTL_SEC,
        probe_ttl_sec: float = DEFAULT_PROBE_TTL_SEC,
        generation_grace_sec: float = DEFAULT_GENERATION_GRACE_SEC,
        discover: Optional[Callable[[Any, dict], Any]] = None,
        probe: Optional[Callable[[dict, dict], Any]] = None,
        jumps: Optional[JumpPool] = None,
        on_capacity: Optional[Callable[[], None]] = None,
        reconcile: Optional[Callable[[], Any]] = None,
        inspect_run: Optional[Callable[[dict, dict, Any], Any]] = None,
    ):
        self._mongo = mongo
        self._opener = opener
        self._interval = interval_sec
        self._threshold = max(1, failure_threshold)
        self._reservation_ttl = reservation_ttl_sec
        self._probe_ttl = probe_ttl_sec
        self._generation_grace = generation_grace_sec
        # Hop 1, opened at most once per host per turn. Owned here because the
        # monitor is the only thing that knows when a host's turn is over.
        self._jumps = jumps or JumpPool(opener, on_learn=self._pin_host_key)
        self._discover = discover or self._default_discover
        self._probe = probe or self._default_probe
        self._on_capacity = on_capacity
        # Injected rather than imported: the reconciliation needs a transport
        # and the process's follower registry, neither of which belongs here.
        self._reconcile = reconcile
        # Asks a container what became of a recorded run. The same probe the
        # reconciliation uses; this cycle needs it to decide whether a held
        # worker may come back.
        self._inspect_run = inspect_run
        self._stop = threading.Event()

    # ── the loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info(
            "Health monitor started (every %ss, offline after %d consecutive "
            "failures, a ready probe trusted for %ss)",
            self._interval, self._threshold, self._probe_ttl,
        )
        while not self._stop.wait(self._interval):
            try:
                self.cycle()
            except Exception:
                # The cycle is the only thing that promotes containers into
                # the pool; if it dies, the pool freezes as it is.
                log.exception("Health cycle failed; continuing")

    def stop(self) -> None:
        self._stop.set()
        self._jumps.close()

    def cycle(self, *, force: bool = False) -> list[HostHealth]:
        """One full pass: requests first, then every host, then housekeeping.

        ``force`` measures every container instead of trusting a recent probe.
        The first cycle of a process uses it: a probe left behind by the
        process that just died describes a pool this one has not seen, and it
        is the cycle before the first placement that the dispatcher relies on.
        """
        try:
            self._serve_requests()
        except Exception:
            log.exception("Serving check/discovery requests failed; continuing")

        results = [
            self._check_host_isolated(host, force=force)
            for host in self._mongo.host_repo.find_enabled()
        ]

        try:
            self.sweep_stale_reservations()
        except Exception:
            log.exception("Sweeping stale reservations failed; continuing")

        try:
            self.reconcile_unwatched()
        except Exception:
            log.exception("Reconciling unwatched runs failed; continuing")

        try:
            self.close_finished_generations()
        except Exception:
            log.exception("Closing finished generations failed; continuing")

        try:
            self.resolve_held_workers()
        except Exception:
            log.exception("Resolving held workers failed; continuing")

        if any(result.ready for result in results) and self._on_capacity is not None:
            self._on_capacity()
        return results

    def reconcile_unwatched(self) -> Any:
        """Look again at runs that nobody is following.

        The startup reconciliation handles what a dead process left behind,
        but it ran **once**. A container whose host was unreachable at that
        moment kept its reservation — correctly, since unreachable says
        nothing about the run — and then nothing ever came back to it: this
        cycle skips occupied containers, and the TTL sweep only covers
        reserved ones. The container stayed ``busy`` until someone restarted
        the master-node again.

        So the same reconciliation runs every cycle, over the containers no
        thread of this process is watching. For a healthy run that is the
        empty set, and this costs one query.
        """
        if self._reconcile is None:
            return None
        return self._reconcile()

    def resolve_held_workers(self) -> list[str]:
        """Ask the container about each worker held for something unresolved.

        The counterpart of the quarantine, and the reason it can afford to be
        strict. A worker is taken out of the pool whenever something about it
        is unaccounted for — a run whose stop was never confirmed, a launch
        whose confirmation was lost, a result that could neither be stored nor
        copied aside. Most of those are transient: the connection came back,
        or nothing was running in the first place.

        Without this pass each one would need a person, which is a bad trade
        for a network blip. With it, the only ones that do are the ones where
        the container still has something on it.

        **One rule, and it is deliberately conservative:** the hold is cleared
        only when the container answers *and* has nothing alive *and* has left
        nothing behind. Anything else — still running, still holding bytes,
        or unreachable — stays held. A testlog sitting there is an uncollected
        result, and letting the worker take the next simulation is what
        deletes it.
        """
        if self._inspect_run is None:
            return []
        try:
            held = self._mongo.resource_repo.find_held()
        except Exception:
            log.exception("Could not read the workers held for resolution")
            return []
        freed: list[str] = []
        for resource in held:
            name = str(resource.get("name", "?"))
            record = resource.get("held_for") or {}
            host = self._mongo.host_repo.get(resource.get("host_id"))
            if host is None:
                log.warning(
                    "Worker %s is held but its host is gone; leaving it", name)
                continue
            # May be ``None``, and that is passed through deliberately: the
            # probe then asks the *directory*, because a launch whose
            # confirmation was lost before the pid could be read leaves a JVM
            # with nobody holding its number. Treating "no pid" as "nothing
            # running" would clear exactly the hold that case created.
            pid = record.get("remote_pid") or resource.get("remote_pid")
            try:
                state = self._inspect_run(host, resource, pid)
            except Exception as error:
                log.warning("Could not ask %s about its held run: %s", name, error)
                continue
            if state is None:
                log.info(
                    "Worker %s is held for %s and could not be reached; "
                    "leaving it", name, record.get("kind", "?"))
                continue
            if state.alive:
                log.warning(
                    "Worker %s is held for %s and something is still running "
                    "there (pid %s); leaving it",
                    name, record.get("kind", "?"), pid)
                continue
            if state.testlog_bytes > 0:
                log.warning(
                    "Worker %s is held for %s and still has %d bytes of "
                    "testlog; leaving it so the result is not destroyed",
                    name, record.get("kind", "?"), state.testlog_bytes)
                continue
            if self._unresolved_result(record, name):
                continue
            if self._mongo.resource_repo.clear_held_for(resource["_id"]):
                log.warning(
                    "Worker %s was held for %s; the container answered with "
                    "nothing running and nothing left behind, so it is back "
                    "in service", name, record.get("kind", "?"))
                freed.append(name)
        if freed and self._on_capacity is not None:
            self._on_capacity()
        return freed

    def _unresolved_result(self, record: dict, name: str) -> bool:
        """Whether a held worker is still waiting on a decision about a result.

        The checks above ask the *container*, and they look at
        ``COOJA.testlog``. A result that was preserved has been copied to a
        name the cleanup does not know — which is the point of copying it —
        so the container answers "nothing alive, nothing left behind" and the
        hold would be cleared while the bytes are still sitting there waiting
        for somebody to decide what they are worth.

        Found by running the recovery procedure: the worker came back into the
        pool on its own, before anything had looked at the result it was
        holding.

        So for a collection that is pending, the simulation decides. Resolved
        means published (``recovered_result``) or explicitly given up
        (``unfetched_result.discarded``) — ``util/recover_result.py`` writes
        both, and refuses to release in the wrong order.
        """
        if record.get("kind") != "collection_pending":
            return False
        simulation_id = record.get("simulation_id")
        if not simulation_id:
            return False
        try:
            simulation = self._mongo.simulation_repo.get(str(simulation_id))
        except Exception:
            log.exception(
                "Could not re-read simulation %s; leaving %s held",
                simulation_id, name)
            return True
        if simulation is None:
            # The simulation is gone, so nothing is waiting on the bytes.
            return False
        pending = simulation.get("unfetched_result") or {}
        if not pending or pending.get("discarded"):
            return False
        if simulation.get("recovered_result"):
            return False
        log.info(
            "Worker %s is holding a result nobody has decided about yet "
            "(simulation %s); leaving it. See util/recover_result.py",
            name, simulation_id,
        )
        return True

    def close_finished_generations(self) -> list[str]:
        """Close generations whose simulations are all finished.

        The repair half of R24. Publishing a result and closing its generation
        are two writes over two documents, and the second one can fail on its
        own. It used to propagate: the failure came back out of the publish
        and the generic handler marked the *finished* simulation in error, so
        an administrative failure changed an experiment's outcome.

        The publish now isolates it, which leaves a generation open with
        nothing running in it — and the experiment waiting on an event that
        will never come. This is what comes back for it. ``close_if_finished``
        re-reads everything, so repeating it is free and deciding twice is
        impossible.

        The grace keeps it off a generation that is still being enqueued: the
        simulations are inserted one at a time, and a fast worker can finish
        the first before the second exists.
        """
        if self._generation_grace <= 0:
            return []
        cutoff = utc_now() - timedelta(seconds=self._generation_grace)
        try:
            open_generations = self._mongo.generation_repo.find_open_before(cutoff)
        except Exception:
            log.exception("Could not read the generations still open")
            return []
        closed: list[str] = []
        for generation in open_generations:
            try:
                decided = self._mongo.generation_repo.close_if_finished(
                    generation["_id"], require_simulations=True)
            except Exception:
                log.exception(
                    "Could not close generation %s", generation.get("_id"))
                continue
            if decided:
                log.warning(
                    "Generation %s had nothing running and was still open; "
                    "closed it as %s", generation.get("_id"), decided,
                )
                closed.append(str(generation["_id"]))
        return closed

    def _check_host_isolated(self, host: dict, *, force: bool) -> HostHealth:
        """Check one host, absorbing whatever it throws.

        The pool is decided here, so one host must not be able to stop the
        others from being measured. A pinned-key mismatch on the first host
        used to end the whole pass: the hosts after it were never checked, the
        reservation sweep never ran, and the pool froze as it was. The
        failure is recorded against the host that caused it — an unexplained
        host is worse than an offline one.
        """
        name = str(host.get("name", "?"))
        try:
            return self.check_host(host, force=force)
        except Exception as error:
            log.exception("Checking host %s failed", name)
            message = f"health check failed: {type(error).__name__}: {error}"
            try:
                return self._record(
                    host, name, EnumHostStatus.OFFLINE, message, healthy=False)
            except Exception:
                # Even the bookkeeping failed — Mongo is probably gone. The
                # next cycle will try again; what matters is not dying here.
                log.exception("Could not record the failure of host %s", name)
                return HostHealth(
                    host_name=name, status=EnumHostStatus.OFFLINE, message=message)

    # ── one host ───────────────────────────────────────────────────────────

    def check_host(
        self, host: dict, *, discover: bool = True, force: bool = False
    ) -> HostHealth:
        """Check one host at both levels and record what was observed.

        ``force`` measures every container even if it answered a moment ago —
        what an operator asking for a check means by it.
        """
        try:
            return self._check_host(host, discover=discover, force=force)
        finally:
            # Hop 1 lives exactly as long as this host's turn: holding it open
            # between cycles would be a connection per host sitting idle, and
            # reopening it per container is what this avoids.
            self._jumps.close_host(host)

    def _check_host(self, host: dict, *, discover: bool, force: bool) -> HostHealth:
        name = str(host.get("name", "?"))
        discovery_message = ""

        if discover:
            report = self._discover(self._mongo, host)
            if report is not None and not report.available:
                discovery_message = f"docker: {report.unavailable_reason}"

        resources = self._mongo.resource_repo.find_by_host(host["_id"])
        if not resources:
            # Nothing to check, and nothing to place. A host with no
            # containers is misconfigured rather than down — and if discovery
            # could not run, that is why.
            return self._record(
                host, name, EnumHostStatus.MISCONFIGURED,
                discovery_message or "no Cooja containers found on this host",
                healthy=False,
            )

        health = HostHealth(host_name=name, status=EnumHostStatus.OFFLINE, message="")
        for resource in resources:
            self._check_resource(host, resource, health, force=force)

        status, message = self._conclude(health, discovery_message)
        return self._record(
            host, name, status, message,
            healthy=status in (EnumHostStatus.ONLINE, EnumHostStatus.DEGRADED),
            health=health,
        )

    def _check_resource(
        self, host: dict, resource: dict, health: HostHealth, *, force: bool = False
    ) -> None:
        name = str(resource.get("name", "?"))

        if resource.get("status") in (
            EnumResourceStatus.RESERVED.value, EnumResourceStatus.BUSY.value
        ):
            # Holding a simulation. Probing it would be harmless, but acting
            # on the answer would not, so it is left out of the cycle entirely.
            health.skipped.append(name)
            return

        if not resource.get("enabled", True):
            health.skipped.append(name)
            return

        container_state = (resource.get("container") or {}).get("state")
        checked_at = (resource.get("container") or {}).get("checked_at")
        if (
            checked_at is not None
            and container_state != "running"
            # ...unless the operator stated where this worker is. Docker not
            # seeing it then means only "I could not see it" — on a host whose
            # SSH user has no access to the daemon, it never will — and must
            # not overrule an address a person typed. Observed in deployment:
            # two workers configured with their own addresses and credentials
            # were marked `container is missing` and never probed at all.
            and not operator_configured(host, resource)
        ):
            # Level 1 already answered, and nothing contradicts it: no point
            # opening an SSH connection to a container Docker says is down.
            self._mongo.resource_repo.update_probe(
                resource["_id"], {"ssh_ok": False, "message": f"container {container_state}"})
            self._mongo.resource_repo.mark_offline(
                resource["_id"], f"container is {container_state}")
            health.unhealthy.append(name)
            return

        if not force and probe_is_fresh(resource, self._probe_ttl, host=host):
            # It is in the pool and the last measurement found it ready, so
            # there is nothing a new connection could tell us. Anything that
            # did change — a stopped container, a failed placement, an edited
            # address — either took it out of the pool or cleared the probe,
            # and both put it back on this path.
            health.ready.append(name)
            health.fresh.append(name)
            return

        # Read before the measurement, written with it, and re-read after:
        # a probe takes seconds over two hops, and an access edit landing in
        # that window makes the answer describe the previous configuration.
        version = _config_version(host)
        try:
            probe = self._probe(host, resource)
        except (CredentialError, PlanError, HostKeyMismatch) as error:
            # A configuration problem, not a reachability one. The host key
            # mismatch belongs here rather than with the network failures:
            # waiting will not fix it, and it is cleared by an operator
            # deciding the host was legitimately rebuilt. It must be caught
            # before TransportError, which it now derives from.
            self._mongo.resource_repo.mark_error(resource["_id"], str(error))
            health.unhealthy.append(name)
            return
        except TransportError as error:
            self._mongo.resource_repo.update_probe(
                resource["_id"], {"ssh_ok": False, "message": str(error)})
            self._mongo.resource_repo.mark_offline(resource["_id"], str(error))
            health.unhealthy.append(name)
            return

        if self._configuration_moved(host, version):
            # Discarded rather than written. Publishing it would put a worker
            # into the pool on the strength of a measurement of how it used
            # to be reached, which is the delayed-publication half of R22.
            log.info(
                "Discarding the probe of %s: the host's access settings "
                "changed while it ran", name,
            )
            health.unhealthy.append(name)
            return

        self._mongo.resource_repo.update_probe(resource["_id"], {
            "ssh_ok": probe.reachable,
            "java_version": probe.java_version,
            "java_ok": probe.java_ok,
            "cooja_jar_sha256": probe.cooja_jar_sha256,
            "message": probe.message,
        }, config_version=version)
        if probe.ready:
            # The version travels with the promotion, and `reserve` requires
            # the two to match: a measurement of the previous configuration
            # cannot make a worker schedulable, even in the window between the
            # re-read above and this write.
            self._mongo.resource_repo.mark_available(
                resource["_id"], probe.message, config_version=version)
            health.ready.append(name)
        else:
            self._mongo.resource_repo.mark_offline(resource["_id"], probe.message)
            health.unhealthy.append(name)

    def _configuration_moved(self, host: dict, version: int) -> bool:
        """Whether this host's access settings changed during the measurement."""
        try:
            fresh = self._mongo.host_repo.get(host["_id"])
        except Exception:
            # Cannot tell. Writing the probe is the previous behaviour and the
            # placement is still guarded by the freshness check, which
            # compares the version this write is about to store.
            log.exception(
                "Could not re-read host %s to confirm its configuration",
                host.get("name"),
            )
            return False
        return fresh is not None and _config_version(fresh) != version

    def _conclude(
        self, health: HostHealth, discovery_message: str
    ) -> tuple[EnumHostStatus, str]:
        """Decide the host's status from what its containers reported."""
        prefix = f"{discovery_message}; " if discovery_message else ""
        if health.ready and health.unhealthy:
            # One sick container must not take a whole host out of the pool.
            return EnumHostStatus.DEGRADED, (
                f"{prefix}{len(health.unhealthy)} of "
                f"{len(health.ready) + len(health.unhealthy)} container(s) unhealthy"
            )
        if health.ready:
            return EnumHostStatus.ONLINE, f"{prefix}{len(health.ready)} container(s) ready"
        if health.skipped and not health.unhealthy:
            # Everything is busy or disabled; nothing was measured, so the
            # host's previous standing is as good as it gets.
            return EnumHostStatus.ONLINE, f"{prefix}all containers in use"
        return EnumHostStatus.OFFLINE, f"{prefix}no container could be reached"

    def _record(
        self,
        host: dict,
        name: str,
        status: EnumHostStatus,
        message: str,
        *,
        healthy: bool,
        health: Optional[HostHealth] = None,
    ) -> HostHealth:
        """Persist the outcome, applying the threshold and the hysteresis.

        A failure is counted and only becomes ``offline`` once it has happened
        often enough; a success resets the count immediately. The asymmetry is
        deliberate — capacity should be slow to lose and quick to regain.
        """
        failures = int(host.get("consecutive_failures") or 0)
        effective = status

        if not healthy:
            if failures + 1 < self._threshold and host.get("status") in (
                EnumHostStatus.ONLINE.value, EnumHostStatus.DEGRADED.value
            ):
                effective = EnumHostStatus.DEGRADED
                message = (
                    f"{message} (failure {failures + 1} of {self._threshold} "
                    "before going offline)"
                )

        self._mongo.host_repo.mark_check_result(
            host["_id"], status=effective, message=message, reset_failures=healthy)

        result = health or HostHealth(host_name=name, status=effective, message=message)
        result.status = effective
        result.message = message
        log.info("Host %s — %s", name, result.describe())
        return result

    def _close_generation(self, simulation_id: Any) -> None:
        _close_generation_after_sweep(self._mongo, simulation_id)

    def _default_probe(self, host: dict, resource: dict) -> Any:
        transport = build_transport(
            host, resource, self._opener, jump_client=self._jump_client(host))
        with transport:
            return transport.probe()

    def _default_discover(self, mongo: Any, host: dict) -> Any:
        """Level 1, over the session this host's turn already has."""
        try:
            session = self._jumps.session(host)
        except HostWithdrawing as error:
            # Asked to discover a host that is being removed. Recorded like
            # any other reason Docker could not be consulted, rather than
            # raised: `_serve_requests` has no handler, and one withdrawing
            # host would cost every other host its queued request.
            return unavailable_report(mongo, host, str(error))
        except (CredentialError, PlanError, TransportError) as error:
            # Hop 1 is not available, so Docker cannot be consulted either.
            # The probes that follow will raise the same cached error and
            # report it per container; here it only has to be written down,
            # and not attempted a second time.
            return unavailable_report(mongo, host, str(error))
        try:
            return discover_host(
                mongo, host, opener=self._opener, session=session)
        except HostWithdrawing as error:
            return unavailable_report(mongo, host, str(error))

    def _pin_host_key(self, host: dict, fingerprint: str) -> None:
        """Persist a host key accepted on first contact.

        The health cycle is usually the first thing to reach a host — it runs
        before anything is placed — and it had no way to write the key down,
        so the pin that the next connection would be checked against was
        never recorded. The write refuses to overwrite an existing pin: trust
        on first use only means something if the first use is the one that
        sticks.
        """
        try:
            if self._mongo.host_repo.pin_host_key(host["_id"], fingerprint):
                log.info(
                    "Pinned the host key of %s on first contact: %s",
                    host.get("name"), fingerprint,
                )
                return
            stored = ((self._mongo.host_repo.get(host["_id"]) or {}).get(
                "access") or {}).get("host_key_fingerprint")
        except Exception:
            log.exception("Could not pin the host key of %s", host.get("name"))
            return
        if stored and stored != fingerprint:
            # The pin was won by another path with a different key. The policy
            # only calls this when it found none, so the session that just
            # accepted this one is already open — refusing it is the point.
            log.error(
                "Host %s is pinned to %s but presented %s; refusing this "
                "session", host.get("name"), stored, fingerprint,
            )
            raise HostKeyMismatch(
                str(host.get("name") or host["_id"]), stored, fingerprint)

    def _jump_client(self, host: dict) -> Optional[Any]:
        """The host session to use as hop 1, or ``None`` for a local host."""
        session = self._jumps.session(host)
        return session.client if session is not None else None

    # ── requests from the API ──────────────────────────────────────────────

    def _serve_requests(self) -> None:
        """Carry out checks and discoveries the API asked for.

        The API has neither a route to the container network nor the private
        key that opens a credential, so it records a request and this does the
        work. The timestamp is cleared only if no newer request arrived while
        it ran.
        """
        for host in self._mongo.host_repo.find_all():
            for field_name, handler in (
                ("discovery_requested_at", self._serve_discovery),
                ("check_requested_at", self._serve_check),
            ):
                requested_at = host.get(field_name)
                if not requested_at:
                    continue
                log.info(
                    "Serving %s for host %s", field_name, host.get("name"))
                try:
                    handler(host)
                finally:
                    self._mongo.host_repo.clear_request(
                        host["_id"], field_name, requested_at)

    def _serve_discovery(self, host: dict) -> None:
        try:
            self._discover(self._mongo, host)
        finally:
            # This is the one path that opens a host session outside
            # ``check_host``, so it has to end the turn it started: a host
            # that is disabled, or that has no containers, would otherwise
            # leave a connection open until the monitor stops.
            self._jumps.close_host(host)

    def _serve_check(self, host: dict) -> None:
        # Re-read: discovery may have just changed this host's resources.
        fresh = self._mongo.host_repo.get(host["_id"]) or host
        # Forced: being asked to check and answering from a cached probe would
        # make the button a no-op exactly when someone is trying to find out
        # whether their change worked.
        self.check_host(fresh, discover=False, force=True)

    # ── housekeeping ───────────────────────────────────────────────────────

    def sweep_stale_reservations(self) -> list[str]:
        """Free containers claimed for a simulation that never started.

        Covers only the window between the reservation and the JVM reporting
        its PID. A reservation that got that far is ``busy`` and is governed
        by the simulation timeout instead, which may legitimately be hours.
        """
        if self._reservation_ttl <= 0:
            return []
        cutoff = utc_now() - timedelta(seconds=self._reservation_ttl)
        stale = self._mongo.resource_repo.find_stale_reservations(cutoff)
        swept: list[str] = []
        for resource in stale:
            name = str(resource.get("name", "?"))
            log.warning(
                "Container %s has been reserved since %s without a simulation "
                "starting; freeing it", name, resource.get("reserved_at"),
            )
            simulation_id = resource.get("current_simulation_id")
            attempt_id = resource.get("attempt_id")
            host = self._mongo.host_repo.get(resource.get("host_id"))

            # The container first, and only then the simulation. The two live
            # in different documents, so one of them has to go first — and if
            # the runner has meanwhile promoted this reservation to busy and
            # started the JVM, the release below refuses and nothing about
            # the simulation is touched. Doing it the other way round failed a
            # live simulation for a reservation that was never freed.
            if not self._mongo.resource_repo.force_release(
                resource["_id"], STALE_RESERVATION_MESSAGE,
                container_state_counts=not operator_configured(
                    host or {}, resource),
                # Only if it is still the reservation that was read. Between
                # the query and this write the runner may have promoted it to
                # busy and started the JVM, and freeing it then would hand a
                # container that is running something to the next placement.
                expect_attempt_id=attempt_id,
                reserved_before=cutoff,
                # And still only a reservation: the attempt guard alone would
                # also match a container the runner has meanwhile promoted to
                # busy with the same attempt, which is a live simulation.
                expect_reserved=True,
            ):
                log.info(
                    "Container %s was no longer the reservation that was read; "
                    "leaving it and its simulation alone", name,
                )
                continue

            swept.append(name)
            if simulation_id and attempt_id:
                try:
                    # Expired *and revoked*: clearing the attempt is what stops
                    # a thread that still believes in it from following this
                    # with a start and a result.
                    self._mongo.simulation_repo.expire_attempt(
                        simulation_id, attempt_id, STALE_RESERVATION_MESSAGE)
                except Exception:
                    log.exception("Failed to expire simulation %s", simulation_id)
            self._close_generation(simulation_id)
        if swept and self._on_capacity is not None:
            self._on_capacity()
        return swept


def _close_generation_after_sweep(mongo: Any, simulation_id: Any) -> None:
    """Re-evaluate the generation of a simulation the sweep just failed.

    The sweep was the one place that failed a simulation without this. If it
    failed the last one still active, the generation stayed pending for ever
    and the experiment waited on a strategy event that would never come.
    """
    if not simulation_id:
        return
    try:
        simulation = mongo.simulation_repo.get(str(simulation_id))
    except Exception:
        log.exception("Could not re-read simulation %s", simulation_id)
        return
    generation_id = (simulation or {}).get("generation_id")
    if not generation_id:
        return
    try:
        mongo.generation_repo.close_if_finished(generation_id)
    except Exception:
        log.exception("Failed to close generation %s", generation_id)


def _config_version(host: dict) -> int:
    return int((host or {}).get("config_version") or 0)


def probe_is_fresh(
    resource: dict, ttl_sec: float, *, host: Optional[dict] = None
) -> bool:
    """Whether this worker's last probe still speaks for it.

    Only a worker *in the pool* whose last probe found it *ready* qualifies:
    it answered and a Cooja build was there. A failure, a worker that was
    never probed, one the scheduler took offline, one whose probe was cleared
    because its address changed — all are measured again. That asymmetry is
    the point: silence is only inherited from good news, so nothing is slower
    to recover than before, and only the no-news case is skipped.
    """
    if ttl_sec <= 0:
        return False
    if resource.get("status") != EnumResourceStatus.AVAILABLE.value:
        return False
    probe = resource.get("probe") or {}
    # The same three requirements readiness asks for. A probe written before
    # the JVM became one of them has no `java_ok`, so it is simply measured
    # again — which is the right answer for an observation that no longer
    # means what it used to.
    if not probe.get("ssh_ok") or not probe.get("cooja_jar_sha256"):
        return False
    if not probe.get("java_ok"):
        return False
    if host is not None and int(
        probe.get("config_version") or 0
    ) != _config_version(host):
        # It measured a different configuration. Not a failure — just not
        # evidence about this one (R22).
        return False
    # Naive or aware, depending on when the document was written: the infra
    # collections store aware UTC now, and anything from before does not say.
    elapsed = elapsed_since(probe.get("checked_at"))
    if elapsed is None:
        return False
    return 0 <= elapsed <= ttl_sec


def local_host_only(hosts: list[dict]) -> list[dict]:
    """The bootstrapped local host, if it is registered."""
    return [h for h in hosts if h.get("type") == EnumHostType.LOCAL.value]
