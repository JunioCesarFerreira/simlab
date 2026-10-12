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
* **It replaces the optimistic defaults** that existed while nothing measured
  anything. A host now reaches ``online`` because a probe said so.
"""
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from lib.credentials import CredentialError, SecretOpener
from lib.discovery import discover_host
from lib.transport import PlanError, TransportError, build_transport
from pylib.db.models.infra import (
    EnumAddressMode,
    EnumDiscoveryMode,
    EnumHostStatus,
    EnumHostType,
    EnumResourceStatus,
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

    def describe(self) -> str:
        return (
            f"{self.status.value}: {len(self.ready)} ready, "
            f"{len(self.unhealthy)} unhealthy, {len(self.skipped)} busy"
        )


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
        discover: Optional[Callable[[Any, dict], Any]] = None,
        probe: Optional[Callable[[dict, dict], Any]] = None,
        on_capacity: Optional[Callable[[], None]] = None,
    ):
        self._mongo = mongo
        self._opener = opener
        self._interval = interval_sec
        self._threshold = max(1, failure_threshold)
        self._reservation_ttl = reservation_ttl_sec
        self._discover = discover or (
            lambda mongo, host: discover_host(mongo, host, opener=self._opener))
        self._probe = probe or self._default_probe
        self._on_capacity = on_capacity
        self._stop = threading.Event()

    # ── the loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        log.info(
            "Health monitor started (every %ss, offline after %d consecutive failures)",
            self._interval, self._threshold,
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

    def cycle(self) -> list[HostHealth]:
        """One full pass: requests first, then every host, then housekeeping."""
        self._serve_requests()
        results = [self.check_host(host) for host in self._mongo.host_repo.find_enabled()]
        self.sweep_stale_reservations()
        if any(result.ready for result in results) and self._on_capacity is not None:
            self._on_capacity()
        return results

    # ── one host ───────────────────────────────────────────────────────────

    def check_host(self, host: dict, *, discover: bool = True) -> HostHealth:
        """Check one host at both levels and record what was observed."""
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
            self._check_resource(host, resource, health)

        status, message = self._conclude(health, discovery_message)
        return self._record(
            host, name, status, message,
            healthy=status in (EnumHostStatus.ONLINE, EnumHostStatus.DEGRADED),
            health=health,
        )

    def _check_resource(self, host: dict, resource: dict, health: HostHealth) -> None:
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

        try:
            probe = self._probe(host, resource)
        except (CredentialError, PlanError) as error:
            # A configuration problem, not a reachability one.
            self._mongo.resource_repo.mark_error(resource["_id"], str(error))
            health.unhealthy.append(name)
            return
        except TransportError as error:
            self._mongo.resource_repo.update_probe(
                resource["_id"], {"ssh_ok": False, "message": str(error)})
            self._mongo.resource_repo.mark_offline(resource["_id"], str(error))
            health.unhealthy.append(name)
            return

        self._mongo.resource_repo.update_probe(resource["_id"], {
            "ssh_ok": probe.reachable,
            "java_version": probe.java_version,
            "cooja_jar_sha256": probe.cooja_jar_sha256,
            "message": probe.message,
        })
        if probe.ready:
            self._mongo.resource_repo.mark_available(resource["_id"], probe.message)
            health.ready.append(name)
        else:
            self._mongo.resource_repo.mark_offline(resource["_id"], probe.message)
            health.unhealthy.append(name)

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

    def _default_probe(self, host: dict, resource: dict) -> Any:
        transport = build_transport(host, resource, self._opener)
        with transport:
            return transport.probe()

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
        self._discover(self._mongo, host)

    def _serve_check(self, host: dict) -> None:
        # Re-read: discovery may have just changed this host's resources.
        fresh = self._mongo.host_repo.get(host["_id"]) or host
        self.check_host(fresh, discover=False)

    # ── housekeeping ───────────────────────────────────────────────────────

    def sweep_stale_reservations(self) -> list[str]:
        """Free containers claimed for a simulation that never started.

        Covers only the window between the reservation and the JVM reporting
        its PID. A reservation that got that far is ``busy`` and is governed
        by the simulation timeout instead, which may legitimately be hours.
        """
        if self._reservation_ttl <= 0:
            return []
        cutoff = datetime.now() - timedelta(seconds=self._reservation_ttl)
        stale = self._mongo.resource_repo.find_stale_reservations(cutoff)
        swept: list[str] = []
        for resource in stale:
            name = str(resource.get("name", "?"))
            log.warning(
                "Container %s has been reserved since %s without a simulation "
                "starting; freeing it", name, resource.get("reserved_at"),
            )
            simulation_id = resource.get("current_simulation_id")
            if simulation_id:
                try:
                    self._mongo.simulation_repo.mark_error(
                        simulation_id, STALE_RESERVATION_MESSAGE)
                except Exception:
                    log.exception("Failed to fail simulation %s", simulation_id)
            if self._mongo.resource_repo.force_release(
                resource["_id"], STALE_RESERVATION_MESSAGE
            ):
                swept.append(name)
        if swept and self._on_capacity is not None:
            self._on_capacity()
        return swept


def operator_configured(host: dict, resource: dict) -> bool:
    """Whether a person, not Docker, decided how this worker is reached.

    When they did, the SSH probe is the whole of the evidence: Docker's
    silence about a worker says nothing about whether it answers, and on a
    shared host the daemon may be unreachable by design.

    Three signals, any of which is enough:

    * the worker carries its own ``connection`` settings;
    * it was registered by hand (``origin`` is ``manual``);
    * its host lists workers manually, or states addresses explicitly.
    """
    connection = resource.get("connection") or {}
    # `transport` alone is a default the mapper always writes, so it does not
    # count as the operator having configured anything.
    if any(key != "transport" for key in connection):
        return True
    if resource.get("origin") == "manual":
        return True
    discovery_mode = (host.get("discovery") or {}).get("mode")
    if discovery_mode == EnumDiscoveryMode.MANUAL.value:
        return True
    address_mode = (host.get("container_access") or {}).get("address_mode")
    return address_mode == EnumAddressMode.EXPLICIT.value


def local_host_only(hosts: list[dict]) -> list[dict]:
    """The bootstrapped local host, if it is registered."""
    return [h for h in hosts if h.get("type") == EnumHostType.LOCAL.value]
