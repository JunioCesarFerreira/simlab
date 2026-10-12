"""Finding out which Cooja containers a host has, by asking Docker.

One of the two ways a host's workers get into the inventory. This one queries
the host's Docker daemon, which answers both what exists and *where* each
container is reachable — an address nobody then has to maintain. The other way
is registering each worker with its own parameters, which is what a host whose
SSH user cannot reach the daemon requires; a host set to ``manual`` is skipped
here entirely.

This is also the first of the two availability levels. It answers "does the
container exist, and is it running", which is necessary but not sufficient: a
container can be running with no usable Cooja build in it. The SSH probe in
``lib.health`` answers the rest, and only a worker that passes both enters the
pool. A manually registered worker has no Docker answer at all, so the probe
is the whole of its evidence.

Deliberate limits
-----------------
* A manually registered worker is never touched. It is invisible to Docker by
  definition, so treating its absence as "vanished" would take a working
  worker out of the pool on every pass.
* A container that has vanished is marked, never deleted. The resource is what
  ties a finished simulation to the container it ran on, and that record
  outlives the container by design.
* A vanished container that is still *holding* a simulation is only reported.
  Whether that simulation died is a question for reconciliation, which can
  probe; a discovery pass has no business deciding it.
* Docker being unreachable is reported as *unavailable*, not as the host
  failing. A host that already has resources can still be checked over SSH —
  which is all the master-node ever did before this feature — so losing the
  Docker socket must not take a working host out of the pool.
"""
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from lib.credentials import SecretOpener, resolve_host_credential
from lib.dockerctl import (
    DockerError,
    DockerUnavailable,
    SocketDockerInspector,
    SSHDockerInspector,
    container_name,
    normalize_container,
    resolve_endpoint,
)
from lib.transport.base import TransportError
from lib.transport.plan import DEFAULT_CONTAINER_SSH_PORT, DEFAULT_REMOTE_DIR, SSHEndpoint
from lib.transport.session import HostSession
from pylib.db.models.infra import EnumAddressMode, EnumDiscoveryMode, EnumHostType

log = logging.getLogger(__name__)

DEFAULT_LABEL_FILTER = "simlab.type=cooja"


@dataclass
class DiscoveryReport:
    """What one discovery pass found."""
    host_name: str
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    missing_occupied: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    unavailable_reason: Optional[str] = None
    #: The host lists its workers by hand, so nothing was queried.
    skipped_manual: bool = False

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def seen(self) -> int:
        return len(self.created) + len(self.updated)

    def describe(self) -> str:
        if self.skipped_manual:
            return "workers are listed manually on this host"
        if not self.available:
            return f"unavailable: {self.unavailable_reason}"
        parts = [f"{self.seen} container(s)"]
        if self.created:
            parts.append(f"{len(self.created)} new")
        if self.missing:
            parts.append(f"{len(self.missing)} gone")
        if self.unresolved:
            parts.append(f"{len(self.unresolved)} without an address")
        return ", ".join(parts)


@contextmanager
def open_inspector(host: dict, opener: Optional[SecretOpener] = None) -> Iterator[Any]:
    """Yield the right Docker inspector for this host, and close what it used.

    Local hosts go through the mounted socket; remote ones run the Docker CLI
    inside the SSH session that hop 1 already authenticated — no extra port,
    no daemon exposed over TCP, no second credential.
    """
    host_type = host.get("type") or EnumHostType.REMOTE.value
    if host_type == EnumHostType.LOCAL.value:
        yield SocketDockerInspector()
        return

    access = host.get("access") or {}
    address = access.get("address")
    if not address:
        raise DockerUnavailable(
            f"host {host.get('name')!r} has no SSH address configured")

    credential = resolve_host_credential(
        access, opener, host_name=str(host.get("name", "?")))
    session = HostSession(
        SSHEndpoint(address, int(access.get("port") or 22)),
        credential,
        pinned_host_key=access.get("host_key_fingerprint"),
        host_name=str(host.get("name", "?")),
    )
    try:
        session.open()
    except TransportError as error:
        raise DockerUnavailable(str(error)) from None
    try:
        yield SSHDockerInspector(session)
    finally:
        session.close()


def discover_host(
    mongo: Any,
    host: dict,
    *,
    opener: Optional[SecretOpener] = None,
    inspector: Optional[Any] = None,
) -> DiscoveryReport:
    """Reconcile one host's resources with what Docker reports.

    ``inspector`` is injectable so this can be driven without a daemon or a
    network; left out, the right one is opened for the host.
    """
    report = DiscoveryReport(host_name=str(host.get("name", "?")))
    discovery = host.get("discovery") or {}

    if (discovery.get("mode") or EnumDiscoveryMode.DOCKER.value) == \
            EnumDiscoveryMode.MANUAL.value:
        # Its workers are registered individually, so there is nothing to
        # enumerate — and on a host whose SSH user cannot reach the Docker
        # socket, trying would only produce a failure that means nothing.
        report.skipped_manual = True
        log.debug(
            "Host %s lists its workers manually; skipping discovery",
            report.host_name,
        )
        return report

    label_filter = discovery.get("label_filter") or DEFAULT_LABEL_FILTER
    name_prefix = discovery.get("name_prefix")

    try:
        if inspector is not None:
            raw_containers = inspector.list_containers(label_filter, name_prefix)
        else:
            with open_inspector(host, opener) as opened:
                raw_containers = opened.list_containers(label_filter, name_prefix)
    except (DockerUnavailable, DockerError) as error:
        # Not a host failure: see the module docstring.
        report.unavailable_reason = str(error)
        log.warning("Discovery unavailable for host %s: %s", report.host_name, error)
        mongo.host_repo.record_discovery(
            host["_id"], count=0, message=f"unavailable: {error}")
        return report
    except Exception as error:
        report.unavailable_reason = f"{type(error).__name__}: {error}"
        log.exception("Discovery failed for host %s", report.host_name)
        mongo.host_repo.record_discovery(
            host["_id"], count=0, message=report.unavailable_reason)
        return report

    containers, endpoints, unresolved = _normalize(host, raw_containers)
    report.unresolved = unresolved

    synced = mongo.resource_repo.sync_discovered(
        host["_id"],
        containers,
        remote_dir=_remote_dir(host),
        endpoints=endpoints,
    )
    report.created = synced["created"]
    report.updated = synced["updated"]
    report.missing = synced["missing"]
    report.missing_occupied = synced["missing_occupied"]

    mongo.host_repo.record_discovery(
        host["_id"], count=report.seen, message=report.describe())
    log.info("Discovery on host %s: %s", report.host_name, report.describe())
    if report.missing_occupied:
        # Reported, not decided: reconciliation probes before concluding that
        # the simulations on those containers are over.
        log.warning(
            "Host %s: container(s) %s vanished while holding a simulation",
            report.host_name, ", ".join(report.missing_occupied),
        )
    return report


def _normalize(
    host: dict, raw_containers: list[dict]
) -> tuple[list[dict], dict[str, dict], list[str]]:
    """Turn inspect documents into resource facts plus hop-2 addresses."""
    container_access = host.get("container_access") or {}
    address_mode = container_access.get("address_mode") or (
        EnumAddressMode.CONTAINER_NAME.value
        if (host.get("type") == EnumHostType.LOCAL.value)
        else EnumAddressMode.CONTAINER_IP.value
    )
    container_port = int(container_access.get("port") or DEFAULT_CONTAINER_SSH_PORT)

    containers: list[dict] = []
    endpoints: dict[str, dict] = {}
    unresolved: list[str] = []

    for raw in raw_containers:
        name = container_name(raw)
        if not name:
            continue
        containers.append(normalize_container(raw))
        endpoint = resolve_endpoint(
            raw, address_mode=address_mode, container_port=container_port)
        if endpoint is None:
            # No address means no placement: build_plan refuses the resource
            # rather than sending a simulation somewhere nobody verified.
            unresolved.append(name)
            continue
        endpoints[name] = endpoint
    return containers, endpoints, unresolved


def _remote_dir(host: dict) -> str:
    return (host.get("container_access") or {}).get("remote_dir") or DEFAULT_REMOTE_DIR


def discover_all(
    mongo: Any,
    *,
    opener: Optional[SecretOpener] = None,
    inspector_for: Optional[Callable[[dict], Any]] = None,
) -> list[DiscoveryReport]:
    """Run a discovery pass over every enabled host."""
    reports: list[DiscoveryReport] = []
    for host in mongo.host_repo.find_enabled():
        inspector = inspector_for(host) if inspector_for is not None else None
        reports.append(discover_host(mongo, host, opener=opener, inspector=inspector))
    return reports
