"""Distributed execution infrastructure: hosts, Cooja resources, crypto keys.

Three entities, deliberately separate (see
docs/markdown/DISTRIBUTED_RESOURCES_PLAN.md, decision D2):

* **Host** — a machine, and the unit of load balancing.
* **Cooja resource** — a container, and the unit of reservation.
* **Connection** — *how* a resource is reached, composed from the host's access
  config plus the resource's discovered endpoint.

Keeping the connection out of the resource's identity is what lets an address,
a port or a credential change by editing a registration instead of re-enrolling
every container, and leaves room for a transport other than SSH later.

Only stdlib and ``pylib.crypto.envelope`` are imported here, so the REST API
can map these documents without gaining the ability to open a secret.
"""
from datetime import datetime
from enum import Enum
from typing import Any, NotRequired, Optional, TypedDict

from bson import ObjectId

from pylib.crypto.envelope import SecretEnvelope

HOSTS_COLLECTION = "hosts"
RESOURCES_COLLECTION = "cooja_resources"
CRYPTO_COLLECTION = "infra_crypto"


class EnumHostType(str, Enum):
    LOCAL = "local"      # the master-node's own Docker daemon, reached directly
    REMOTE = "remote"    # reached over SSH, then tunnelled into its containers


class EnumHostStatus(str, Enum):
    ONLINE = "online"                # reachable, and its containers answered
    DEGRADED = "degraded"            # reachable, but some resource is failing
    OFFLINE = "offline"              # unreachable for N consecutive checks
    MISCONFIGURED = "misconfigured"  # credential missing, unusable, or host key changed
    DISABLED = "disabled"            # administratively out of the pool


class EnumResourceStatus(str, Enum):
    AVAILABLE = "available"   # container running, probe OK, free to take work
    RESERVED = "reserved"     # claimed by the dispatcher, not yet started
    BUSY = "busy"             # a simulation is running on it
    OFFLINE = "offline"       # container gone/stopped, or probe failing
    ERROR = "error"           # a fault was recorded against it
    DISABLED = "disabled"     # administratively out of the pool


class EnumAuthMethod(str, Enum):
    KEY = "key"
    PASSWORD = "password"


class EnumAddressMode(str, Enum):
    """How a container's address is obtained for the second SSH hop.

    The first three are derived from Docker inspection. ``EXPLICIT`` means the
    operator stated it per worker, which is the only option that works on a
    host where the SSH user cannot reach the Docker socket — a shared machine,
    typically — and the clearest one wherever the ports are already known.
    """
    CONTAINER_IP = "container_ip"      # bridge IP; derived
    PUBLISHED_PORT = "published_port"  # a port published on the host; derived
    CONTAINER_NAME = "container_name"  # Docker DNS; only on the local host
    EXPLICIT = "explicit"              # stated per worker, not derived


class EnumDiscoveryMode(str, Enum):
    """Where a host's list of workers comes from."""
    DOCKER = "docker"    # enumerated by querying Docker on the host
    MANUAL = "manual"    # registered one by one, with their own parameters


class EnumContainerState(str, Enum):
    RUNNING = "running"
    EXITED = "exited"
    PAUSED = "paused"
    CREATED = "created"
    RESTARTING = "restarting"
    DEAD = "dead"
    MISSING = "missing"    # not returned by Docker any more


# States in which a resource is holding on to a simulation. Used by the
# reservation guard, the delete guard in the API and the inventory snapshot.
OCCUPIED_STATUSES = (EnumResourceStatus.RESERVED, EnumResourceStatus.BUSY)

# Host states in which the scheduler may still place work. DEGRADED stays in:
# one sick container must not take a whole host out of the pool.
SCHEDULABLE_HOST_STATUSES = (EnumHostStatus.ONLINE, EnumHostStatus.DEGRADED)


class AccessConfig(TypedDict, total=False):
    """Hop 1 — how the master-node reaches the host itself.

    Absent for a local host, which needs no hop. ``secret`` is a sealed
    envelope: the private key or password, readable only by the master-node.
    """
    transport: str                 # "ssh"
    address: str                   # IP or DNS name
    port: int
    username: str
    auth_method: str               # EnumAuthMethod
    secret: Optional[SecretEnvelope]
    key_fingerprint: Optional[str]       # of the SSH key, for display
    host_key_fingerprint: Optional[str]  # pinned on first contact (TOFU)


class ContainerAccess(TypedDict, total=False):
    """Hop 2 — how the host reaches its own Cooja containers.

    A property of the container image rather than of any single container,
    which is why it lives on the host and not on each resource.
    """
    transport: str                 # "ssh"
    username: str
    port: int                      # sshd port inside the container
    address_mode: str              # EnumAddressMode
    auth_method: str               # EnumAuthMethod
    secret: Optional[SecretEnvelope]


class DiscoveryConfig(TypedDict, total=False):
    """Where this host's workers come from.

    ``docker`` enumerates them by querying the host's Docker daemon, which
    also resolves each one's address. ``manual`` means they are registered
    individually — necessary when the SSH user cannot reach the Docker socket,
    and simpler whenever the addresses are already known.
    """
    mode: str                      # EnumDiscoveryMode
    label_filter: str              # e.g. "simlab.type=cooja"
    name_prefix: Optional[str]
    endpoint: str                  # "socket" (local) | "ssh-cli" (remote)
    last_discovery: Optional[datetime]
    discovered_count: NotRequired[int]
    last_discovery_message: NotRequired[str]


class Host(TypedDict, total=False):
    id: str
    name: str
    type: str                      # EnumHostType
    enabled: bool
    status: str                    # EnumHostStatus
    access: Optional[AccessConfig]
    container_access: ContainerAccess
    discovery: DiscoveryConfig
    origin: str                    # "bootstrap" | "manual"
    last_check: Optional[datetime]
    last_check_message: str
    consecutive_failures: int
    # Set by the API, consumed by the master-node: the API has neither a route
    # to the containers nor the private key, so it asks instead of acting.
    check_requested_at: Optional[datetime]
    discovery_requested_at: Optional[datetime]
    created_time: datetime
    updated_time: datetime


class ContainerInfo(TypedDict, total=False):
    """What Docker reported about the container backing a resource."""
    id: str
    name: str
    image: str
    state: str                     # EnumContainerState
    labels: dict[str, str]
    checked_at: Optional[datetime]


class ResourceEndpoint(TypedDict, total=False):
    """Hop-2 target: where this worker's sshd answers.

    Filled by discovery on a Docker-managed host, and stated by the operator
    on a manually registered one.
    """
    address: str
    port: int


class ResourceConnection(TypedDict, total=False):
    """One worker's own SSH parameters, overriding the host's defaults.

    Present only on the fields that differ: anything absent falls back to the
    host's ``container_access``, which keeps the common case to a single
    configuration while still letting one worker out of eight use a different
    user, port or credential.

    This is the ``connection_config`` the specification asks for in §3.2. It
    exists because deriving every address from Docker, as discovery does,
    assumes an access to the daemon that a shared host will not grant.
    """
    transport: str                 # "ssh"
    username: str
    port: int
    auth_method: str               # EnumAuthMethod
    secret: Optional[SecretEnvelope]


class ResourceProbe(TypedDict, total=False):
    """Functional readiness, as opposed to Docker-level existence.

    A container can be ``running`` and still not be able to run a simulation,
    so availability needs both levels; this is the second one.
    """
    ssh_ok: bool
    java_version: Optional[str]
    #: Whether the JVM that launches Cooja actually ran. The version string
    #: alone was collected and ignored, so a container with a jar and a broken
    #: runtime entered the pool and failed at launch instead.
    java_ok: NotRequired[bool]
    cooja_jar_sha256: Optional[str]
    checked_at: Optional[datetime]
    message: NotRequired[str]


class CoojaResource(TypedDict, total=False):
    id: str
    host_id: ObjectId
    name: str
    container: ContainerInfo
    endpoint: ResourceEndpoint
    # Per-worker overrides. Absent fields fall back to host.container_access.
    connection: ResourceConnection
    remote_dir: str
    enabled: bool                  # administrative override; survives discovery
    status: str                    # EnumResourceStatus
    current_simulation_id: Optional[ObjectId]
    # Identifies one placement attempt. Persisted on both the resource and the
    # simulation so a late thread from a previous attempt can never release a
    # resource that has already been handed to someone else.
    attempt_id: Optional[str]
    remote_pid: Optional[int]
    reserved_at: Optional[datetime]
    started_at: Optional[datetime]
    probe: ResourceProbe
    origin: str                    # "discovered" | "manual"
    last_check: Optional[datetime]
    last_check_message: str
    created_time: datetime
    updated_time: datetime


class InfraCryptoKey(TypedDict):
    key_id: str
    public_key: str                # base64 X25519, safe to publish
    created_at: datetime


class InfraCrypto(TypedDict, total=False):
    """The published half of the master-node's keystore.

    Written by the master-node, read by the API so the browser can seal against
    the right key. Private material never appears here.
    """
    id: str
    alg: str
    key_id: str
    public_key: str
    fingerprint: str
    created_at: datetime
    updated_at: datetime
    retired_keys: list[dict[str, Any]]


# ── Elegibilidade: de quem é a palavra final ─────────────────────────────────

def operator_configured(host: dict[str, Any], resource: dict[str, Any]) -> bool:
    """Whether a person, not Docker, decided how this worker is reached.

    When they did, the SSH probe is the whole of the evidence. Docker's
    silence about a worker says nothing about whether it answers, and on a
    shared host the daemon may be unreachable by design — the SSH user often
    has no access to it, so discovery reports nothing and the stored container
    state says ``missing`` while the worker runs simulations perfectly well.

    Three signals, any of which is enough:

    * the worker carries its own ``connection`` settings;
    * it was registered by hand (``origin`` is ``manual``);
    * its host lists workers manually, or states addresses explicitly.

    This lives here, in the shared library, because three different decisions
    depend on it and they must agree: the health cycle, which probes such a
    worker regardless of Docker; the release, which must not evict it on a
    stale ``missing``; and the API, which tells the page why a column is not
    being used. They disagreed once, and the cost was measurable — two remote
    workers left the pool on *every* completed simulation and only came back
    at the next health cycle, because the release path alone had not been
    given the exception.
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
