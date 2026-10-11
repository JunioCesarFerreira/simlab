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
    """How the second SSH hop's address is derived from Docker inspection."""
    CONTAINER_IP = "container_ip"      # bridge IP; the default on a remote host
    PUBLISHED_PORT = "published_port"  # a port published on the host's loopback
    CONTAINER_NAME = "container_name"  # Docker DNS; only on the local host


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
    """How this host's resources are enumerated.

    Resources are discovered, never hand-enrolled: Docker is the authority on
    which containers exist, and the inspection also resolves the hop-2 address.
    """
    mode: str                      # "docker"
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
    """Hop-2 target, resolved by discovery rather than registered."""
    address: str
    port: int


class ResourceProbe(TypedDict, total=False):
    """Functional readiness, as opposed to Docker-level existence.

    A container can be ``running`` and still not be able to run a simulation,
    so availability needs both levels; this is the second one.
    """
    ssh_ok: bool
    java_version: Optional[str]
    cooja_jar_sha256: Optional[str]
    checked_at: Optional[datetime]
    message: NotRequired[str]


class CoojaResource(TypedDict, total=False):
    id: str
    host_id: ObjectId
    name: str
    container: ContainerInfo
    endpoint: ResourceEndpoint
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
