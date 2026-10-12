"""DTOs for the infrastructure endpoints.

The asymmetry between what comes in and what goes out is the point, and it is
deliberate on both sides:

* **In**, a credential is accepted only as a sealed envelope —
  ``{key_id, ciphertext}`` and nothing else. The API cannot open one and must
  never be handed a plaintext to store.
* **Out**, a credential is described but never returned. The caller learns
  that a secret exists, which key it was sealed against and the fingerprint of
  that key, and no more. Handing the ciphertext back would invite offline
  attacks the moment a private key leaked.

Fields the master-node alone may write — status, the check bookkeeping, the
discovery results, everything a probe measured — appear in the read DTOs and
not in the write ones. They are observations, and an API client producing them
would make the GUI show a state nobody measured.
"""
from datetime import datetime
from typing import Any, NotRequired, Optional, TypedDict


class SealedSecretDto(TypedDict):
    """A credential on its way in. Opaque by construction."""
    key_id: str
    ciphertext: str


class SecretInfoDto(TypedDict):
    """A credential on its way out: that it exists, and against which key."""
    has_secret: bool
    alg: NotRequired[str]
    key_id: NotRequired[str]
    updated_at: NotRequired[Optional[datetime]]


class PublicKeyDto(TypedDict):
    """What the browser seals against.

    Served by the API but produced by the master-node, which is the only
    holder of the matching private half.
    """
    key_id: str
    alg: str
    public_key: str
    fingerprint: str
    created_at: Optional[datetime]
    retired_key_ids: list[str]


# ── Writes ───────────────────────────────────────────────────────────────────

class HostAccessInput(TypedDict, total=False):
    """How to reach the host itself — hop 1."""
    address: str
    port: int
    username: str
    auth_method: str                       # "key" | "password"
    secret: Optional[SealedSecretDto]


class ContainerAccessInput(TypedDict, total=False):
    """How the host reaches its own containers — hop 2.

    A property of the container image rather than of any single container,
    which is why it is configured on the host.
    """
    username: str
    port: int
    address_mode: str                      # container_ip | published_port | container_name
    auth_method: str
    secret: Optional[SealedSecretDto]
    remote_dir: str


class DiscoveryInput(TypedDict, total=False):
    """How this host's containers are enumerated."""
    label_filter: str
    name_prefix: Optional[str]


class HostCreateDto(TypedDict, total=False):
    """A new remote host.

    No ``status`` and no ``type``: the status is measured, and only remote
    hosts are registered this way — the local one is created by the
    master-node from its own environment.
    """
    name: str
    enabled: NotRequired[bool]
    access: HostAccessInput
    container_access: NotRequired[ContainerAccessInput]
    discovery: NotRequired[DiscoveryInput]


class HostUpdateDto(TypedDict, total=False):
    """A change to an existing host. Every field optional."""
    name: NotRequired[str]
    enabled: NotRequired[bool]
    access: NotRequired[HostAccessInput]
    container_access: NotRequired[ContainerAccessInput]
    discovery: NotRequired[DiscoveryInput]


# ── Reads ────────────────────────────────────────────────────────────────────

class HostAccessDto(TypedDict, total=False):
    address: str
    port: int
    username: str
    auth_method: str
    secret: SecretInfoDto
    key_fingerprint: Optional[str]
    host_key_fingerprint: Optional[str]


class ContainerAccessDto(TypedDict, total=False):
    username: str
    port: int
    address_mode: str
    auth_method: str
    secret: SecretInfoDto
    remote_dir: NotRequired[str]


class DiscoveryDto(TypedDict, total=False):
    mode: str
    label_filter: str
    name_prefix: Optional[str]
    endpoint: str
    last_discovery: Optional[datetime]
    discovered_count: NotRequired[int]
    last_discovery_message: NotRequired[str]


class ContainerDto(TypedDict, total=False):
    """What Docker reported — the first of the two availability levels."""
    id: str
    name: str
    image: str
    state: str
    checked_at: Optional[datetime]


class ProbeDto(TypedDict, total=False):
    """What the SSH probe found — the second level.

    ``java_version`` and ``cooja_jar_sha256`` are what make a mixed inventory
    visible: the same generation distributed across different Cooja builds can
    introduce a host effect in the results.
    """
    ssh_ok: bool
    java_version: Optional[str]
    cooja_jar_sha256: Optional[str]
    checked_at: Optional[datetime]
    message: NotRequired[str]


class ResourceDto(TypedDict, total=False):
    id: str
    host_id: str
    host_name: NotRequired[str]
    name: str
    status: str
    enabled: bool
    container: ContainerDto
    endpoint: dict[str, Any]
    remote_dir: str
    probe: ProbeDto
    current_simulation_id: Optional[str]
    remote_pid: Optional[int]
    reserved_at: Optional[datetime]
    started_at: Optional[datetime]
    origin: str
    last_check: Optional[datetime]
    last_check_message: str


class HostDto(TypedDict, total=False):
    id: str
    name: str
    type: str
    enabled: bool
    status: str
    access: Optional[HostAccessDto]
    container_access: ContainerAccessDto
    discovery: DiscoveryDto
    origin: str
    last_check: Optional[datetime]
    last_check_message: str
    consecutive_failures: int
    check_pending: bool
    discovery_pending: bool
    created_time: Optional[datetime]
    updated_time: Optional[datetime]


class HostOccupancyDto(TypedDict, total=False):
    """One host's capacity, as the scheduler itself reads it.

    ``free`` and ``active`` count containers; ``schedulable`` describes the
    host. They are separate because an unreachable host still *has* its
    capacity, and folding the two together would under-report it.
    """
    id: str
    name: str
    type: str
    status: str
    enabled: bool
    schedulable: bool
    resource_total: int
    free: int
    active: int
    unavailable: int
    disabled: int
    last_check: Optional[datetime]
    last_check_message: str


class OverviewTotalsDto(TypedDict):
    hosts: int
    hosts_operational: int
    resources: int
    free: int
    active: int
    unavailable: int
    disabled: int


class OverviewDto(TypedDict):
    """Everything the Infrastructure page needs for its summary."""
    totals: OverviewTotalsDto
    hosts: list[HostOccupancyDto]
    heterogeneous: bool
    cooja_builds: list[str]


class HostDetailDto(TypedDict):
    host: HostDto
    resources: list[ResourceDto]
    occupancy: Optional[HostOccupancyDto]


class ResourceEnabledDto(TypedDict):
    """The only field of a resource an API client may write."""
    enabled: bool


class AcceptedDto(TypedDict):
    """A request the master-node will carry out.

    The API has neither a route to the container network nor the private key
    that opens a credential, so it records the request and answers 202.
    """
    accepted: bool
    requested_at: Optional[datetime]
    detail: str
