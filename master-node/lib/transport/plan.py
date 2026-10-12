"""Where a simulation runs, resolved from the host and resource records.

A ``ConnectionPlan`` is a composition, resolved in a fixed order of
precedence so that one host's worth of configuration covers the common case
while any single worker can still differ:

1. the worker's own ``connection`` and ``endpoint``, if stated;
2. the host's ``container_access``, as the default for everything absent;
3. built-in defaults.

That order is what makes both deployments work. On a host whose Docker socket
the SSH user can reach, discovery fills in each worker's endpoint and nothing
has to be typed. On a shared host where it cannot — a university machine, say
— each worker is registered with its own address and port, and no Docker
access is needed at all.

The local host produces a plan with no jump — the master-node sits on
``cooja-net`` and resolves container names through Docker's DNS — so the local
execution path stays exactly what it was before this feature.
"""
from dataclasses import dataclass
from typing import Any, Optional

from pylib.db.models.infra import EnumHostType

DEFAULT_REMOTE_DIR = "/opt/contiki-ng/tools/cooja"
DEFAULT_CONTAINER_SSH_PORT = 22
DEFAULT_HOST_SSH_PORT = 22


class PlanError(ValueError):
    """The records do not describe a reachable target."""


@dataclass(frozen=True)
class SSHEndpoint:
    address: str
    port: int

    def __str__(self) -> str:
        return f"{self.address}:{self.port}"


@dataclass(frozen=True)
class ConnectionPlan:
    """One hop or two, plus where Cooja lives inside the container."""
    target: SSHEndpoint
    remote_dir: str
    jump: Optional[SSHEndpoint] = None
    host_name: str = "?"
    resource_name: str = "?"

    @property
    def is_tunnelled(self) -> bool:
        return self.jump is not None

    def describe(self) -> str:
        """A short label for logs — addresses only, never credentials."""
        if self.jump is None:
            return f"{self.host_name}/{self.resource_name} direct {self.target}"
        return f"{self.host_name}/{self.resource_name} via {self.jump} -> {self.target}"


def build_plan(host: dict[str, Any], resource: dict[str, Any]) -> ConnectionPlan:
    """Compose the plan for one placement.

    Raises ``PlanError`` rather than guessing: a worker with no address has
    neither been discovered nor configured, and inventing one would send a
    simulation somewhere nobody verified.
    """
    endpoint = resource.get("endpoint") or {}
    address = endpoint.get("address")
    if not address:
        raise PlanError(
            f"worker {resource.get('name')!r} has no address. Either run a "
            "discovery on its host, or give it one in its own connection "
            "settings."
        )

    container_access = host.get("container_access") or {}
    connection = resource.get("connection") or {}
    target = SSHEndpoint(
        address=address,
        # The worker's own port wins, then its endpoint's, then the host
        # default. A worker reached through a published port differs from its
        # neighbours in exactly this field.
        port=int(
            connection.get("port")
            or endpoint.get("port")
            or container_access.get("port")
            or DEFAULT_CONTAINER_SSH_PORT
        ),
    )

    jump: Optional[SSHEndpoint] = None
    if (host.get("type") or EnumHostType.REMOTE.value) != EnumHostType.LOCAL.value:
        access = host.get("access") or {}
        jump_address = access.get("address")
        if not jump_address:
            raise PlanError(
                f"remote host {host.get('name')!r} has no SSH address configured"
            )
        jump = SSHEndpoint(
            address=jump_address,
            port=int(access.get("port") or DEFAULT_HOST_SSH_PORT),
        )

    return ConnectionPlan(
        target=target,
        remote_dir=resource.get("remote_dir") or DEFAULT_REMOTE_DIR,
        jump=jump,
        host_name=str(host.get("name") or "?"),
        resource_name=str(resource.get("name") or "?"),
    )
