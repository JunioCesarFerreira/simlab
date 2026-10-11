"""Where a simulation runs, resolved from the host and resource records.

A ``ConnectionPlan`` is the composition the design calls for: the host supplies
how to reach *it*, the resource supplies the container's own address as Docker
reported it, and neither is part of the resource's identity. Changing a host's
address or credential is an edit to one record, not a re-enrollment of every
container on it.

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

    Raises ``PlanError`` rather than guessing: a resource with no endpoint has
    not been through a successful discovery pass, and inventing an address
    would send a simulation somewhere nobody verified.
    """
    endpoint = resource.get("endpoint") or {}
    address = endpoint.get("address")
    if not address:
        raise PlanError(
            f"resource {resource.get('name')!r} has no resolved endpoint; "
            "it has not completed a discovery pass"
        )

    container_access = host.get("container_access") or {}
    target = SSHEndpoint(
        address=address,
        port=int(endpoint.get("port") or container_access.get("port")
                 or DEFAULT_CONTAINER_SSH_PORT),
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
