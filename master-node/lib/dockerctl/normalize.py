"""Turning a ``docker inspect`` document into a resource record.

Two outputs per container:

* a ``ContainerInfo`` — what the resource is, and whether it is running. This
  is the first of the two availability levels; the SSH probe is the second.
* an endpoint — *where* the second SSH hop should connect. Deriving it here is
  what makes remote support work without publishing container ports on the
  LAN, and without an operator maintaining an address per container.
"""
import logging
from typing import Any, Optional

from pylib.db.models.infra import ContainerInfo, EnumAddressMode, EnumContainerState

log = logging.getLogger(__name__)

# Docker's own vocabulary is stored verbatim rather than mapped onto an enum.
# It has grown values before ("removing"), and a mapping table that falls
# behind would silently report a container as something it is not. Only
# "running" is given meaning, and only here.
RUNNING = EnumContainerState.RUNNING.value

LOOPBACK = "127.0.0.1"
ANY_IPV4 = "0.0.0.0"


def container_name(raw: dict[str, Any]) -> str:
    """Docker reports ``/cooja1``; the resource is named ``cooja1``."""
    return str(raw.get("Name") or "").lstrip("/")


def normalize_container(raw: dict[str, Any]) -> ContainerInfo:
    config = raw.get("Config") or {}
    state = raw.get("State") or {}
    status = str(state.get("Status") or "").lower()
    if not status:
        # Older/odd payloads expose only the boolean.
        status = RUNNING if state.get("Running") else EnumContainerState.DEAD.value
    return {
        "id": str(raw.get("Id") or ""),
        "name": container_name(raw),
        "image": str(config.get("Image") or raw.get("Image") or ""),
        "state": status,
        "labels": dict(config.get("Labels") or {}),
    }


def is_running(container: ContainerInfo) -> bool:
    return container.get("state") == RUNNING


def resolve_endpoint(
    raw: dict[str, Any],
    *,
    address_mode: str,
    container_port: int,
) -> Optional[dict[str, Any]]:
    """Work out where to connect for hop 2, or ``None`` if it cannot be known.

    Returning ``None`` matters: a resource with no endpoint is refused by
    ``build_plan`` instead of being guessed at, so a simulation is never sent
    to an address nobody verified.
    """
    networks = ((raw.get("NetworkSettings") or {}).get("Networks")) or {}

    if address_mode == EnumAddressMode.CONTAINER_NAME.value:
        # Only valid where Docker's embedded DNS answers for the caller, i.e.
        # the master-node sharing a network with the containers.
        name = container_name(raw)
        return {"address": name, "port": container_port} if name else None

    if address_mode == EnumAddressMode.PUBLISHED_PORT.value:
        ports = (raw.get("NetworkSettings") or {}).get("Ports") or {}
        bindings = ports.get(f"{container_port}/tcp") or []
        for binding in bindings:
            host_port = binding.get("HostPort")
            if not host_port:
                continue
            host_ip = binding.get("HostIp") or LOOPBACK
            # A binding on 0.0.0.0 is reachable from the host over loopback,
            # which is the shortest path and does not depend on the host
            # knowing its own external address.
            if host_ip in ("", ANY_IPV4, "::"):
                host_ip = LOOPBACK
            return {"address": host_ip, "port": int(host_port)}
        return None

    # container_ip (the default for a remote host): the bridge address, which
    # the host itself can route to. Sorted by network name so repeated
    # discovery passes of a multi-network container agree with each other.
    for net_name in sorted(networks):
        address = (networks[net_name] or {}).get("IPAddress")
        if address:
            return {"address": address, "port": container_port}

    legacy = (raw.get("NetworkSettings") or {}).get("IPAddress")
    if legacy:
        return {"address": legacy, "port": container_port}
    return None


def matches_name_prefix(raw: dict[str, Any], name_prefix: Optional[str]) -> bool:
    if not name_prefix:
        return True
    return container_name(raw).startswith(name_prefix)


def parse_label_filter(label_filter: Optional[str]) -> Optional[tuple[str, Optional[str]]]:
    """Split ``key=value`` (or a bare ``key``) for client-side filtering."""
    if not label_filter:
        return None
    if "=" in label_filter:
        key, value = label_filter.split("=", 1)
        return key.strip(), value.strip()
    return label_filter.strip(), None


def matches_label(raw: dict[str, Any], label_filter: Optional[str]) -> bool:
    """Check a label client-side.

    The daemon already filters, so this is a second line of defence for the
    SSH path, where the filter travels through a shell.
    """
    parsed = parse_label_filter(label_filter)
    if parsed is None:
        return True
    key, value = parsed
    labels = dict(((raw.get("Config") or {}).get("Labels")) or {})
    if key not in labels:
        return False
    return value is None or labels[key] == value
