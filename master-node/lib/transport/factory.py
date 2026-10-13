"""Building a transport from the host and resource records.

The single seam between persistence and the network: everything needed to
reach a container is derived here from the two documents plus the keystore, so
no caller has to know whether a host is local or remote, or where its
credentials came from.
"""
import logging
from typing import Any, Callable, Optional

from lib.credentials import (
    SecretOpener,
    resolve_container_credential,
    resolve_host_credential,
)
from lib.transport.plan import build_plan
from lib.transport.ssh import SSHTransport
from pylib.db.models.infra import EnumHostType

log = logging.getLogger(__name__)


def build_transport(
    host: dict[str, Any],
    resource: dict[str, Any],
    opener: Optional[SecretOpener] = None,
    *,
    jump_client: Optional[Any] = None,
    on_learn_host_key: Optional[Callable[[str], None]] = None,
) -> SSHTransport:
    """Compose the transport for one placement, without opening it.

    Opening is left to the caller so a reservation can be made first and the
    connection attempted inside the attempt's own error handling.

    ``jump_client`` offers an already-open session on the host to use as hop 1
    (see ``lib.transport.pool``). Given one, nothing here resolves a host
    credential: that connection is authenticated and its host key pinned, so
    opening the secret again would only decrypt it once per container for an
    answer already in hand.
    """
    plan = build_plan(host, resource)
    host_name = plan.host_name

    container_credential = resolve_container_credential(
        host.get("container_access") or {},
        opener,
        host_name=host_name,
        resource_connection=resource.get("connection"),
        resource_name=plan.resource_name,
    )

    jump_credential = None
    pinned = None
    if (
        jump_client is None
        and (host.get("type") or EnumHostType.REMOTE.value) != EnumHostType.LOCAL.value
    ):
        access = host.get("access") or {}
        jump_credential = resolve_host_credential(access, opener, host_name=host_name)
        pinned = access.get("host_key_fingerprint")

    return SSHTransport(
        plan,
        target_credential=container_credential,
        jump_credential=jump_credential,
        jump_client=jump_client,
        pinned_host_key=pinned,
        on_learn_host_key=on_learn_host_key,
    )
