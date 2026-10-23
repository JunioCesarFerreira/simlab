"""Hop 1 as a first-class thing: a session on the host itself.

Two callers need the host without wanting a container: the Docker inspector,
which asks the host what containers it has, and the health monitor, which asks
whether the host answers at all. Both predate any resource, so neither can go
through a container-oriented transport.

``connect_host`` is shared with ``SSHTransport``'s jump so the host-key
pinning, the keepalive, the name resolution and the error classification
exist once.
"""
import logging
from typing import Callable, Optional

import paramiko
from paramiko import SSHClient
from paramiko.ssh_exception import NoValidConnectionsError, SSHException

from lib.credentials import SSHCredential
from lib.transport.base import ExecResult, TransportError, TransportUnreachable
from lib.transport.hostkey import HostKeyMismatch, PinnedHostKeyPolicy
from lib.transport.plan import SSHEndpoint
from lib.transport.resolve import DEFAULT_RESOLVER, HostResolver, Resolution

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 20.0
KEEPALIVE_INTERVAL = 30


def connect_host(
    endpoint: SSHEndpoint,
    credential: SSHCredential,
    *,
    pinned_host_key: Optional[str] = None,
    on_learn_host_key: Optional[Callable[[str], None]] = None,
    timeout: float = CONNECT_TIMEOUT,
    resolver: Optional[HostResolver] = None,
    on_connected: Optional[Callable[[Resolution, str], None]] = None,
) -> SSHClient:
    """Open an authenticated session on a host, with its key pinned.

    ``TransportUnreachable`` is raised only for "could not get there", kept
    separate from an authentication or protocol failure: the caller treats the
    two differently, because unreachable says nothing about work already
    running on that host. A name that does not resolve raises its subclass
    ``NameResolutionFailed``, which also says whether the master-node's
    resolver or the host's name is at fault.

    The name is resolved here rather than by paramiko, so a failed lookup is
    bounded and classified, and a host with a pinned key can still be reached
    at its last known address while DNS is down. ``on_connected`` is told how
    the address was found and which one answered.

    ``timeout`` bounds the TCP connect and the wait for the server's banner.
    """
    resolver = resolver or DEFAULT_RESOLVER
    resolution = resolver.resolve(
        endpoint.address, endpoint.port, allow_cached=bool(pinned_host_key))

    last_error: Optional[BaseException] = None
    for address in resolution.addresses:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(
            PinnedHostKeyPolicy(pinned_host_key, on_learn_host_key))
        try:
            client.connect(
                address,
                port=endpoint.port,
                timeout=timeout,
                banner_timeout=timeout,
                **credential.connect_kwargs(),
            )
        except HostKeyMismatch:
            client.close()
            raise
        except (NoValidConnectionsError, OSError) as e:
            # Not reachable on this address; the next one may be — a name
            # with an IPv6 record on a network without IPv6 is the usual case.
            client.close()
            last_error = e
            continue
        except SSHException as e:
            client.close()
            raise TransportError(f"SSH to host {endpoint} failed: {e}") from None

        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(KEEPALIVE_INTERVAL)
        resolver.remember(endpoint.address, address)
        if on_connected is not None:
            on_connected(resolution, address)
        return client

    where = ""
    if resolution.source != "literal":
        where = f" at {', '.join(resolution.addresses)}"
    if resolution.from_cache:
        where += f" (its last known address; {resolution.dns_error})"
    raise TransportUnreachable(
        f"host {endpoint} is unreachable{where}: {last_error}") from None


class HostSession:
    """A command session on a host — no container involved."""

    def __init__(
        self,
        endpoint: SSHEndpoint,
        credential: SSHCredential,
        *,
        pinned_host_key: Optional[str] = None,
        on_learn_host_key: Optional[Callable[[str], None]] = None,
        host_name: str = "?",
        timeout: float = CONNECT_TIMEOUT,
        resolver: Optional[HostResolver] = None,
    ):
        self.endpoint = endpoint
        self.host_name = host_name
        self._credential = credential
        self._pinned_host_key = pinned_host_key
        self._on_learn_host_key = on_learn_host_key
        self._timeout = timeout
        self._resolver = resolver
        self._client: Optional[SSHClient] = None
        #: How the address was found and which one answered, once open.
        self.resolution: Optional[Resolution] = None
        self.connected_address: Optional[str] = None

    def open(self) -> None:
        if self._client is None:
            self._client = connect_host(
                self.endpoint,
                self._credential,
                pinned_host_key=self._pinned_host_key,
                on_learn_host_key=self._on_learn_host_key,
                timeout=self._timeout,
                resolver=self._resolver,
                on_connected=self._connected,
            )

    def _connected(self, resolution: Resolution, address: str) -> None:
        self.resolution = resolution
        self.connected_address = address

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("Error closing session to %s", self.endpoint)
            self._client = None

    @property
    def client(self) -> SSHClient:
        """The underlying client, for a caller that needs to open a channel.

        Hop 2 is a ``direct-tcpip`` channel on this connection, so a transport
        reusing this session needs the client itself rather than a command
        interface. It is exposed read-only and never closed by the borrower —
        the session owns it.
        """
        if self._client is None:
            raise TransportError(f"session to {self.endpoint} is not open")
        return self._client

    def __enter__(self) -> "HostSession":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def exec_command(self, command: str, timeout: Optional[float] = None) -> ExecResult:
        if self._client is None:
            raise TransportError(f"session to {self.endpoint} is not open")
        try:
            _, stdout, stderr = self._client.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="ignore")
            err = stderr.read().decode("utf-8", errors="ignore")
            return ExecResult(stdout.channel.recv_exit_status(), out, err)
        except SSHException as e:
            raise TransportError(f"command failed on {self.endpoint}: {e}") from None
