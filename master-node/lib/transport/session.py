"""Hop 1 as a first-class thing: a session on the host itself.

Two callers need the host without wanting a container: the Docker inspector,
which asks the host what containers it has, and the health monitor, which asks
whether the host answers at all. Both predate any resource, so neither can go
through a container-oriented transport.

``connect_host`` is shared with ``SSHTransport``'s jump so the host-key
pinning, the keepalive and the error classification exist once.
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
) -> SSHClient:
    """Open an authenticated session on a host, with its key pinned.

    ``TransportUnreachable`` is raised only for "could not get there", kept
    separate from an authentication or protocol failure: the caller treats the
    two differently, because unreachable says nothing about work already
    running on that host.
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(
        PinnedHostKeyPolicy(pinned_host_key, on_learn_host_key))
    try:
        client.connect(
            endpoint.address,
            port=endpoint.port,
            timeout=timeout,
            **credential.connect_kwargs(),
        )
    except HostKeyMismatch:
        client.close()
        raise
    except (NoValidConnectionsError, OSError) as e:
        client.close()
        raise TransportUnreachable(f"host {endpoint} is unreachable: {e}") from None
    except SSHException as e:
        client.close()
        raise TransportError(f"SSH to host {endpoint} failed: {e}") from None

    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(KEEPALIVE_INTERVAL)
    return client


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
    ):
        self.endpoint = endpoint
        self.host_name = host_name
        self._credential = credential
        self._pinned_host_key = pinned_host_key
        self._on_learn_host_key = on_learn_host_key
        self._timeout = timeout
        self._client: Optional[SSHClient] = None

    def open(self) -> None:
        if self._client is None:
            self._client = connect_host(
                self.endpoint,
                self._credential,
                pinned_host_key=self._pinned_host_key,
                on_learn_host_key=self._on_learn_host_key,
                timeout=self._timeout,
            )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("Error closing session to %s", self.endpoint)
            self._client = None

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
