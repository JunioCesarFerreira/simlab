"""Host-key policy, applied asymmetrically to the two hops.

Replaces the blanket ``AutoAddPolicy`` the master-node used while every target
was a container on a private Docker bridge. Hop 1 now crosses the LAN, which
changes the threat model; hop 2 did not change at all.

* **Hop 1 — the host: pinned on first contact (TOFU).** The first successful
  connection records the fingerprint on the host record; any later change
  refuses the connection and marks the host ``misconfigured`` instead of
  trusting a key that was not there yesterday.
* **Hop 2 — the container: auto-add, on purpose.** A Cooja container is
  recreated routinely and generates a new key every time, and it is reached
  only through a tunnel that hop 1 already authenticated. Pinning it would
  turn ordinary container churn into a stream of false alarms, so the trust
  here is inherited from hop 1 rather than asserted again.
"""
import base64
import hashlib
import logging
from typing import Callable, Optional

from paramiko import MissingHostKeyPolicy, PKey, SSHClient

log = logging.getLogger(__name__)


class HostKeyMismatch(Exception):
    """The host presented a key other than the pinned one."""

    def __init__(self, hostname: str, expected: str, actual: str):
        super().__init__(
            f"host key for {hostname} is {actual}, but {expected} is pinned; "
            "refusing to connect. If the host was legitimately rebuilt, clear "
            "the pinned fingerprint on its registration."
        )
        self.hostname = hostname
        self.expected = expected
        self.actual = actual


def fingerprint(key: PKey) -> str:
    """OpenSSH-style SHA256 fingerprint of a host key."""
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


class PinnedHostKeyPolicy(MissingHostKeyPolicy):
    """Trust on first use, then require the same key.

    ``on_learn`` is called with the fingerprint of a key accepted on first
    contact so the caller can persist it. It is not called on a match, so a
    steady state costs no writes.
    """

    def __init__(
        self,
        expected: Optional[str] = None,
        on_learn: Optional[Callable[[str], None]] = None,
    ):
        self.expected = expected
        self._on_learn = on_learn

    def missing_host_key(self, client: SSHClient, hostname: str, key: PKey) -> None:
        actual = fingerprint(key)
        if self.expected and self.expected != actual:
            raise HostKeyMismatch(hostname, self.expected, actual)
        if not self.expected:
            log.info("Pinning host key for %s: %s", hostname, actual)
            if self._on_learn is not None:
                self._on_learn(actual)
        self.expected = actual


class TunnelledHostKeyPolicy(MissingHostKeyPolicy):
    """Accept a container's key, which is ephemeral by design.

    Deliberately not ``AutoAddPolicy``: that one writes into the client's
    known_hosts, and a container that is recreated with a new key would then
    collide with its own stale entry. This accepts without recording.
    """

    def missing_host_key(self, client: SSHClient, hostname: str, key: PKey) -> None:
        log.debug("Accepting container host key for %s: %s", hostname, fingerprint(key))
