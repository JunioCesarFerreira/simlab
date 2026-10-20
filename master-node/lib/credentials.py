"""Turning a sealed envelope into something paramiko can authenticate with.

The one place where a credential exists in the clear, and only for as long as a
connection takes to open. Nothing here is persisted, logged or returned in an
exception message: failures name the host and what was missing, never the
material.

Two hops need credentials, and they come from different places:

* **hop 1 — the host.** Always an envelope from the database. A missing or
  unopenable one makes the host ``misconfigured``; it never surfaces as a
  simulation failure.
* **hop 2 — the container.** A property of the image, so it lives on the host
  record. With no envelope stored it falls back to ``COOJA_SSH_PASSWORD``
  (default ``root``), which is what the Cooja image has always used — that
  fallback is what keeps the bootstrapped local host working with no
  registration step, without writing a password into the database.
"""
import logging
import os
from dataclasses import dataclass
from io import StringIO
from typing import Any, Optional

from paramiko import ECDSAKey, Ed25519Key, PKey, RSAKey
from paramiko.ssh_exception import SSHException

from pylib.crypto.envelope import EnvelopeError, SecretEnvelope
from pylib.db.models.infra import EnumAuthMethod

log = logging.getLogger(__name__)

# Tried in this order. Ed25519 first because it is what ssh-keygen produces by
# default now, and because trying RSA first makes a wrong guess slower.
#
# No DSA: paramiko 4 dropped DSSKey outright, and ssh-dss has been disabled by
# default in OpenSSH since 7.0. Listing it would only make this module fail to
# import on the newer paramiko for a key type no server would accept.
_KEY_TYPES = (Ed25519Key, ECDSAKey, RSAKey)

DEFAULT_CONTAINER_USER = "root"
# The Cooja image's built-in root password. Read from the environment so it is
# configurable without a schema change, and so it is never written to Mongo.
CONTAINER_PASSWORD_ENV = "COOJA_SSH_PASSWORD"


class CredentialError(RuntimeError):
    """A credential could not be produced. Carries no secret material."""


class CredentialUnavailable(CredentialError):
    """No credential is stored, or the keystore cannot open the one that is."""


@dataclass(frozen=True)
class SSHCredential:
    """Resolved authentication for one SSH hop."""
    username: str
    pkey: Optional[PKey] = None
    password: Optional[str] = None

    def connect_kwargs(self) -> dict[str, Any]:
        """Arguments for ``SSHClient.connect``.

        ``look_for_keys`` and ``allow_agent`` are off: this process must
        authenticate with exactly the credential it was given, never with
        whatever happens to be in the container's filesystem or an agent it
        inherited. A silent fallback would make a misconfigured host appear to
        work on the developer's machine and fail in deployment.
        """
        kwargs: dict[str, Any] = {
            "username": self.username,
            "look_for_keys": False,
            "allow_agent": False,
        }
        if self.pkey is not None:
            kwargs["pkey"] = self.pkey
        if self.password is not None:
            kwargs["password"] = self.password
        return kwargs

    def __repr__(self) -> str:  # pragma: no cover - defensive
        # Never let a credential reach a log through repr() of a frame local.
        kind = "key" if self.pkey is not None else "password" if self.password else "none"
        return f"SSHCredential(username={self.username!r}, auth={kind})"


class SecretOpener:
    """Opens envelopes with the master-node's keystore.

    A thin seam around ``Keystore`` so the transport layer can be tested
    without crypto, and so there is a single place that converts an envelope
    problem into a message safe to store and show.
    """

    def __init__(self, keystore: Any):
        self._keystore = keystore

    def open(self, envelope: Optional[SecretEnvelope], *, what: str) -> str:
        if not envelope:
            raise CredentialUnavailable(f"no {what} credential is stored")
        try:
            return self._keystore.open(envelope)
        except EnvelopeError as e:
            # The envelope's own message already avoids the contents; it says
            # which key was expected, which is what an operator needs.
            raise CredentialUnavailable(f"{what} credential cannot be opened: {e}") from None


def load_private_key(pem: str) -> PKey:
    """Parse a private key, trying each supported type.

    The envelope is the protection, so the key itself is expected to be
    unencrypted. A passphrase-protected key fails here with a clear message
    rather than a paramiko error about a bad format.
    """
    stream = StringIO(pem)
    errors: list[str] = []
    for key_type in _KEY_TYPES:
        stream.seek(0)
        try:
            return key_type.from_private_key(stream)
        except SSHException as e:
            errors.append(f"{key_type.__name__}: {e}")
        except Exception as e:  # pragma: no cover - defensive
            errors.append(f"{key_type.__name__}: {type(e).__name__}")
    if "encrypted" in " ".join(errors).lower():
        raise CredentialError(
            "private key is passphrase-protected; store an unencrypted key "
            "(the sealed envelope is what protects it at rest)"
        )
    raise CredentialError(
        "private key is not in a supported format (tried "
        + ", ".join(t.__name__ for t in _KEY_TYPES) + ")"
    )


def resolve_host_credential(
    access: dict[str, Any],
    opener: SecretOpener,
    *,
    host_name: str = "?",
) -> SSHCredential:
    """Hop 1 — authenticate to the host itself."""
    username = access.get("username")
    if not username:
        raise CredentialUnavailable(f"host {host_name} has no SSH username configured")

    auth_method = access.get("auth_method") or EnumAuthMethod.KEY.value
    try:
        plaintext = opener.open(access.get("secret"), what=f"host {host_name} SSH")
    except CredentialUnavailable:
        raise

    if auth_method == EnumAuthMethod.PASSWORD.value:
        return SSHCredential(username=username, password=plaintext)
    return SSHCredential(username=username, pkey=load_private_key(plaintext))


def resolve_container_credential(
    container_access: dict[str, Any],
    opener: Optional[SecretOpener] = None,
    *,
    host_name: str = "?",
    resource_connection: Optional[dict[str, Any]] = None,
    resource_name: str = "?",
) -> SSHCredential:
    """Hop 2 — authenticate to a Cooja worker.

    Resolved in order of precedence: the worker's own settings, then the
    host's defaults, then the image's built-in root password. One
    configuration per host therefore covers the common case, while a single
    worker can still use a different user or credential.

    The final fallback is deliberate and narrow — it applies only when no
    envelope exists anywhere, so storing one silently disables it.
    """
    connection = resource_connection or {}
    username = (
        connection.get("username")
        or container_access.get("username")
        or DEFAULT_CONTAINER_USER
    )
    # A worker's own envelope wins outright; it is not merged with the host's.
    envelope = connection.get("secret") or container_access.get("secret")
    auth_source = connection if connection.get("secret") else container_access

    if not envelope:
        password = os.getenv(CONTAINER_PASSWORD_ENV, "root")
        log.debug(
            "Worker %s on host %s has no stored credential; using the image "
            "default for user %s", resource_name, host_name, username,
        )
        return SSHCredential(username=username, password=password)

    if opener is None:  # pragma: no cover - defensive
        raise CredentialUnavailable(
            f"worker {resource_name} on host {host_name} has a stored "
            "credential but no keystore was provided to open it"
        )

    plaintext = opener.open(
        envelope, what=f"host {host_name} worker {resource_name} SSH")
    auth_method = auth_source.get("auth_method") or EnumAuthMethod.PASSWORD.value
    if auth_method == EnumAuthMethod.KEY.value:
        return SSHCredential(username=username, pkey=load_private_key(plaintext))
    return SSHCredential(username=username, password=plaintext)


def key_fingerprint(pkey: PKey) -> str:
    """OpenSSH-style fingerprint, for display next to a stored credential."""
    import base64
    import hashlib

    digest = hashlib.sha256(pkey.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")
