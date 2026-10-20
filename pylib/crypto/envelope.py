"""Sealed-secret envelope: the shape, and nothing that needs a crypto library.

Infrastructure credentials (host SSH keys and passwords) are sealed in the
browser with libsodium's ``crypto_box_seal`` against the master-node's public
key, travel through the REST API as opaque ciphertext and are stored that way.
Only the master-node holds the private key, so the API can neither read nor
write a plaintext secret — it is structurally outside the trust boundary.

That property is why this module exists separately from ``sealed``: the API has
to *validate* what it is handed (and refuse a plaintext pasted into the field),
but it must never gain the ability to open it. Everything here runs on the
standard library alone; ``pylib.crypto.sealed`` holds the half that needs
PyNaCl and is imported only by the master-node.
"""
import base64
import binascii
import hashlib
import re
from typing import Any, NotRequired, TypedDict

# libsodium's sealed box. The only algorithm this codebase accepts: a stored
# envelope naming anything else is from a future version, not a variant to
# interpret.
ENVELOPE_ALG = "crypto_box_seal"

# key_id is derived from the public key (see sealed.derive_key_id), so it is a
# verifiable label rather than a counter someone has to keep in sync.
KEY_ID_RE = re.compile(r"^[0-9a-f]{16}$")

# A sealed box is a 32-byte ephemeral public key plus a 16-byte MAC around the
# plaintext, so 48 bytes is the floor even for an empty secret. Anything
# shorter cannot be a sealed box — most likely someone sent the secret in the
# clear, which is exactly what this check is here to catch.
SEALED_BOX_OVERHEAD = 48

# An ed25519 private key is ~400 bytes of PEM, an RSA-4096 one ~3.2 KB. 16 KiB
# of plaintext leaves generous headroom while keeping a stray upload out of the
# database.
MAX_PLAINTEXT_BYTES = 16 * 1024
MAX_CIPHERTEXT_BYTES = MAX_PLAINTEXT_BYTES + SEALED_BOX_OVERHEAD

_ALLOWED_FIELDS = {"alg", "key_id", "ciphertext", "updated_at"}


class SecretEnvelope(TypedDict):
    """A sealed secret as persisted inside a host document."""
    alg: str
    key_id: str
    ciphertext: str          # base64, standard alphabet with padding
    updated_at: NotRequired[Any]


class EnvelopeError(ValueError):
    """Base class: raised instead of leaking anything about the contents."""


class MalformedEnvelope(EnvelopeError):
    """The value is not a sealed envelope (wrong shape, alg, or encoding)."""


class UnknownKeyId(EnvelopeError):
    """The envelope was sealed against a key this keystore does not hold."""


def validate_envelope(value: Any) -> SecretEnvelope:
    """Check that ``value`` is a well-formed sealed envelope and return it.

    Raises ``MalformedEnvelope`` with a message that describes the *shape*
    problem only — never the value, so an envelope (or a secret mistakenly sent
    as one) cannot reach a log through an error message.
    """
    if not isinstance(value, dict):
        raise MalformedEnvelope("secret must be an object, not a bare value")

    unexpected = set(value) - _ALLOWED_FIELDS
    if unexpected:
        # Named explicitly: a client sending `password` or `private_key` should
        # be told what is wrong, and those names are the client's own, not data.
        raise MalformedEnvelope(
            "unexpected field(s) in secret: " + ", ".join(sorted(unexpected))
        )

    for field in ("alg", "key_id", "ciphertext"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise MalformedEnvelope(f"secret.{field} must be a non-empty string")

    if value["alg"] != ENVELOPE_ALG:
        raise MalformedEnvelope(
            f"secret.alg must be {ENVELOPE_ALG!r}, got {value['alg']!r}"
        )

    if not KEY_ID_RE.match(value["key_id"]):
        raise MalformedEnvelope("secret.key_id must be 16 lowercase hex characters")

    try:
        raw = base64.b64decode(value["ciphertext"], validate=True)
    except (binascii.Error, ValueError):
        raise MalformedEnvelope("secret.ciphertext must be valid base64") from None

    if len(raw) < SEALED_BOX_OVERHEAD + 1:
        raise MalformedEnvelope(
            "secret.ciphertext is too short to be a sealed box — it looks like "
            "the secret was not sealed before being sent"
        )
    if len(raw) > MAX_CIPHERTEXT_BYTES:
        raise MalformedEnvelope(
            f"secret.ciphertext exceeds {MAX_CIPHERTEXT_BYTES} bytes"
        )

    return value  # type: ignore[return-value]


def is_envelope(value: Any) -> bool:
    """Non-raising form of :func:`validate_envelope`."""
    try:
        validate_envelope(value)
    except EnvelopeError:
        return False
    return True


def public_key_fingerprint(public_key_b64: str) -> str:
    """OpenSSH-style fingerprint of a public key, for display and comparison.

    The GUI shows this next to the key it sealed against so an operator can
    confirm the master-node still holds the matching private half.
    """
    raw = base64.b64decode(public_key_b64, validate=True)
    digest = hashlib.sha256(raw).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def redact(envelope: SecretEnvelope | None) -> dict[str, Any] | None:
    """Metadata safe to serialize: never the ciphertext.

    What the API returns for a stored secret. The ciphertext is withheld not
    because it is readable — it is not — but because handing it out invites
    offline attacks the moment a private key leaks.
    """
    if not envelope:
        return None
    out: dict[str, Any] = {
        "has_secret": True,
        "alg": envelope["alg"],
        "key_id": envelope["key_id"],
    }
    if envelope.get("updated_at") is not None:
        out["updated_at"] = envelope["updated_at"]
    return out
