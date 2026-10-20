"""Opening sealed secrets: the master-node half, and the only half with keys.

Holds the X25519 keystore behind ``crypto_box_seal``. The browser seals a host
credential against the public key; this module is the only place in SimLab that
can open one, which is what keeps the REST API out of the trust boundary (see
``pylib.crypto.envelope``).

Requires PyNaCl, which arrives with paramiko on the master-node and is pinned
explicitly there. Nothing in ``rest-api`` may import this module.
"""
import base64
import hashlib
import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.exceptions import CryptoError

from pylib.crypto.envelope import (
    ENVELOPE_ALG,
    MAX_PLAINTEXT_BYTES,
    MalformedEnvelope,
    SecretEnvelope,
    UnknownKeyId,
    public_key_fingerprint,
    validate_envelope,
)

log = logging.getLogger(__name__)

KEYSTORE_VERSION = 1

# Where the private key lives when nothing says otherwise: a mounted secret,
# never the image and never the database. Losing this file makes every stored
# secret unrecoverable, so it belongs in the backup routine.
DEFAULT_KEY_PATH = "/run/secrets/simlab-infra/x25519.json"


class KeystoreError(RuntimeError):
    """The keystore file is missing, unreadable, or unsafe to use."""


def derive_key_id(public_key: bytes) -> str:
    """Label a key by its own public half.

    Derived rather than counted: two master-nodes can generate keys without
    coordinating, and an envelope's key_id can be checked against the key it
    claims instead of trusted.
    """
    return hashlib.sha256(public_key).hexdigest()[:16]


def generate_keypair() -> tuple[bytes, bytes]:
    """A fresh X25519 pair as raw 32-byte values, ``(private, public)``."""
    sk = PrivateKey.generate()
    return bytes(sk), bytes(sk.public_key)


def seal_for(public_key_b64: str, plaintext: str, *, key_id: str | None = None) -> SecretEnvelope:
    """Seal ``plaintext`` against a public key — no private key involved.

    The browser does this with libsodium; this server-side twin exists for
    tests, for the rotation path and for seeding a host from a script.
    """
    raw_plain = plaintext.encode("utf-8")
    if not raw_plain:
        raise ValueError("refusing to seal an empty secret")
    if len(raw_plain) > MAX_PLAINTEXT_BYTES:
        raise ValueError(f"secret exceeds {MAX_PLAINTEXT_BYTES} bytes")

    public_key = base64.b64decode(public_key_b64, validate=True)
    ciphertext = SealedBox(PublicKey(public_key)).encrypt(raw_plain)
    return {
        "alg": ENVELOPE_ALG,
        "key_id": key_id or derive_key_id(public_key),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        "updated_at": datetime.now(timezone.utc),
    }


@dataclass(frozen=True)
class _Key:
    key_id: str
    private_key: bytes

    @property
    def public_key(self) -> bytes:
        return bytes(PrivateKey(self.private_key).public_key)


class Keystore:
    """The master-node's X25519 keys: one active, plus retired ones for rotation.

    Retired keys are kept so secrets sealed before a rotation still open. They
    are dropped only once every envelope has been re-sealed, which
    :meth:`reseal` reports on.
    """

    def __init__(self, active: _Key, retired: list[_Key], path: Path | None = None):
        self._active = active
        self._retired = retired
        self._path = path

    # ── identity ────────────────────────────────────────────────────────────

    @property
    def key_id(self) -> str:
        return self._active.key_id

    @property
    def public_key_b64(self) -> str:
        return base64.b64encode(self._active.public_key).decode("ascii")

    @property
    def fingerprint(self) -> str:
        return public_key_fingerprint(self.public_key_b64)

    @property
    def retired_key_ids(self) -> list[str]:
        return [k.key_id for k in self._retired]

    # ── sealing and opening ─────────────────────────────────────────────────

    def seal(self, plaintext: str) -> SecretEnvelope:
        """Seal against the active key."""
        return seal_for(self.public_key_b64, plaintext, key_id=self.key_id)

    def open(self, envelope: SecretEnvelope) -> str:
        """Recover the plaintext, trying the active key then the retired ones.

        Raises ``UnknownKeyId`` when no held key matches — the caller's cue to
        ask for the secret again rather than retry. Failure messages carry the
        key_id only: never the ciphertext, never a partial plaintext.
        """
        validate_envelope(envelope)
        key_id = envelope["key_id"]
        raw = base64.b64decode(envelope["ciphertext"], validate=True)

        for key in (self._active, *self._retired):
            if key.key_id != key_id:
                continue
            try:
                return SealedBox(PrivateKey(key.private_key)).decrypt(raw).decode("utf-8")
            except CryptoError:
                # The key_id matched but the box did not open: the ciphertext
                # was altered or truncated in storage.
                raise MalformedEnvelope(
                    f"secret sealed against key {key_id} failed to open; "
                    "the stored ciphertext is corrupt"
                ) from None
            except UnicodeDecodeError:
                raise MalformedEnvelope(
                    f"secret sealed against key {key_id} is not valid UTF-8"
                ) from None

        raise UnknownKeyId(
            f"secret was sealed against key {key_id}, which this master-node "
            f"does not hold (active: {self.key_id}; "
            f"retired: {', '.join(self.retired_key_ids) or 'none'})"
        )

    def reseal(self, envelope: SecretEnvelope) -> SecretEnvelope | None:
        """Re-seal an envelope against the active key, or ``None`` if current.

        The rotation step: open with whichever key still holds, seal with the
        new one. Returns ``None`` when there is nothing to do, so a caller can
        skip the write.
        """
        if envelope["key_id"] == self.key_id:
            return None
        return self.seal(self.open(envelope))

    # ── persistence ─────────────────────────────────────────────────────────

    def rotate(self) -> str:
        """Generate a new active key, retiring the current one. Returns its id.

        Does not touch stored envelopes: the caller walks them through
        :meth:`reseal`. Until it does, both keys are needed, which is precisely
        why the retired half is persisted.
        """
        private_key, public_key = generate_keypair()
        self._retired.insert(0, self._active)
        self._active = _Key(derive_key_id(public_key), private_key)
        if self._path is not None:
            self.save(self._path)
        log.info(
            "Rotated infrastructure keystore: active=%s retired=%s",
            self.key_id, ", ".join(self.retired_key_ids),
        )
        return self.key_id

    def save(self, path: str | Path) -> None:
        """Write the keystore at 0600, replacing any previous file atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": KEYSTORE_VERSION,
            "active": _key_to_json(self._active),
            "retired": [_key_to_json(k) for k in self._retired],
        }
        # Written to a sibling temp file so a crash mid-write cannot leave a
        # truncated keystore — which would lose every sealed secret.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".x25519-", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._path = path

    @classmethod
    def load(cls, path: str | Path) -> "Keystore":
        """Load an existing keystore, refusing one other users can read."""
        path = Path(path)
        try:
            mode = path.stat().st_mode
        except OSError as e:
            raise KeystoreError(f"cannot read keystore {path}: {e}") from None

        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            # Same stance as ssh(1): a private key readable by group or others
            # is treated as compromised, not as a warning.
            raise KeystoreError(
                f"keystore {path} is accessible to group/others "
                f"(mode {stat.S_IMODE(mode):04o}); chmod 600 it"
            )

        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise KeystoreError(f"keystore {path} is not readable JSON: {e}") from None

        version = payload.get("version")
        if version != KEYSTORE_VERSION:
            raise KeystoreError(
                f"keystore {path} has version {version!r}, expected {KEYSTORE_VERSION}"
            )
        try:
            active = _key_from_json(payload["active"])
            retired = [_key_from_json(k) for k in payload.get("retired", [])]
        except (KeyError, TypeError, ValueError) as e:
            raise KeystoreError(f"keystore {path} is malformed: {e}") from None

        return cls(active, retired, path)

    @classmethod
    def load_or_create(cls, path: str | Path | None = None) -> "Keystore":
        """Load the keystore, creating one on first boot.

        The master-node calls this at startup: a fresh deployment gets a
        keypair without an operator step, and an existing one is never
        overwritten — the file is only written when absent.
        """
        path = Path(path or os.getenv("INFRA_KEY_PATH", DEFAULT_KEY_PATH))
        if path.exists():
            store = cls.load(path)
            log.info(
                "Loaded infrastructure keystore %s (active=%s, %s)",
                path, store.key_id, store.fingerprint,
            )
            return store

        private_key, public_key = generate_keypair()
        store = cls(_Key(derive_key_id(public_key), private_key), [])
        store.save(path)
        log.warning(
            "Created infrastructure keystore %s (active=%s, %s). Back this file "
            "up: without it every stored host credential is unrecoverable.",
            path, store.key_id, store.fingerprint,
        )
        return store


def _key_to_json(key: _Key) -> dict[str, str]:
    return {
        "key_id": key.key_id,
        "private_key": base64.b64encode(key.private_key).decode("ascii"),
    }


def _key_from_json(raw: dict) -> _Key:
    private_key = base64.b64decode(raw["private_key"], validate=True)
    if len(private_key) != 32:
        raise ValueError("private_key must be 32 bytes")
    key_id = raw["key_id"]
    expected = derive_key_id(bytes(PrivateKey(private_key).public_key))
    if key_id != expected:
        # The id is derived, so a mismatch means the file was hand-edited and
        # the pairing can no longer be trusted.
        raise ValueError(f"key_id {key_id} does not match its public key")
    return _Key(key_id, private_key)
