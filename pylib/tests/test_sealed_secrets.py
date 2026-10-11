"""Tests for the sealed-secret envelope and the master-node keystore.

These lock in the properties the design depends on:
  * a secret sealed in the browser opens on the master-node and nowhere else;
  * a plaintext sent where an envelope belongs is refused, not stored;
  * rotation keeps older secrets readable until they are re-sealed;
  * no error message or metadata path can carry a secret out;
  * the REST API's half never imports PyNaCl.
"""
import base64
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from pylib.crypto import envelope as env
from pylib.crypto.sealed import (
    Keystore,
    KeystoreError,
    derive_key_id,
    generate_keypair,
    seal_for,
)

SECRET = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA\n-----END OPENSSH PRIVATE KEY-----"


@pytest.fixture
def store(tmp_path: Path) -> Keystore:
    return Keystore.load_or_create(tmp_path / "x25519.json")


# ── Round trip ───────────────────────────────────────────────────────────────

class TestRoundTrip:
    def test_seal_then_open(self, store: Keystore):
        assert store.open(store.seal(SECRET)) == SECRET

    def test_envelope_shape_is_persistable(self, store: Keystore):
        e = store.seal(SECRET)
        assert e["alg"] == env.ENVELOPE_ALG
        assert e["key_id"] == store.key_id
        assert env.validate_envelope(e) is e

    def test_ciphertext_differs_every_time(self, store: Keystore):
        # Sealed boxes use a fresh ephemeral key per call, so re-sealing the
        # same secret must not produce a comparable value — otherwise equal
        # ciphertexts would reveal that two hosts share a credential.
        assert store.seal(SECRET)["ciphertext"] != store.seal(SECRET)["ciphertext"]

    def test_sealing_needs_only_the_public_key(self, store: Keystore):
        # This is what the browser does: it holds no private key.
        e = seal_for(store.public_key_b64, SECRET)
        assert e["key_id"] == store.key_id
        assert store.open(e) == SECRET

    def test_another_keystore_cannot_open_it(self, store: Keystore, tmp_path: Path):
        other = Keystore.load_or_create(tmp_path / "other.json")
        with pytest.raises(env.UnknownKeyId):
            other.open(store.seal(SECRET))

    @pytest.mark.parametrize("secret", [
        "x",                                   # single byte
        "senha com acento: ção",               # non-ASCII
        "a" * env.MAX_PLAINTEXT_BYTES,         # at the limit
        "line1\nline2\n",                      # PEM-ish newlines
    ])
    def test_payload_variants(self, store: Keystore, secret: str):
        assert store.open(store.seal(secret)) == secret

    def test_refuses_empty_and_oversized(self, store: Keystore):
        with pytest.raises(ValueError):
            store.seal("")
        with pytest.raises(ValueError):
            store.seal("a" * (env.MAX_PLAINTEXT_BYTES + 1))


# ── Refusing plaintext: the check that protects the database ─────────────────

class TestRefusesPlaintext:
    def test_bare_string_is_not_an_envelope(self):
        with pytest.raises(env.MalformedEnvelope, match="must be an object"):
            env.validate_envelope("hunter2")

    def test_plaintext_smuggled_as_ciphertext_is_refused(self):
        # A password base64'd by hand is valid base64 but far too short to be a
        # sealed box. This is the realistic mistake, and it must not be stored.
        with pytest.raises(env.MalformedEnvelope, match="not sealed"):
            env.validate_envelope({
                "alg": env.ENVELOPE_ALG,
                "key_id": "0" * 16,
                "ciphertext": base64.b64encode(b"hunter2").decode(),
            })

    def test_plaintext_field_names_are_refused(self):
        with pytest.raises(env.MalformedEnvelope, match="password"):
            env.validate_envelope({
                "alg": env.ENVELOPE_ALG,
                "key_id": "0" * 16,
                "ciphertext": base64.b64encode(b"x" * 60).decode(),
                "password": "hunter2",
            })

    @pytest.mark.parametrize("broken, match", [
        ({"alg": "aes-gcm"}, "must be 'crypto_box_seal'"),
        ({"key_id": "NOTHEX"}, "16 lowercase hex"),
        ({"key_id": "abc"}, "16 lowercase hex"),
        ({"ciphertext": "not base64!!"}, "valid base64"),
        ({"alg": ""}, "non-empty string"),
        ({"key_id": None}, "non-empty string"),
    ])
    def test_malformed_fields(self, broken, match):
        base = {
            "alg": env.ENVELOPE_ALG,
            "key_id": "0" * 16,
            "ciphertext": base64.b64encode(b"x" * 60).decode(),
        }
        with pytest.raises(env.MalformedEnvelope, match=match):
            env.validate_envelope({**base, **broken})

    def test_oversized_ciphertext_is_refused(self):
        with pytest.raises(env.MalformedEnvelope, match="exceeds"):
            env.validate_envelope({
                "alg": env.ENVELOPE_ALG,
                "key_id": "0" * 16,
                "ciphertext": base64.b64encode(b"x" * (env.MAX_CIPHERTEXT_BYTES + 1)).decode(),
            })

    def test_is_envelope_does_not_raise(self):
        assert env.is_envelope("hunter2") is False


# ── Nothing leaks ────────────────────────────────────────────────────────────

class TestNoLeak:
    def test_redact_withholds_the_ciphertext(self, store: Keystore):
        e = store.seal(SECRET)
        shown = env.redact(e)
        assert shown == {
            "has_secret": True,
            "alg": env.ENVELOPE_ALG,
            "key_id": store.key_id,
            "updated_at": e["updated_at"],
        }
        assert "ciphertext" not in json.dumps(shown, default=str)

    def test_redact_of_nothing(self):
        assert env.redact(None) is None

    def test_corrupt_ciphertext_error_carries_no_contents(self, store: Keystore):
        e = store.seal(SECRET)
        raw = bytearray(base64.b64decode(e["ciphertext"]))
        raw[-1] ^= 0xFF                      # break the MAC
        e["ciphertext"] = base64.b64encode(bytes(raw)).decode()
        with pytest.raises(env.MalformedEnvelope) as exc:
            store.open(e)
        assert "corrupt" in str(exc.value)
        assert e["ciphertext"] not in str(exc.value)

    def test_unknown_key_error_names_keys_not_contents(self, store: Keystore, tmp_path: Path):
        other = Keystore.load_or_create(tmp_path / "other.json")
        e = store.seal(SECRET)
        with pytest.raises(env.UnknownKeyId) as exc:
            other.open(e)
        message = str(exc.value)
        assert store.key_id in message and other.key_id in message
        assert e["ciphertext"] not in message


# ── Keystore file handling ───────────────────────────────────────────────────

class TestKeystoreFile:
    def test_created_private(self, tmp_path: Path):
        path = tmp_path / "x25519.json"
        Keystore.load_or_create(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_existing_keystore_is_never_overwritten(self, tmp_path: Path):
        path = tmp_path / "x25519.json"
        first = Keystore.load_or_create(path)
        again = Keystore.load_or_create(path)
        assert again.key_id == first.key_id
        assert again.open(first.seal(SECRET)) == SECRET

    def test_group_readable_keystore_is_refused(self, tmp_path: Path):
        path = tmp_path / "x25519.json"
        Keystore.load_or_create(path)
        path.chmod(0o640)
        with pytest.raises(KeystoreError, match="group/others"):
            Keystore.load(path)

    def test_key_id_must_match_its_public_key(self, tmp_path: Path):
        path = tmp_path / "x25519.json"
        store = Keystore.load_or_create(path)
        payload = json.loads(path.read_text())
        payload["active"]["key_id"] = "f" * 16        # hand-edited
        path.write_text(json.dumps(payload))
        path.chmod(0o600)
        with pytest.raises(KeystoreError, match="does not match"):
            Keystore.load(path)
        assert store.key_id != "f" * 16

    def test_unknown_version_is_refused(self, tmp_path: Path):
        path = tmp_path / "x25519.json"
        Keystore.load_or_create(path)
        payload = json.loads(path.read_text())
        payload["version"] = 99
        path.write_text(json.dumps(payload))
        path.chmod(0o600)
        with pytest.raises(KeystoreError, match="version"):
            Keystore.load(path)

    def test_missing_keystore_is_refused_by_load(self, tmp_path: Path):
        with pytest.raises(KeystoreError, match="cannot read"):
            Keystore.load(tmp_path / "absent.json")

    def test_env_var_selects_the_path(self, tmp_path: Path, monkeypatch):
        path = tmp_path / "from-env.json"
        monkeypatch.setenv("INFRA_KEY_PATH", str(path))
        store = Keystore.load_or_create()
        assert path.exists() and store.key_id


# ── Rotation ─────────────────────────────────────────────────────────────────

class TestRotation:
    def test_old_secrets_still_open_after_rotation(self, store: Keystore):
        old = store.seal(SECRET)
        old_key_id = store.key_id

        new_key_id = store.rotate()

        assert new_key_id != old_key_id
        assert store.retired_key_ids == [old_key_id]
        # The whole point: nothing has to be re-sealed before it can be read.
        assert store.open(old) == SECRET

    def test_reseal_moves_an_envelope_to_the_active_key(self, store: Keystore):
        old = store.seal(SECRET)
        store.rotate()

        resealed = store.reseal(old)

        assert resealed is not None
        assert resealed["key_id"] == store.key_id
        assert store.open(resealed) == SECRET

    def test_reseal_of_a_current_envelope_is_a_no_op(self, store: Keystore):
        # Lets the rotation walk skip a write instead of churning documents.
        assert store.reseal(store.seal(SECRET)) is None

    def test_rotation_survives_a_reload(self, tmp_path: Path):
        path = tmp_path / "x25519.json"
        store = Keystore.load_or_create(path)
        old = store.seal(SECRET)
        store.rotate()

        reloaded = Keystore.load(path)

        assert reloaded.key_id == store.key_id
        assert reloaded.retired_key_ids == store.retired_key_ids
        assert reloaded.open(old) == SECRET

    def test_two_rotations_keep_both_retired_keys(self, store: Keystore):
        first = store.seal(SECRET)
        store.rotate()
        second = store.seal(SECRET)
        store.rotate()

        assert len(store.retired_key_ids) == 2
        assert store.open(first) == SECRET
        assert store.open(second) == SECRET


# ── Identity helpers ─────────────────────────────────────────────────────────

class TestIdentity:
    def test_key_id_is_derived_from_the_public_key(self):
        _, public_key = generate_keypair()
        key_id = derive_key_id(public_key)
        assert len(key_id) == 16
        assert env.KEY_ID_RE.match(key_id)
        assert derive_key_id(public_key) == key_id      # stable

    def test_fingerprint_is_openssh_shaped(self, store: Keystore):
        fp = store.fingerprint
        assert fp.startswith("SHA256:") and not fp.endswith("=")
        assert fp == env.public_key_fingerprint(store.public_key_b64)


# ── Cross-language compatibility: the premise the design rests on ────────────

# A throwaway keypair that exists only in this file, and the ciphertexts that
# `libsodium-wrappers` (the library the GUI seals with) produced against its
# public half. Checked in so that a change to either side — the browser
# library, PyNaCl, or our envelope format — fails here instead of failing when
# an operator tries to register a host.
#
# Regenerate with, from gui/simlab/:
#   node --input-type=module -e "import _s from 'libsodium-wrappers'; \
#     await _s.ready; const V=_s.base64_variants.ORIGINAL; \
#     console.log(_s.to_base64(_s.crypto_box_seal(_s.from_string('x'), \
#     _s.from_base64('<VECTOR_PUBLIC_KEY>', V)), V))"
VECTOR_PRIVATE_KEY = "Ip31WMHfEDvJ8raxkABwYNJOXxjTFUEhWtC79++uXHs="
VECTOR_PUBLIC_KEY = "dowFp5nE7ZgT6hoHVOwOPOfnOtOiRg+ONJ4+ewJ+N04="
VECTOR_KEY_ID = "b114dd6d8410a62a"

JS_SEALED = {
    "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA\n-----END OPENSSH PRIVATE KEY-----":
        "x4tOgNgVj/w4ULjJWf5JLvw/gvIH4EeDxDPJmXktY1QdUwT3BTLGj+Jx1I+KifzmYCfWC/AZOVcQJA6i"
        "FmuWY9A9Zzib2VX4uvUwVR5H9x+r4ZuO1ku2zSQA5T6VchzCsp3dGaLleZ32m/suNJ3Jw03lIfmv9uy6"
        "+gwKKsS8JZjHJ5av5Zhufbgc",
    "senha com acento: ção":
        "KKTxyv+it8GZhChtttIYvi21dN0Ij5V75A9WKG9EUBp/kv9TDtMLOcjWeOtOt9d8mkWwRKjuwOtDa2vh"
        "v9SySPNbxrfJfsw=",
    "x":
        "nxd8VUTMFZ+XK/AxXB2cgtFL61r59fmjME1zquCjXwtoCwoyEmpipWBH+9rVNB04Ng==",
}


@pytest.fixture
def vector_store(tmp_path: Path) -> Keystore:
    """A keystore holding the fixed vector key."""
    path = tmp_path / "vector.json"
    path.write_text(json.dumps({
        "version": 1,
        "active": {"key_id": VECTOR_KEY_ID, "private_key": VECTOR_PRIVATE_KEY},
        "retired": [],
    }))
    path.chmod(0o600)
    return Keystore.load(path)


class TestBrowserCompatibility:
    def test_vector_key_is_self_consistent(self, vector_store: Keystore):
        assert vector_store.key_id == VECTOR_KEY_ID
        assert vector_store.public_key_b64 == VECTOR_PUBLIC_KEY

    @pytest.mark.parametrize("plaintext", list(JS_SEALED))
    def test_opens_what_the_browser_sealed(self, vector_store: Keystore, plaintext: str):
        envelope: env.SecretEnvelope = {
            "alg": env.ENVELOPE_ALG,
            "key_id": VECTOR_KEY_ID,
            "ciphertext": JS_SEALED[plaintext],
        }
        env.validate_envelope(envelope)
        assert vector_store.open(envelope) == plaintext


# ── The API must stay outside the trust boundary ─────────────────────────────

def test_api_half_does_not_need_pynacl():
    """``pylib.crypto`` must import with no crypto library present.

    rest-api/.venv has no PyNaCl, and that is deliberate: the API validates
    envelopes but must never be able to open one. A stray re-export here would
    both break its startup and quietly widen the trust boundary.
    """
    code = (
        "import sys;"
        "sys.modules['nacl'] = None;"
        "sys.modules['nacl.public'] = None;"
        "import pylib.crypto as c;"
        "c.validate_envelope({'alg': 'crypto_box_seal', 'key_id': '0'*16,"
        " 'ciphertext': __import__('base64').b64encode(b'x'*60).decode()});"
        "print('ok')"
    )
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
