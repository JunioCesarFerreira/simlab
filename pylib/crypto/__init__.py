"""Sealed secrets for infrastructure credentials.

Deliberately re-exports only the envelope half, which runs on the standard
library. ``pylib.crypto.sealed`` needs PyNaCl and is imported explicitly, by
the master-node alone — importing it from here would quietly give the REST API
(whose environment has no PyNaCl) the ability to open secrets, and break its
startup besides.
"""
from pylib.crypto.envelope import (
    ENVELOPE_ALG,
    MAX_PLAINTEXT_BYTES,
    EnvelopeError,
    MalformedEnvelope,
    SecretEnvelope,
    UnknownKeyId,
    is_envelope,
    public_key_fingerprint,
    redact,
    validate_envelope,
)

__all__ = [
    "ENVELOPE_ALG",
    "MAX_PLAINTEXT_BYTES",
    "EnvelopeError",
    "MalformedEnvelope",
    "SecretEnvelope",
    "UnknownKeyId",
    "is_envelope",
    "public_key_fingerprint",
    "redact",
    "validate_envelope",
]
