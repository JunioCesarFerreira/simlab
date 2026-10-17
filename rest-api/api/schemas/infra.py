"""The request shapes the ``/infra`` routes actually accept, as JSON Schema.

Why this is generated rather than declared
------------------------------------------
These routes take a ``dict`` body on purpose. The rules they apply cannot be
expressed by a Pydantic model without losing them: a field that is **absent**,
one that is **null** and one that **carries a value** mean three different
things (leave it, stop overriding it, override it), and a credential has to be
refused unless it arrives sealed. Those rules live in ``api.mappers.infra``
and are tested there.

What was missing is the *published contract*. ``Body(..., description="See
HostCreateDto")`` tells a reader to go and find a TypedDict that FastAPI never
publishes, so the OpenAPI document described an arbitrary object and the
generated clients and the Swagger page said nothing true.

So the schema is built here **from the same tuples the mapper enforces**. One
source of truth for the field names, and
``tests/test_infra_openapi.py`` fails if the two ever drift — which is the
only thing that makes a hand-written schema trustworthy.
"""
from typing import Any

from api.mappers.infra import (
    _ACCESS_FIELDS,
    _CONNECTION_FIELDS,
    _CONTAINER_ACCESS_FIELDS,
    _DISCOVERY_FIELDS,
    _ENDPOINT_FIELDS,
    _HOST_FIELDS,
    _RESOURCE_FIELDS,
    _MAX_PORT,
    _MIN_PORT,
)
from pylib.db.models.infra import EnumAddressMode, EnumAuthMethod

#: A sealed credential. Never plaintext: the browser or the CLI seals it
#: against the master-node's published key, and the API stores the ciphertext
#: without being able to open it.
SEALED_SECRET: dict[str, Any] = {
    "type": ["object", "null"],
    "description": (
        "A credential sealed against the master-node's public key "
        "(`GET /infra/crypto/public-key`). Plaintext is refused with 422. "
        "`null` removes the stored credential and falls back to what is "
        "inherited."
    ),
    "properties": {
        "alg": {"const": "crypto_box_seal"},
        "key_id": {"type": "string"},
        "ciphertext": {"type": "string", "description": "base64"},
    },
    "required": ["alg", "key_id", "ciphertext"],
    "additionalProperties": False,
}

_PORT: dict[str, Any] = {
    "type": "integer", "minimum": _MIN_PORT, "maximum": _MAX_PORT,
    "description": (
        "A whole number. A fractional one is refused rather than truncated, "
        "and so is a boolean or a string that is not a number."
    ),
}

_FIELDS: dict[str, dict[str, Any]] = {
    "name": {"type": "string", "minLength": 1},
    "enabled": {
        "type": "boolean",
        "description": "A JSON boolean. The string \"false\" is refused.",
    },
    "transport": {"const": "ssh"},
    "address": {"type": "string", "minLength": 1},
    "port": _PORT,
    "username": {"type": "string", "minLength": 1},
    "auth_method": {"enum": [m.value for m in EnumAuthMethod]},
    "address_mode": {"enum": [m.value for m in EnumAddressMode]},
    "secret": SEALED_SECRET,
    "key_fingerprint": {"type": ["string", "null"]},
    "host_key_fingerprint": {"type": ["string", "null"]},
    "remote_dir": {"type": "string"},
    "mode": {"enum": ["docker", "manual"]},
    "label_filter": {"type": ["string", "null"]},
    "name_prefix": {"type": ["string", "null"]},
    "endpoint": {"type": ["object", "null"]},
    "last_discovery": {"type": ["string", "null"]},
    "discovered_count": {"type": ["integer", "null"]},
    "last_discovery_message": {"type": ["string", "null"]},
}


def _block(fields: tuple[str, ...], *, nullable: bool = True,
           description: str = "") -> dict[str, Any]:
    """One nested object, with exactly the fields the mapper accepts.

    ``additionalProperties: false`` is the published form of
    ``_refuse_unknown``: an unknown field is a 422, not something dropped
    quietly.
    """
    schema: dict[str, Any] = {
        "type": ["object", "null"] if nullable else "object",
        "properties": {
            name: _FIELDS.get(name, {}) for name in fields
        },
        "additionalProperties": False,
    }
    if description:
        schema["description"] = description
    return schema


def _root(fields: tuple[str, ...], *, required: tuple[str, ...] = ()) -> dict:
    nested = {
        "access": _block(
            _ACCESS_FIELDS,
            description="Hop 1 — how the master-node reaches the machine."),
        "container_access": _block(
            _CONTAINER_ACCESS_FIELDS,
            description=(
                "Hop 2 — the default for every worker on this host. A worker "
                "overrides any of it in its own `connection`.")),
        "discovery": _block(
            _DISCOVERY_FIELDS,
            description="How this host's workers are found."),
        "endpoint": _block(
            _ENDPOINT_FIELDS,
            description=(
                "Where this worker's sshd answers. The port sent here is the "
                "effective one: a conflicting `connection.port` left by an "
                "older client is removed.")),
        "connection": _block(
            _CONNECTION_FIELDS,
            description=(
                "What this worker overrides for itself. A field sent as "
                "`null` stops overriding and inherits again; one left out "
                "keeps what is stored.")),
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: nested.get(name, _FIELDS.get(name, {})) for name in fields
        },
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


HOST_CREATE = _root(_HOST_FIELDS, required=("name", "access"))
HOST_UPDATE = _root(_HOST_FIELDS)
RESOURCE_CREATE = _root(_RESOURCE_FIELDS, required=("name", "endpoint"))
RESOURCE_UPDATE = _root(_RESOURCE_FIELDS)
RESOURCE_ENABLED = {
    "type": "object",
    "properties": {"enabled": _FIELDS["enabled"]},
    "required": ["enabled"],
    "additionalProperties": False,
}


def request_body(schema: dict[str, Any], description: str) -> dict[str, Any]:
    """The ``openapi_extra`` block for a route that validates by hand."""
    return {
        "requestBody": {
            "required": True,
            "description": description,
            "content": {"application/json": {"schema": schema}},
        }
    }
