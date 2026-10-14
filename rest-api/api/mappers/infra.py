"""Mapping infrastructure documents to and from the API.

Two rules run through all of it:

* **A credential never comes out.** Every path that could serialize one goes
  through :func:`pylib.crypto.redact`, which returns metadata and withholds the
  ciphertext.
* **An observation never goes in.** Status, check bookkeeping, probe results
  and discovery output are produced by the master-node; the write mappers drop
  them rather than trusting a client to supply them.

Only ``pylib.crypto`` is imported — the envelope half, which runs on the
standard library. The API has no way to open a secret and its environment has
no crypto library, which is the property that keeps it outside the trust
boundary.
"""
from copy import deepcopy
from typing import Any, Optional

from pylib.crypto import MalformedEnvelope, redact, validate_envelope
from pylib.db.models.infra import (
    SCHEDULABLE_HOST_STATUSES,
    EnumDiscoveryMode,
    EnumHostStatus,
    EnumHostType,
    operator_configured,
)

from api.domain.infra import (
    ContainerAccessDto,
    DiscoveryDto,
    HostAccessDto,
    HostDto,
    HostOccupancyDto,
    OverviewDto,
    PublicKeyDto,
    ResourceDto,
)
from api.mappers.helpers import oid_to_str, pop_id

# Fields an API client may never write: each one is something a probe, a
# discovery pass or the scheduler measured.
_OBSERVED_HOST_FIELDS = frozenset({
    "status", "type", "origin", "last_check", "last_check_message",
    "consecutive_failures", "check_requested_at", "discovery_requested_at",
    "created_time", "updated_time", "_id", "id",
})
# `mode` is absent on purpose: where a host's workers come from is an
# operator's decision, not an observation. The rest is produced by a
# discovery pass.
_OBSERVED_DISCOVERY_FIELDS = frozenset({
    "endpoint", "last_discovery", "discovered_count",
    "last_discovery_message",
})
# Anything resembling a credential in the clear. Named explicitly so the
# refusal can say which field was wrong.
_PLAINTEXT_FIELDS = frozenset({
    "password", "passphrase", "private_key", "key", "secret_key",
    "credential", "plaintext",
})


class PlaintextRefused(ValueError):
    """A write tried to send a credential that was not sealed."""


# ── Reads ────────────────────────────────────────────────────────────────────

def host_from_mongo(doc: dict) -> HostDto:
    id_str, d = pop_id(doc)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "type": d.get("type", EnumHostType.REMOTE.value),
        "enabled": bool(d.get("enabled", False)),
        "status": d.get("status", EnumHostStatus.OFFLINE.value),
        "access": _access_from_mongo(d.get("access")),
        "container_access": _container_access_from_mongo(d.get("container_access")),
        "discovery": _discovery_from_mongo(d.get("discovery")),
        "origin": d.get("origin", "manual"),
        "last_check": d.get("last_check"),
        "last_check_message": d.get("last_check_message", ""),
        "consecutive_failures": int(d.get("consecutive_failures") or 0),
        # Whether the master-node still owes an answer, without exposing the
        # timestamps the two sides use to coordinate.
        "check_pending": d.get("check_requested_at") is not None,
        "discovery_pending": d.get("discovery_requested_at") is not None,
        "created_time": d.get("created_time"),
        "updated_time": d.get("updated_time"),
    }


def _access_from_mongo(access: Optional[dict]) -> Optional[HostAccessDto]:
    if not access:
        return None
    return {
        "address": access.get("address", ""),
        "port": int(access.get("port") or 22),
        "username": access.get("username", ""),
        "auth_method": access.get("auth_method", "key"),
        "secret": redact(access.get("secret")) or {"has_secret": False},
        "key_fingerprint": access.get("key_fingerprint"),
        "host_key_fingerprint": access.get("host_key_fingerprint"),
    }


def _container_access_from_mongo(access: Optional[dict]) -> ContainerAccessDto:
    access = access or {}
    out: ContainerAccessDto = {
        "username": access.get("username", "root"),
        "port": int(access.get("port") or 22),
        "address_mode": access.get("address_mode", "container_ip"),
        "auth_method": access.get("auth_method", "password"),
        "secret": redact(access.get("secret")) or {"has_secret": False},
    }
    if access.get("remote_dir"):
        out["remote_dir"] = access["remote_dir"]
    return out


def _discovery_from_mongo(discovery: Optional[dict]) -> DiscoveryDto:
    discovery = discovery or {}
    return {
        "mode": discovery.get("mode", "docker"),
        "label_filter": discovery.get("label_filter", ""),
        "name_prefix": discovery.get("name_prefix"),
        "endpoint": discovery.get("endpoint", ""),
        "last_discovery": discovery.get("last_discovery"),
        "discovered_count": int(discovery.get("discovered_count") or 0),
        "last_discovery_message": discovery.get("last_discovery_message", ""),
    }


def resource_from_mongo(
    doc: dict, host_name: str = "", host: Optional[dict] = None
) -> ResourceDto:
    """Describe one worker, including whose observation decides about it.

    ``host`` is needed for ``docker_applies``: two of the three signals in
    that rule live on the host record. Passing ``None`` is honest but partial
    — the worker's own settings still count, the host's do not — so callers
    that have the host should supply it.
    """
    id_str, d = pop_id(doc)
    container = d.get("container") or {}
    probe = d.get("probe") or {}
    connection = d.get("connection") or {}
    out: ResourceDto = {
        "id": id_str,
        "host_id": oid_to_str(d.get("host_id")),
        "name": d.get("name", ""),
        "status": d.get("status", ""),
        "enabled": bool(d.get("enabled", False)),
        "container": {
            "id": container.get("id", ""),
            "name": container.get("name", ""),
            "image": container.get("image", ""),
            "state": container.get("state", ""),
            "checked_at": container.get("checked_at"),
        },
        "endpoint": dict(d.get("endpoint") or {}),
        # The worker's own overrides, with the credential described rather
        # than returned — same rule as the host's.
        "connection": {
            k: v for k, v in {
                "username": connection.get("username"),
                "port": connection.get("port"),
                "auth_method": connection.get("auth_method"),
                "secret": redact(connection.get("secret")),
            }.items() if v is not None
        },
        "remote_dir": d.get("remote_dir", ""),
        "probe": {
            "ssh_ok": bool(probe.get("ssh_ok", False)),
            "java_version": probe.get("java_version"),
            "cooja_jar_sha256": probe.get("cooja_jar_sha256"),
            "checked_at": probe.get("checked_at"),
            "message": probe.get("message", ""),
        },
        "current_simulation_id": oid_to_str(d.get("current_simulation_id")) or None,
        "remote_pid": d.get("remote_pid"),
        "reserved_at": d.get("reserved_at"),
        "started_at": d.get("started_at"),
        "origin": d.get("origin", ""),
        "last_check": d.get("last_check"),
        "last_check_message": d.get("last_check_message", ""),
        "docker_applies": not operator_configured(host or {}, d),
    }
    if host_name:
        out["host_name"] = host_name
    return out


def occupancy_from_snapshot(row: dict) -> HostOccupancyDto:
    """One row of the inventory snapshot the scheduler reads."""
    id_str, d = pop_id(row)
    return {
        "id": id_str,
        "name": d.get("name", ""),
        "type": d.get("type", ""),
        "status": d.get("status", ""),
        "enabled": bool(d.get("enabled", False)),
        "schedulable": bool(d.get("schedulable", False)),
        "resource_total": int(d.get("resource_total") or 0),
        "free": int(d.get("free") or 0),
        "active": int(d.get("active") or 0),
        "unavailable": int(d.get("unavailable") or 0),
        "disabled": int(d.get("disabled") or 0),
        "last_check": d.get("last_check"),
        "last_check_message": d.get("last_check_message", ""),
    }


def overview_from_snapshot(
    snapshot: list[dict], cooja_builds: Optional[list[str]] = None
) -> OverviewDto:
    """Totals and per-host occupancy for the Infrastructure page."""
    hosts = [occupancy_from_snapshot(dict(row)) for row in snapshot]
    operational = [h for h in hosts if h["status"] in [
        s.value for s in SCHEDULABLE_HOST_STATUSES]]
    builds = sorted({b for b in (cooja_builds or []) if b})
    return {
        "totals": {
            "hosts": len(hosts),
            "hosts_operational": len(operational),
            "resources": sum(h["resource_total"] for h in hosts),
            "free": sum(h["free"] for h in hosts),
            "active": sum(h["active"] for h in hosts),
            "unavailable": sum(h["unavailable"] for h in hosts),
            "disabled": sum(h["disabled"] for h in hosts),
        },
        "hosts": hosts,
        # More than one Cooja build in the pool means a generation can be
        # split across different simulators, which can show up as a host
        # effect in the results.
        "heterogeneous": len(builds) > 1,
        "cooja_builds": builds,
    }


def public_key_from_mongo(doc: dict) -> PublicKeyDto:
    return {
        "key_id": doc.get("key_id", ""),
        "alg": doc.get("alg", ""),
        "public_key": doc.get("public_key", ""),
        "fingerprint": doc.get("fingerprint", ""),
        "created_at": doc.get("created_at"),
        "retired_key_ids": list(doc.get("retired_key_ids") or []),
    }


# ── Writes ───────────────────────────────────────────────────────────────────

_OBSERVED_RESOURCE_FIELDS = frozenset({
    "status", "container", "probe", "origin", "current_simulation_id",
    "attempt_id", "remote_pid", "reserved_at", "started_at", "last_check",
    "last_check_message", "created_time", "updated_time", "_id", "id",
    "host_id",
})


def resource_to_mongo(dto: dict, *, for_update: bool = False) -> dict:
    """Build the fields of a manually registered worker.

    The point of this mapper: a worker states where its sshd answers and, if
    it differs from its host's default, how to authenticate to it. Everything
    else about a worker is measured.
    """
    d = deepcopy(dict(dto))
    _refuse_plaintext(d, where="worker")

    rejected = sorted(set(d) & _OBSERVED_RESOURCE_FIELDS)
    if rejected:
        raise PlaintextRefused(
            "these fields are measured by the master-node and cannot be set: "
            + ", ".join(rejected)
        )

    doc: dict[str, Any] = {}
    if "name" in d:
        name = str(d["name"]).strip()
        if not name:
            raise PlaintextRefused("name cannot be empty")
        doc["name"] = name
    if "enabled" in d:
        doc["enabled"] = bool(d["enabled"])
    if "remote_dir" in d and d["remote_dir"]:
        doc["remote_dir"] = str(d["remote_dir"])

    if "endpoint" in d:
        endpoint = d.get("endpoint") or {}
        address = str(endpoint.get("address") or "").strip()
        if not address:
            raise PlaintextRefused(
                "endpoint.address is required: it is where this worker's sshd "
                "answers, and the master-node will not guess it"
            )
        doc["endpoint"] = {"address": address}
        if endpoint.get("port"):
            doc["endpoint"]["port"] = int(endpoint["port"])

    if "connection" in d:
        doc["connection"] = _connection_to_mongo(d.get("connection") or {})

    if not for_update:
        if not doc.get("name"):
            raise PlaintextRefused("name is required")
        if not doc.get("endpoint"):
            raise PlaintextRefused("endpoint.address is required")
    return doc


def _connection_to_mongo(connection: dict) -> dict:
    """A worker's own SSH parameters. Only what is stated is recorded.

    Absent fields fall back to the host's ``container_access`` at connect
    time, so one host's configuration still covers the common case.
    """
    _refuse_plaintext(connection, where="worker connection")
    out: dict[str, Any] = {"transport": "ssh"}
    if "username" in connection and connection["username"]:
        out["username"] = str(connection["username"]).strip()
    if connection.get("port"):
        out["port"] = int(connection["port"])
    if "auth_method" in connection and connection["auth_method"]:
        out["auth_method"] = str(connection["auth_method"])
    if "secret" in connection:
        out["secret"] = _secret_to_mongo(connection.get("secret"))
    return out


#: The host's nested configuration blocks. Updating one of these must touch
#: only the fields that were sent, which is what ``dotted_update`` is for.
NESTED_HOST_BLOCKS = ("access", "container_access", "discovery")


def dotted_update(payload: dict) -> dict:
    """Turn a nested patch into the dotted paths a partial ``$set`` needs.

    ``{"access": {"port": 2222}}`` becomes ``{"access.port": 2222}``.

    Without this, ``$set`` replaces the whole ``access`` subdocument with the
    two or three fields the client happened to send: a request that only
    changed the port removed the address, the username, the sealed credential
    and the pinned host key, and the host stopped authenticating. One level is
    flattened and no more — ``secret`` is a document that must be written
    whole, not field by field.
    """
    out: dict = {}
    for key, value in payload.items():
        if key in NESTED_HOST_BLOCKS and isinstance(value, dict):
            for inner, inner_value in value.items():
                out[f"{key}.{inner}"] = inner_value
        else:
            out[key] = value
    return out


def host_to_mongo(dto: dict, *, for_update: bool = False) -> dict:
    """Build the document fields a client is allowed to set.

    Raises ``PlaintextRefused`` when a credential was not sealed, and
    ``MalformedEnvelope`` when it was sealed badly. Both are rejections before
    anything is written: the database must never hold a credential in the
    clear, not even briefly.
    """
    d = deepcopy(dict(dto))
    _refuse_plaintext(d, where="host")

    rejected = sorted(set(d) & _OBSERVED_HOST_FIELDS)
    if rejected:
        raise PlaintextRefused(
            "these fields are measured by the master-node and cannot be set: "
            + ", ".join(rejected)
        )

    doc: dict[str, Any] = {}
    if "name" in d:
        name = str(d["name"]).strip()
        if not name:
            raise PlaintextRefused("name cannot be empty")
        doc["name"] = name
    if "enabled" in d:
        doc["enabled"] = bool(d["enabled"])

    if "access" in d:
        doc["access"] = _access_to_mongo(
            d.get("access") or {}, for_update=for_update)
    if "container_access" in d:
        doc["container_access"] = _container_access_to_mongo(
            d.get("container_access") or {}, for_update=for_update)
    if "discovery" in d:
        doc["discovery"] = _discovery_to_mongo(
            d.get("discovery") or {}, for_update=for_update)

    if not for_update:
        # Set once, at creation: only remote hosts are registered through the
        # API, and everything measured starts empty.
        doc["type"] = EnumHostType.REMOTE.value
        doc.setdefault("enabled", True)
        doc["status"] = EnumHostStatus.OFFLINE.value
        doc["origin"] = "manual"
        doc["consecutive_failures"] = 0
        doc["last_check"] = None
        doc["last_check_message"] = ""
        doc["check_requested_at"] = None
        doc["discovery_requested_at"] = None
        doc.setdefault("container_access", _container_access_to_mongo({}))
        doc.setdefault("discovery", _discovery_to_mongo({}))
    return doc


def _access_to_mongo(access: dict, *, for_update: bool = False) -> dict:
    _refuse_plaintext(access, where="access")
    out: dict[str, Any] = {} if for_update else {"transport": "ssh"}
    if "address" in access:
        out["address"] = str(access["address"]).strip()
    if "port" in access:
        out["port"] = int(access["port"])
    if "username" in access:
        out["username"] = str(access["username"]).strip()
    if "auth_method" in access:
        out["auth_method"] = str(access["auth_method"])
    if "secret" in access:
        out["secret"] = _secret_to_mongo(access.get("secret"))
    return out


def _container_access_to_mongo(access: dict, *, for_update: bool = False) -> dict:
    """Hop 2's settings.

    On an update only the fields that were sent are described. The defaults
    below belong to creation: applying them to a patch is what replaced a
    worker credential with ``None`` and reset the address mode, because a
    client changing the port sends nothing else.
    """
    _refuse_plaintext(access, where="container_access")
    if for_update:
        out: dict[str, Any] = {}
        if "username" in access:
            out["username"] = str(access["username"]).strip()
        if "port" in access:
            out["port"] = int(access["port"])
        if "address_mode" in access:
            out["address_mode"] = str(access["address_mode"])
        if "auth_method" in access:
            out["auth_method"] = str(access["auth_method"])
        if "secret" in access:
            # Present and null means "remove it"; absent means "leave it".
            out["secret"] = _secret_to_mongo(access.get("secret"))
        if "remote_dir" in access:
            out["remote_dir"] = str(access["remote_dir"] or "")
        return out
    out = {
        "transport": "ssh",
        "username": str(access.get("username") or "root"),
        "port": int(access.get("port") or 22),
        "address_mode": str(access.get("address_mode") or "container_ip"),
        "auth_method": str(access.get("auth_method") or "password"),
        "secret": _secret_to_mongo(access.get("secret")),
    }
    if access.get("remote_dir"):
        out["remote_dir"] = str(access["remote_dir"])
    return out


def _discovery_to_mongo(discovery: dict, *, for_update: bool = False) -> dict:
    """How this host's workers are enumerated.

    As with the access blocks, an update describes only what was sent. The
    creation defaults include ``last_discovery: None``, which on a patch would
    erase the result of the last inspection along with its count and message.
    """
    rejected = sorted(set(discovery) & _OBSERVED_DISCOVERY_FIELDS)
    if rejected:
        raise PlaintextRefused(
            "these discovery fields are produced by the master-node and cannot "
            "be set: " + ", ".join(rejected)
        )

    def _mode(raw: Any) -> str:
        mode = str(raw or EnumDiscoveryMode.DOCKER.value)
        if mode not in [m.value for m in EnumDiscoveryMode]:
            raise PlaintextRefused(
                f"discovery.mode must be one of "
                f"{', '.join(m.value for m in EnumDiscoveryMode)}, got {mode!r}"
            )
        return mode

    if for_update:
        out: dict[str, Any] = {}
        if "mode" in discovery:
            out["mode"] = _mode(discovery["mode"])
        if "label_filter" in discovery:
            out["label_filter"] = str(
                discovery["label_filter"] or "simlab.type=cooja")
        if "name_prefix" in discovery:
            out["name_prefix"] = discovery["name_prefix"] or None
        return out

    return {
        "mode": _mode(discovery.get("mode")),
        "label_filter": str(discovery.get("label_filter") or "simlab.type=cooja"),
        "name_prefix": discovery.get("name_prefix") or None,
        # Server-derived: how the daemon is reached follows from the host type,
        # and is meaningless when the workers are listed by hand.
        "endpoint": "ssh-cli",
        "last_discovery": None,
    }


def _secret_to_mongo(secret: Any) -> Optional[dict]:
    """Validate a sealed envelope, or refuse it.

    ``None`` is allowed and means "leave it unset", which is how a host whose
    containers use the image's built-in login is registered.
    """
    if secret is None:
        return None
    if not isinstance(secret, dict):
        raise PlaintextRefused(
            "a credential must be a sealed envelope, not a bare value"
        )
    envelope = {
        "alg": secret.get("alg", "crypto_box_seal"),
        "key_id": secret.get("key_id"),
        "ciphertext": secret.get("ciphertext"),
    }
    # The envelope's own validation catches a plaintext smuggled through as
    # base64: it is far too short to be a sealed box.
    validate_envelope(envelope)
    return envelope


def _refuse_plaintext(payload: dict, *, where: str) -> None:
    offending = sorted(set(payload) & _PLAINTEXT_FIELDS)
    if offending:
        raise PlaintextRefused(
            f"{where}: {', '.join(offending)} cannot be sent to this API. "
            "Seal the credential against the master-node's public key "
            "(GET /infra/crypto/public-key) and send it as "
            "`secret: {key_id, ciphertext}`."
        )


__all__ = [
    "MalformedEnvelope",
    "PlaintextRefused",
    "host_from_mongo",
    "host_to_mongo",
    "resource_from_mongo",
    "resource_to_mongo",
    "occupancy_from_snapshot",
    "overview_from_snapshot",
    "public_key_from_mongo",
]
