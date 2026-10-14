"""Editing a host must not lose what the client did not send.

The defect this file exists for was invisible to the tests that already
covered the route: they asserted the API answered 200. It did — and the write
replaced the whole ``access`` subdocument with the two fields the client had
sent, so changing a port removed the address, the username, the sealed
credential and the pinned host key. The host stopped authenticating.

So these drive the real chain — endpoint, mapper and ``HostRepository`` — over
a store that actually applies ``$set``, and then **read the document back**.
Asserting on a call is not enough when the question is what survives.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from pylib.db.repositories.infra import HostRepository

HOST_ID = ObjectId("507f1f77bcf86cd799430031")
BASE = "/api/v1/infra"

# A sealed box is 48 bytes of overhead plus the payload, so a short string is
# refused by the envelope validation — these are long enough to be real.
_CIPHERTEXT = "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ=="
SEALED = {"alg": "crypto_box_seal", "key_id": "a" * 16, "ciphertext": _CIPHERTEXT}
LAST_DISCOVERY = datetime(2026, 10, 12, 20, 12, 17)

STORED_HOST = {
    "_id": HOST_ID,
    "name": "lab-01",
    "type": "remote",
    "enabled": True,
    "status": "online",
    "origin": "manual",
    "consecutive_failures": 0,
    "access": {
        "transport": "ssh",
        "address": "192.168.0.42",
        "port": 2222,
        "username": "simlab",
        "auth_method": "key",
        "secret": SEALED,
        "host_key_fingerprint": "SHA256:kept",
    },
    "container_access": {
        "transport": "ssh",
        "username": "root",
        "port": 22,
        "address_mode": "explicit",
        "auth_method": "password",
        "secret": SEALED,
        "remote_dir": "/opt/contiki-ng/tools/cooja",
    },
    "discovery": {
        "mode": "manual",
        "label_filter": "simlab.type=cooja-remote",
        "name_prefix": None,
        "endpoint": "ssh-cli",
        "last_discovery": LAST_DISCOVERY,
        "discovered_count": 4,
        "last_discovery_message": "4 container(s)",
    },
}


def _apply(doc: dict, update: dict) -> None:
    """Apply a ``$set`` the way MongoDB does, dotted paths included."""
    for key, value in (update.get("$set") or {}).items():
        target = doc
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


class SetApplyingCollection:
    """One document, and a ``$set`` that really lands on it."""

    def __init__(self, doc: dict):
        self.doc = deepcopy(doc)

    def create_index(self, *a, **k):
        return None

    def find_one(self, flt, *a, **k):
        if "name" in flt and flt["name"] != self.doc.get("name"):
            return None
        return deepcopy(self.doc)

    def update_one(self, flt, update, **k):
        _apply(self.doc, update)
        return MagicMock(matched_count=1, modified_count=1)


@pytest.fixture
def stored() -> SetApplyingCollection:
    return SetApplyingCollection(STORED_HOST)


@pytest.fixture
def client(stored) -> TestClient:
    """The real endpoint, mapper and repository over a store that applies $set."""
    from main import app
    from api.dependencies import get_factory
    from api.auth import get_api_key

    collection = {"hosts": stored, "cooja_resources": MagicMock()}
    connection = MagicMock()

    @contextmanager
    def connect():
        yield collection

    connection.connect = connect

    factory = MagicMock()
    factory.host_repo = HostRepository(connection)
    app.dependency_overrides[get_factory] = lambda: factory
    app.dependency_overrides[get_api_key] = lambda: "api-password"

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


class TestEditingAHost:
    def test_changing_the_port_keeps_the_credential(self, client, stored):
        resp = client.put(f"{BASE}/hosts/{HOST_ID}", json={"access": {"port": 2223}})

        assert resp.status_code == 200
        access = stored.doc["access"]
        assert access["port"] == 2223
        assert access["secret"] == SEALED
        assert access["address"] == "192.168.0.42"
        assert access["username"] == "simlab"

    def test_changing_the_port_keeps_the_pinned_host_key(self, client, stored):
        """The pin is the whole of the TOFU guarantee.

        It is never even sent by the client, so a write that replaces the
        subdocument always loses it — and the next connection would learn a
        new key as though it were the first contact.
        """
        client.put(f"{BASE}/hosts/{HOST_ID}", json={"access": {"port": 2223}})

        assert stored.doc["access"]["host_key_fingerprint"] == "SHA256:kept"

    def test_renaming_keeps_both_hops(self, client, stored):
        client.put(f"{BASE}/hosts/{HOST_ID}", json={"name": "lab-01-renamed"})

        assert stored.doc["name"] == "lab-01-renamed"
        assert stored.doc["access"]["secret"] == SEALED
        assert stored.doc["container_access"]["secret"] == SEALED

    def test_editing_hop_1_leaves_hop_2_alone(self, client, stored):
        """The second hop used to be rebuilt from defaults on any update."""
        client.put(f"{BASE}/hosts/{HOST_ID}", json={"access": {"address": "10.0.0.9"}})

        container = stored.doc["container_access"]
        assert container["secret"] == SEALED
        assert container["address_mode"] == "explicit"
        assert container["remote_dir"] == "/opt/contiki-ng/tools/cooja"

    def test_editing_hop_2_port_keeps_its_credential(self, client, stored):
        client.put(
            f"{BASE}/hosts/{HOST_ID}", json={"container_access": {"port": 2022}})

        container = stored.doc["container_access"]
        assert container["port"] == 2022
        assert container["secret"] == SEALED
        assert container["username"] == "root"
        assert container["address_mode"] == "explicit"

    def test_changing_the_discovery_mode_keeps_the_last_result(self, client, stored):
        """`last_discovery: None` is a creation default, not an update."""
        client.put(f"{BASE}/hosts/{HOST_ID}", json={"discovery": {"mode": "docker"}})

        discovery = stored.doc["discovery"]
        assert discovery["mode"] == "docker"
        assert discovery["last_discovery"] == LAST_DISCOVERY
        assert discovery["discovered_count"] == 4
        assert discovery["label_filter"] == "simlab.type=cooja-remote"

    def test_a_new_sealed_credential_does_replace_the_old_one(self, client, stored):
        fresh = {"alg": "crypto_box_seal", "key_id": "b" * 16,
                 "ciphertext": "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg=="}

        resp = client.put(
            f"{BASE}/hosts/{HOST_ID}", json={"access": {"secret": fresh}})

        assert resp.status_code == 200
        assert stored.doc["access"]["secret"] == fresh

    def test_an_explicit_null_removes_the_credential(self, client, stored):
        """How a host moves to the image's built-in login.

        Absent means "leave it"; present and null means "remove it". Without
        the distinction there is no way to undo a credential at all.
        """
        resp = client.put(
            f"{BASE}/hosts/{HOST_ID}",
            json={"container_access": {"secret": None}},
        )

        assert resp.status_code == 200
        assert stored.doc["container_access"]["secret"] is None
        assert stored.doc["access"]["secret"] == SEALED

    def test_a_patch_that_would_leave_hop_1_uncredentialed_is_refused(
        self, client, stored
    ):
        resp = client.put(
            f"{BASE}/hosts/{HOST_ID}", json={"access": {"secret": None}})

        assert resp.status_code == 422
        assert stored.doc["access"]["secret"] == SEALED

    def test_plaintext_is_still_refused(self, client, stored):
        resp = client.put(
            f"{BASE}/hosts/{HOST_ID}",
            json={"access": {"secret": {"ciphertext": "hunter2", "key_id": "a" * 16}}},
        )

        assert resp.status_code == 422
        assert stored.doc["access"]["secret"] == SEALED

    def test_measured_fields_are_still_refused(self, client, stored):
        resp = client.put(f"{BASE}/hosts/{HOST_ID}", json={"status": "online"})

        assert resp.status_code == 422
        assert stored.doc["status"] == "online"
