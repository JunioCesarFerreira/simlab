"""Tests for the infrastructure endpoints.

Two contracts dominate, and both are about what this API must *not* do:

* a credential never comes out of it, and never goes into it unsealed;
* an observation — status, probe results, discovery output — is never written
  by a client, because the GUI would then show a state nobody measured.

The third is that anything needing the container network or the private key is
requested rather than performed, which is why those endpoints answer 202.
"""
import base64

import pytest
from bson import ObjectId

HOST_ID = "507f1f77bcf86cd799430011"
RES_ID = "507f1f77bcf86cd799430021"
SIM_ID = "507f1f77bcf86cd799430031"

# A value long enough to pass for a sealed box: 32 bytes of ephemeral key plus
# a 16-byte MAC, so anything under 49 is definitely not one.
SEALED = base64.b64encode(b"x" * 120).decode()
ENVELOPE = {"alg": "crypto_box_seal", "key_id": "b114dd6d8410a62a", "ciphertext": SEALED}


def sample_host(**overrides) -> dict:
    doc = {
        "_id": ObjectId(HOST_ID),
        "name": "lab-01",
        "type": "remote",
        "enabled": True,
        "status": "online",
        "access": {
            "transport": "ssh", "address": "192.168.0.42", "port": 22,
            "username": "simlab", "auth_method": "key",
            "secret": dict(ENVELOPE),
            "key_fingerprint": "SHA256:abc",
            "host_key_fingerprint": "SHA256:host",
        },
        "container_access": {
            "transport": "ssh", "username": "root", "port": 22,
            "address_mode": "container_ip", "auth_method": "password",
            "secret": None,
        },
        "discovery": {
            "mode": "docker", "label_filter": "simlab.type=cooja",
            "endpoint": "ssh-cli", "last_discovery": None,
            "discovered_count": 2,
        },
        "origin": "manual",
        "last_check": None,
        "last_check_message": "2 container(s) ready",
        "consecutive_failures": 0,
        "check_requested_at": None,
        "discovery_requested_at": None,
    }
    doc.update(overrides)
    return doc


def sample_resource(status="available", connection=None, **overrides) -> dict:
    doc = {
        "_id": ObjectId(RES_ID),
        "host_id": ObjectId(HOST_ID),
        "name": "cooja1",
        "status": status,
        "enabled": True,
        "container": {
            "id": "abc123", "name": "cooja1", "image": "cooja:v1.1",
            "state": "running", "checked_at": None,
        },
        "endpoint": {"address": "172.18.0.5", "port": 22},
        "connection": connection or {},
        "remote_dir": "/opt/contiki-ng/tools/cooja",
        "probe": {
            "ssh_ok": True, "java_version": "21", "cooja_jar_sha256": "a" * 64,
            "checked_at": None, "message": "ready",
        },
        "current_simulation_id": None,
        "remote_pid": None,
        "origin": "discovered",
        "last_check": None,
        "last_check_message": "ready",
    }
    doc.update(overrides)
    return doc


def snapshot_row(**overrides) -> dict:
    row = {
        "_id": ObjectId(HOST_ID), "name": "lab-01", "type": "remote",
        "status": "online", "enabled": True, "schedulable": True,
        "resource_total": 4, "free": 3, "active": 1, "unavailable": 0,
        "disabled": 0, "last_check": None, "last_check_message": "ok",
    }
    row.update(overrides)
    return row


@pytest.fixture
def infra(mock_factory):
    """Wire the repositories the infrastructure endpoints use."""
    mock_factory.host_repo.find_all.return_value = [sample_host()]
    mock_factory.host_repo.get.return_value = sample_host()
    mock_factory.host_repo.get_by_name.return_value = None
    mock_factory.host_repo.inventory_snapshot.return_value = [snapshot_row()]
    mock_factory.resource_repo.find_all.return_value = [sample_resource()]
    mock_factory.resource_repo.find_by_host.return_value = [sample_resource()]
    mock_factory.resource_repo.get.return_value = sample_resource()
    # No other registration points at the same container. A MagicMock's
    # default answer here is truthy, which would refuse every registration —
    # the sort of thing a permissive stand-in turns into a confusing failure
    # rather than a clear one.
    mock_factory.resource_repo.conflicting_target.return_value = None
    mock_factory.infra_crypto_repo.get_active.return_value = {
        "key_id": "b114dd6d8410a62a", "alg": "crypto_box_seal",
        "public_key": "cHVibGlj", "fingerprint": "SHA256:abc",
        "created_at": None, "retired_key_ids": [],
    }
    return mock_factory


# ── No credential ever comes out ─────────────────────────────────────────────

class TestSecretsNeverLeave:
    def test_a_host_describes_its_credential_without_returning_it(self, client, infra):
        body = client.get(f"/api/v1/infra/hosts/{HOST_ID}").json()

        secret = body["host"]["access"]["secret"]
        assert secret["has_secret"] is True
        assert secret["key_id"] == "b114dd6d8410a62a"
        # Handing the ciphertext back would invite offline attacks the moment
        # a private key leaked.
        assert "ciphertext" not in secret

    def test_no_ciphertext_appears_anywhere_in_a_listing(self, client, infra):
        raw = client.get("/api/v1/infra/hosts").text
        assert SEALED not in raw
        assert "ciphertext" not in raw

    def test_a_host_without_a_credential_says_so(self, client, infra):
        body = client.get(f"/api/v1/infra/hosts/{HOST_ID}").json()
        assert body["host"]["container_access"]["secret"] == {"has_secret": False}

    def test_the_public_key_is_served_because_it_is_public(self, client, infra):
        body = client.get("/api/v1/infra/crypto/public-key").json()
        assert body["public_key"] == "cHVibGlj"
        assert body["key_id"] == "b114dd6d8410a62a"

    def test_a_missing_public_key_says_what_to_do(self, client, infra):
        infra.infra_crypto_repo.get_active.return_value = None
        response = client.get("/api/v1/infra/crypto/public-key")
        assert response.status_code == 503
        assert "master-node publishes it" in response.json()["detail"]


# ── No credential goes in unsealed ───────────────────────────────────────────

class TestPlaintextIsRefused:
    def test_a_sealed_credential_is_accepted(self, client, infra):
        infra.host_repo.insert.return_value = ObjectId(HOST_ID)
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "auth_method": "key", "secret": ENVELOPE},
        })
        assert response.status_code == 201
        stored = infra.host_repo.insert.call_args.args[0]
        assert stored["access"]["secret"]["ciphertext"] == SEALED

    @pytest.mark.parametrize("field", ["password", "private_key", "passphrase"])
    def test_a_plaintext_field_is_refused_by_name(self, client, infra, field):
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab", field: "hunter2"},
        })
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert field in detail
        assert "public-key" in detail          # tells the client what to do instead
        infra.host_repo.insert.assert_not_called()

    def test_a_password_base64d_by_hand_is_refused(self, client, infra):
        # The realistic mistake: valid base64, but far too short to be a
        # sealed box. Nothing is written.
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {
                "address": "10.0.0.5", "username": "simlab",
                "secret": {"alg": "crypto_box_seal", "key_id": "0" * 16,
                           "ciphertext": base64.b64encode(b"hunter2").decode()},
            },
        })
        assert response.status_code == 422
        assert "not sealed" in response.json()["detail"]
        infra.host_repo.insert.assert_not_called()

    def test_a_bare_string_credential_is_refused(self, client, infra):
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "secret": "hunter2"},
        })
        assert response.status_code == 422
        infra.host_repo.insert.assert_not_called()

    def test_a_wrong_algorithm_is_refused(self, client, infra):
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "secret": {"alg": "aes-gcm", "key_id": "0" * 16,
                                  "ciphertext": SEALED}},
        })
        assert response.status_code == 422


# ── No observation goes in ───────────────────────────────────────────────────

class TestObservationsAreNotWritable:
    @pytest.mark.parametrize("field,value", [
        ("status", "online"),
        ("last_check_message", "all good"),
        ("consecutive_failures", 0),
        ("origin", "bootstrap"),
        ("type", "local"),
    ])
    def test_a_measured_field_is_refused(self, client, infra, field, value):
        # Accepting these would let a client make the GUI claim a state
        # nobody measured.
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "secret": ENVELOPE},
            field: value,
        })
        assert response.status_code == 422
        assert field in response.json()["detail"]

    def test_the_discovery_mode_is_an_operator_choice(self, client, infra):
        # Where a host's workers come from is a decision, not an observation.
        # A shared host whose SSH user cannot reach Docker needs `manual`.
        infra.host_repo.insert.return_value = ObjectId(HOST_ID)
        response = client.post("/api/v1/infra/hosts", json={
            "name": "vm7",
            "access": {"address": "10.0.0.9", "username": "jcf",
                       "auth_method": "password", "secret": ENVELOPE},
            "discovery": {"mode": "manual"},
        })
        assert response.status_code == 201
        assert infra.host_repo.insert.call_args.args[0]["discovery"]["mode"] == "manual"

    def test_an_unknown_discovery_mode_is_refused(self, client, infra):
        response = client.post("/api/v1/infra/hosts", json={
            "name": "vm7",
            "access": {"address": "10.0.0.9", "username": "jcf",
                       "auth_method": "password", "secret": ENVELOPE},
            "discovery": {"mode": "telepathy"},
        })
        assert response.status_code == 422
        assert "discovery.mode must be one of" in response.json()["detail"]

    def test_discovery_results_are_refused(self, client, infra):
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "secret": ENVELOPE},
            "discovery": {"label_filter": "x", "discovered_count": 99},
        })
        assert response.status_code == 422
        assert "discovered_count" in response.json()["detail"]

    def test_a_new_host_starts_unmeasured(self, client, infra):
        infra.host_repo.insert.return_value = ObjectId(HOST_ID)
        client.post("/api/v1/infra/hosts", json={
            "name": "lab-02",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "secret": ENVELOPE},
        })
        stored = infra.host_repo.insert.call_args.args[0]
        assert stored["status"] == "offline"
        assert stored["type"] == "remote"
        assert stored["last_check"] is None

    def test_only_enabled_can_be_written_on_a_resource(self, client, infra):
        response = client.put(
            f"/api/v1/infra/resources/{RES_ID}", json={"enabled": False})
        assert response.status_code == 200
        infra.resource_repo.set_enabled.assert_called_once()
        assert infra.resource_repo.set_enabled.call_args.args[1] is False

    def test_a_resource_update_without_enabled_is_refused(self, client, infra):
        response = client.put(f"/api/v1/infra/resources/{RES_ID}", json={})
        assert response.status_code == 422


# ── Deleting a host that is in use ───────────────────────────────────────────

class TestDeleteGuard:
    """The condition travels with the write.

    Counting the busy ones in the endpoint and deleting afterwards left a
    window the scheduler could reserve in — and the delete then removed the
    record of a live run, with no way to find it again because the reservation
    went with it. So the repository refuses, and the endpoint reports what it
    said.
    """

    def test_a_host_with_a_running_simulation_is_refused(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (False, ["cooja1"])

        response = client.delete(f"/api/v1/infra/hosts/{HOST_ID}")

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "cooja1" in detail           # names what blocked it
        assert "drain" in detail            # says what to do instead

    def test_an_idle_host_is_deleted(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (True, [])

        response = client.delete(f"/api/v1/infra/hosts/{HOST_ID}")

        assert response.status_code == 204
        infra.host_repo.delete_if_drained.assert_called_once()

    def test_a_host_that_vanished_meanwhile_is_a_404(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (False, [])

        assert client.delete(f"/api/v1/infra/hosts/{HOST_ID}").status_code == 404

    def test_disabling_a_busy_host_is_allowed(self, client, infra):
        # Disabling drains rather than interrupts, so it must work while busy.
        infra.resource_repo.find_by_host.return_value = [sample_resource(status="busy")]
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/disable")
        assert response.status_code == 200
        infra.host_repo.set_enabled.assert_called_once()
        assert infra.host_repo.set_enabled.call_args.args[1] is False


# ── Requests, not actions ────────────────────────────────────────────────────

class TestRequests:
    def test_a_check_is_recorded_and_answered_202(self, client, infra):
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/check")

        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] is True
        # The API cannot do it: no route to the containers, no private key.
        assert "master-node performs the check" in body["detail"]
        infra.host_repo.request_check.assert_called_once()

    def test_a_discovery_is_recorded_and_answered_202(self, client, infra):
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/discover")
        assert response.status_code == 202
        infra.host_repo.request_discovery.assert_called_once()

    def test_a_pending_request_is_visible_without_its_timestamp(self, client, infra):
        from datetime import datetime
        infra.host_repo.get.return_value = sample_host(
            check_requested_at=datetime(2026, 1, 1))

        body = client.get(f"/api/v1/infra/hosts/{HOST_ID}").json()["host"]

        assert body["check_pending"] is True
        assert "check_requested_at" not in body


# ── Overview ─────────────────────────────────────────────────────────────────

class TestOverview:
    def test_totals_come_from_the_scheduler_snapshot(self, client, infra):
        infra.host_repo.inventory_snapshot.return_value = [
            snapshot_row(resource_total=8, free=6, active=2),
            snapshot_row(_id=ObjectId(), name="local", type="local",
                         resource_total=2, free=0, active=2, unavailable=0),
        ]
        body = client.get("/api/v1/infra/overview").json()

        assert body["totals"]["hosts"] == 2
        assert body["totals"]["resources"] == 10
        assert body["totals"]["free"] == 6
        assert body["totals"]["active"] == 4

    def test_capacity_and_eligibility_are_reported_separately(self, client, infra):
        # An unreachable host still has its containers; folding the two
        # together would under-report capacity that is merely out of contact.
        infra.host_repo.inventory_snapshot.return_value = [
            snapshot_row(status="offline", schedulable=False, free=4)]
        body = client.get("/api/v1/infra/overview").json()
        assert body["totals"]["free"] == 4
        assert body["hosts"][0]["schedulable"] is False
        assert body["totals"]["hosts_operational"] == 0

    def test_a_mixed_pool_is_flagged(self, client, infra):
        # The same generation split across different Cooja builds can show up
        # as a host effect in the results.
        infra.resource_repo.find_all.return_value = [
            sample_resource(), sample_resource(
                _id=ObjectId(),
                probe={"ssh_ok": True, "cooja_jar_sha256": "b" * 64}),
        ]
        body = client.get("/api/v1/infra/overview").json()
        assert body["heterogeneous"] is True
        assert len(body["cooja_builds"]) == 2

    def test_a_uniform_pool_is_not_flagged(self, client, infra):
        body = client.get("/api/v1/infra/overview").json()
        assert body["heterogeneous"] is False


# ── Reads ────────────────────────────────────────────────────────────────────

class TestReads:
    def test_a_resource_reports_both_availability_levels(self, client, infra):
        body = client.get(f"/api/v1/infra/resources/{RES_ID}").json()
        # "stopped" and "up but no Cooja" need different fixes, so the GUI has
        # to be able to tell them apart.
        assert body["container"]["state"] == "running"
        assert body["probe"]["ssh_ok"] is True
        assert body["probe"]["cooja_jar_sha256"] == "a" * 64

    def test_a_resource_carries_its_host_name(self, client, infra):
        body = client.get("/api/v1/infra/resources").json()
        assert body[0]["host_name"] == "lab-01"

    def test_an_idle_container_reports_no_simulation(self, client, infra):
        body = client.get(f"/api/v1/infra/resources/{RES_ID}/simulation").json()
        assert body["simulation_id"] is None

    def test_a_busy_container_reports_what_it_is_running(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(
            status="busy", current_simulation_id=ObjectId(SIM_ID),
            attempt_id="att-1")
        infra.simulation_repo.get.return_value = {
            "_id": ObjectId(SIM_ID), "status": "Running",
            "experiment_id": ObjectId("507f1f77bcf86cd799439011"),
            "generation_id": None,
        }

        body = client.get(f"/api/v1/infra/resources/{RES_ID}/simulation").json()

        assert body["simulation_id"] == SIM_ID
        assert body["status"] == "Running"
        assert body["attempt_id"] == "att-1"

    def test_a_missing_host_is_404(self, client, infra):
        infra.host_repo.get.return_value = None
        assert client.get(f"/api/v1/infra/hosts/{HOST_ID}").status_code == 404

    def test_a_missing_resource_is_404(self, client, infra):
        infra.resource_repo.get.return_value = None
        assert client.get(f"/api/v1/infra/resources/{RES_ID}").status_code == 404


# ── Updates ──────────────────────────────────────────────────────────────────

class TestUpdates:
    def test_only_the_fields_sent_are_touched(self, client, infra):
        # So a credential is left alone unless a new sealed one is supplied.
        client.put(f"/api/v1/infra/hosts/{HOST_ID}", json={"enabled": False})
        payload = infra.host_repo.update.call_args.args[1]
        assert set(payload) == {"enabled"}

    def test_a_duplicate_name_is_refused(self, client, infra):
        infra.host_repo.get_by_name.return_value = sample_host(_id=ObjectId())
        response = client.put(f"/api/v1/infra/hosts/{HOST_ID}", json={"name": "taken"})
        assert response.status_code == 409

    def test_an_empty_update_is_refused(self, client, infra):
        assert client.put(f"/api/v1/infra/hosts/{HOST_ID}", json={}).status_code == 422

    def test_creating_a_duplicate_name_is_refused(self, client, infra):
        infra.host_repo.get_by_name.return_value = sample_host()
        response = client.post("/api/v1/infra/hosts", json={
            "name": "lab-01",
            "access": {"address": "10.0.0.5", "username": "simlab",
                       "secret": ENVELOPE},
        })
        assert response.status_code == 409


# ── Workers configured one by one ────────────────────────────────────────────

class TestManualWorkers:
    """Registering a worker by hand, with its own SSH parameters.

    The case this exists for: a shared host where the SSH user cannot reach
    the Docker daemon, so nothing can be discovered, and whose workers sit
    behind published ports the operator already knows.
    """

    def test_a_worker_is_registered_with_its_address(self, client, infra):
        infra.resource_repo.find_by_host_and_name.return_value = None
        infra.resource_repo.insert_manual.return_value = ObjectId(RES_ID)

        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/resources", json={
            "name": "vm7-cooja1",
            "endpoint": {"address": "127.0.0.1", "port": 2222},
        })

        assert response.status_code == 201
        stored = infra.resource_repo.insert_manual.call_args.args[0]
        assert stored["name"] == "vm7-cooja1"
        assert stored["endpoint"] == {"address": "127.0.0.1", "port": 2222}
        assert stored["host_id"] == ObjectId(HOST_ID)

    def test_its_own_ssh_parameters_are_stored(self, client, infra):
        infra.resource_repo.find_by_host_and_name.return_value = None
        infra.resource_repo.insert_manual.return_value = ObjectId(RES_ID)

        client.post(f"/api/v1/infra/hosts/{HOST_ID}/resources", json={
            "name": "vm7-cooja1",
            "endpoint": {"address": "127.0.0.1", "port": 2222},
            "connection": {"username": "cooja", "port": 2222,
                           "auth_method": "password", "secret": ENVELOPE},
        })

        connection = infra.resource_repo.insert_manual.call_args.args[0]["connection"]
        assert connection["username"] == "cooja"
        assert connection["auth_method"] == "password"
        assert connection["secret"]["ciphertext"] == SEALED

    def test_an_address_is_required(self, client, infra):
        # The master-node will not guess where a worker's sshd answers.
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/resources", json={
            "name": "vm7-cooja1",
        })
        assert response.status_code == 422
        assert "endpoint.address is required" in response.json()["detail"]

    def test_a_plaintext_credential_is_refused(self, client, infra):
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/resources", json={
            "name": "vm7-cooja1",
            "endpoint": {"address": "127.0.0.1"},
            "connection": {"username": "root", "password": "hunter2"},
        })
        assert response.status_code == 422
        assert "password" in response.json()["detail"]
        infra.resource_repo.insert_manual.assert_not_called()

    def test_measured_fields_are_refused(self, client, infra):
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/resources", json={
            "name": "vm7-cooja1",
            "endpoint": {"address": "127.0.0.1"},
            "status": "available",
            "probe": {"ssh_ok": True},
        })
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "status" in detail and "probe" in detail

    def test_a_duplicate_name_on_the_same_host_is_refused(self, client, infra):
        infra.resource_repo.find_by_host_and_name.return_value = sample_resource()
        response = client.post(f"/api/v1/infra/hosts/{HOST_ID}/resources", json={
            "name": "cooja1", "endpoint": {"address": "127.0.0.1"},
        })
        assert response.status_code == 409

    def test_editing_only_the_port_keeps_the_credential(self, client, infra):
        # Otherwise changing a port would silently drop the stored secret.
        infra.resource_repo.get.return_value = sample_resource(
            connection={"transport": "ssh", "username": "cooja",
                        "auth_method": "password", "secret": dict(ENVELOPE)})

        client.put(f"/api/v1/infra/resources/{RES_ID}/connection",
                   json={"connection": {"port": 2299}})

        merged = infra.resource_repo.update_manual.call_args.args[1]["connection"]
        assert merged["port"] == 2299
        assert merged["username"] == "cooja"
        assert merged["secret"]["ciphertext"] == SEALED

    def test_clearing_an_override_removes_it(self, client, infra):
        """The third intention, which did not exist.

        The form omitted an emptied field, the endpoint merged with what was
        stored, and the old value survived — there was no way to undo an
        override at all. A null now removes the key, rather than storing a
        null: a stored null would still read as the operator having
        configured something, which is what decides whether Docker's view of
        the worker is used.
        """
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"username": "simlab", "port": 2223})

        response = client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"connection": {"username": None}},
        )

        assert response.status_code == 200
        merged = infra.resource_repo.update_manual.call_args.args[1]["connection"]
        assert "username" not in merged
        # What was not mentioned is kept.
        assert merged["port"] == 2223

    def test_an_empty_string_also_removes_it(self, client, infra):
        # What a cleared text field sends.
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"username": "simlab"})

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"connection": {"username": ""}},
        )

        merged = infra.resource_repo.update_manual.call_args.args[1]["connection"]
        assert "username" not in merged

    def test_a_credential_can_be_handed_back_to_the_host(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(
            status="available",
            connection={"secret": {"alg": "crypto_box_seal",
                                   "key_id": "a" * 16, "ciphertext": "x"}})

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"connection": {"secret": None}},
        )

        merged = infra.resource_repo.update_manual.call_args.args[1]["connection"]
        assert "secret" not in merged

    def test_an_override_still_overrides(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(status="available")

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"connection": {"username": "outro"}},
        )

        merged = infra.resource_repo.update_manual.call_args.args[1]["connection"]
        assert merged["username"] == "outro"

    def test_a_discovered_worker_cannot_be_renamed(self, client, infra):
        """Its name *is* the container's, and discovery matches on it.

        Renaming one left the original name with no record, so the next
        discovery inserted it again — two registrations and two reservations
        for one container.
        """
        infra.resource_repo.get.return_value = sample_resource(status="available")

        response = client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"name": "worker-rapido"},
        )

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "found by Docker" in detail
        # Says what to do instead, rather than only refusing.
        assert "connection settings" in detail and "by hand" in detail
        infra.resource_repo.update_manual.assert_not_called()

    def test_changing_a_discovered_workers_connection_is_allowed(self, client, infra):
        # The address is the operator's business; the name is not.
        infra.resource_repo.get.return_value = sample_resource(status="available")

        response = client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"connection": {"port": 2223}},
        )

        assert response.status_code == 200
        infra.resource_repo.update_manual.assert_called_once()

    def test_a_busy_worker_cannot_be_removed(self, client, infra):
        # The repository refuses; the read in the endpoint is only for the
        # message, because by itself it would be a check-then-act.
        infra.resource_repo.get.return_value = sample_resource(status="busy")
        infra.resource_repo.delete_if_free.return_value = False

        response = client.delete(f"/api/v1/infra/resources/{RES_ID}")

        assert response.status_code == 409
        assert "running a simulation" in response.json()["detail"]

    def test_an_idle_worker_is_removed(self, client, infra):
        infra.resource_repo.delete_if_free.return_value = True

        response = client.delete(f"/api/v1/infra/resources/{RES_ID}")

        assert response.status_code == 204
        infra.resource_repo.delete_if_free.assert_called_once()

    def test_a_worker_reserved_between_the_read_and_the_write_is_refused(
        self, client, infra
    ):
        """The race itself: idle when read, taken by the time it is deleted."""
        infra.resource_repo.get.return_value = sample_resource(status="available")
        infra.resource_repo.delete_if_free.return_value = False

        assert client.delete(
            f"/api/v1/infra/resources/{RES_ID}").status_code == 409

    def test_a_workers_credential_is_described_not_returned(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(
            connection={"username": "cooja", "secret": dict(ENVELOPE)})

        body = client.get(f"/api/v1/infra/resources/{RES_ID}").json()

        assert body["connection"]["secret"]["has_secret"] is True
        assert "ciphertext" not in body["connection"]["secret"]
        assert SEALED not in client.get(f"/api/v1/infra/resources/{RES_ID}").text


# ── A host that cannot authenticate is refused at registration ───────────────

class TestCredentialIsRequired:
    """Observed in practice: a host registered with no credential.

    Every later operation then failed with "no credential is stored", far from
    the cause. The registration is where that belongs.
    """

    @pytest.mark.parametrize("method", ["key", "password"])
    def test_registering_without_one_is_refused(self, client, infra, method):
        response = client.post("/api/v1/infra/hosts", json={
            "name": "vm7",
            "access": {"address": "10.0.0.9", "username": "jcf",
                       "auth_method": method},
        })
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "no credential was sent" in detail
        assert "public-key" in detail          # says how to produce one
        infra.host_repo.insert.assert_not_called()

    def test_an_update_is_judged_on_the_merged_result(self, client, infra):
        # Changing only the port must not read as dropping the credential.
        response = client.put(f"/api/v1/infra/hosts/{HOST_ID}", json={
            "access": {"port": 2297},
        })
        assert response.status_code == 200
        infra.host_repo.update.assert_called_once()

    def test_an_update_that_really_drops_it_is_refused(self, client, infra):
        infra.host_repo.get.return_value = sample_host(
            access={"address": "10.0.0.9", "username": "jcf",
                    "auth_method": "key", "secret": None})
        response = client.put(f"/api/v1/infra/hosts/{HOST_ID}", json={
            "access": {"auth_method": "password"},
        })
        assert response.status_code == 422


# ── The API stays outside the trust boundary ─────────────────────────────────

def test_the_api_cannot_open_a_secret():
    """Nothing in the API's import graph can decrypt a credential.

    The rest-api environment has no PyNaCl, deliberately. A stray import of
    ``pylib.crypto.sealed`` would both break its startup and quietly widen the
    trust boundary, so this asserts the mappers and endpoints load with the
    crypto library sabotaged.
    """
    import subprocess
    import sys
    from pathlib import Path

    restapi_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys;"
        "sys.modules['nacl'] = None;"
        "sys.modules['nacl.public'] = None;"
        "import api.mappers.infra, api.endpoints.infra;"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(restapi_root),
        env={"PYTHONPATH": f"{restapi_root}:{restapi_root.parent}"},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ── R09: removing a host is a withdrawal, not an instant ───────────────────

class TestRemovingAHost:
    """The refusal has to say what it left behind.

    Deleting the workers and then the host is several writes. A registration
    that had already read the host could insert a worker between them, and
    the host was deleted on top of it — one orphaned worker, reproduced
    against the real collection. The removal is now a withdrawal first, and a
    refusal leaves the host in service rather than half-removed.
    """

    def test_a_blocked_removal_names_the_worker_and_says_what_happened(
        self, client, infra
    ):
        infra.host_repo.delete_if_drained.return_value = (False, ["cooja2"])

        resp = client.delete(f"/api/v1/infra/hosts/{HOST_ID}")

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "cooja2" in detail
        # The operator has to know the refusal was partial, and that the host
        # is schedulable again.
        assert "left in service" in detail

    def test_a_drained_host_is_removed(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (True, [])

        assert client.delete(f"/api/v1/infra/hosts/{HOST_ID}").status_code == 204

    def test_registering_a_worker_on_a_withdrawing_host_is_refused(
        self, client, infra
    ):
        from pylib.db.repositories.infra import HostWithdrawing

        infra.resource_repo.find_by_host_and_name.return_value = None
        infra.resource_repo.insert_manual.side_effect = HostWithdrawing(
            "host is being removed")

        resp = client.post(
            f"/api/v1/infra/hosts/{HOST_ID}/resources",
            json={"name": "cooja9", "endpoint": {"address": "10.0.0.9"}},
        )

        assert resp.status_code == 409


# ── R08: two names for one worker ──────────────────────────────────────────

class TestOneRegistrationPerWorker:
    def test_a_second_name_for_the_same_target_is_refused(self, client, infra):
        """Two reservations for one run directory: each run's defensive
        cleanup then deletes the other's inputs, and the lock that exists to
        prevent it is held on the wrong document."""
        infra.resource_repo.find_by_host_and_name.return_value = None
        infra.resource_repo.conflicting_target.return_value = {"name": "cooja1"}

        resp = client.post(
            f"/api/v1/infra/hosts/{HOST_ID}/resources",
            json={"name": "cooja1-again",
                  "endpoint": {"address": "172.18.0.5", "port": 22}},
        )

        assert resp.status_code == 409
        assert "cooja1" in resp.json()["detail"]

    def test_the_check_excludes_the_worker_being_edited(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(status="available")

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"endpoint": {"address": "172.18.0.9"}},
        )

        assert infra.resource_repo.conflicting_target.call_args.kwargs[
            "exclude_id"] == sample_resource()["_id"]


# ── R17: an old document's port override ───────────────────────────────────

class TestThePortTheFormShowsIsThePortUsed:
    """Reproduced in review: endpoint 2223, effective connection still 2222.

    An older version of the form wrote the port in both ``endpoint`` and
    ``connection``, and the override wins in the connection plan. Editing
    such a worker changed the endpoint, reported the new port, and went on
    connecting to the old one.
    """

    def test_a_conflicting_override_is_removed(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"port": 2222})

        resp = client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"endpoint": {"address": "10.0.0.7", "port": 2223}},
        )

        assert resp.status_code == 200
        infra.resource_repo.clear_connection_port.assert_called_once_with(
            sample_resource()["_id"])

    def test_the_payload_the_form_actually_sends_also_works(self, client, infra):
        """R17, reproduced: the form sends a `connection` block.

        The decision used to be made *after* merging that block with the
        stored one, which reintroduced the old port — so the condition never
        held and the override survived. Endpoint saved as 2223, effective
        port still 2222. A test that sent only the endpoint did not represent
        an edit through the form.
        """
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"port": 2222})

        resp = client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={
                "endpoint": {"address": "10.0.0.7", "port": 2223},
                # Exactly what the form posts for "inherit these".
                "connection": {"username": None, "auth_method": None,
                               "secret": None},
            },
        )

        assert resp.status_code == 200
        infra.resource_repo.clear_connection_port.assert_called_once_with(
            sample_resource()["_id"])
        # And the override is not written back by the merge either.
        payload = infra.resource_repo.update_manual.call_args.args[1]
        assert "port" not in payload.get("connection", {})

    def test_an_override_sent_in_the_same_request_is_kept(self, client, infra):
        """A client that means to keep one says so."""
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"port": 2222})

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"endpoint": {"address": "10.0.0.7", "port": 2223},
                  "connection": {"port": 2299}},
        )

        infra.resource_repo.clear_connection_port.assert_not_called()
        payload = infra.resource_repo.update_manual.call_args.args[1]
        assert payload["connection"]["port"] == 2299

    def test_a_matching_override_is_left_alone(self, client, infra):
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"port": 2223})

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"endpoint": {"address": "10.0.0.7", "port": 2223}},
        )

        infra.resource_repo.clear_connection_port.assert_not_called()

    def test_an_edit_that_does_not_touch_the_endpoint_changes_nothing(
        self, client, infra
    ):
        infra.resource_repo.get.return_value = sample_resource(
            status="available", connection={"port": 2222})

        client.put(
            f"/api/v1/infra/resources/{RES_ID}/connection",
            json={"connection": {"username": "root"}},
        )

        infra.resource_repo.clear_connection_port.assert_not_called()


# ── R22: a configuration edit invalidates more than the measurement ───────

class TestEditingAccessInvalidatesTheEvidence:
    def test_the_configuration_version_is_bumped(self, client, infra):
        """A probe already running finishes after the clear and would write a
        positive result for a path that no longer exists. The version is what
        lets that result be recognised as describing something else."""
        client.put(
            f"/api/v1/infra/hosts/{HOST_ID}",
            json={"container_access": {"port": 2223}},
        )

        infra.host_repo.bump_config_version.assert_called_once_with(
            sample_host()["_id"])

    def test_an_edit_that_touches_no_access_block_does_not_bump_it(
        self, client, infra
    ):
        client.put(f"/api/v1/infra/hosts/{HOST_ID}", json={"name": "lab-renamed"})

        infra.host_repo.bump_config_version.assert_not_called()


class TestAHostRemovedWithStrandedWorkers:
    """The one surprising outcome, and it is reported rather than hidden.

    A placement that started from a snapshot taken before the withdrawal can
    reserve a worker during it. The final sweep leaves that worker alone —
    deleting it would take a live run with it — so the host is gone and the
    worker is not.
    """

    def test_the_names_come_back_with_a_200(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (True, ["cooja9"])

        resp = client.delete(f"/api/v1/infra/hosts/{HOST_ID}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["deleted"] is True
        assert body["stranded"] == ["cooja9"]
        assert "running simulation" in body["detail"]

    def test_a_clean_removal_is_still_a_204(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (True, [])

        assert client.delete(f"/api/v1/infra/hosts/{HOST_ID}").status_code == 204

    def test_a_refusal_is_still_a_409(self, client, infra):
        infra.host_repo.delete_if_drained.return_value = (False, ["cooja1"])

        resp = client.delete(f"/api/v1/infra/hosts/{HOST_ID}")

        assert resp.status_code == 409
        assert "left in service" in resp.json()["detail"]
