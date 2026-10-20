"""Tests for the local host and resource documents.

The point of this module is that nothing about the local deployment changes:
the containers the environment describes must come out as documents that
resolve to the same SSH targets the master-node used before this feature.

These tests are the guard on that equivalence, and on the one that follows
from it — that `docker compose up` needs no registration step.
"""
from unittest.mock import MagicMock

import pytest

from lib.bootstrap import (
    DEFAULT_COOJA_LABEL,
    LOCAL_HOST_NAME,
    ensure_local_host,
    local_host_document,
    local_resource_documents,
    publish_public_key,
)
from lib.transport.plan import build_plan

REMOTE_DIR = "/opt/contiki-ng/tools/cooja"


def docker_resources(n: int = 3):
    return local_resource_documents(
        hostnames=[f"cooja{i + 1}" for i in range(n)],
        ports=[22] * n,
        remote_dir=REMOTE_DIR,
    )


def host_resources(n: int = 3):
    return local_resource_documents(
        hostnames=["localhost"] * n,
        ports=[2231 + i for i in range(n)],
        remote_dir=REMOTE_DIR,
    )


# ── The local host ───────────────────────────────────────────────────────────

class TestLocalHost:
    def test_has_no_first_hop(self):
        host = local_host_document(is_docker=True)
        # A local host has no host to SSH into; the transport decides on the
        # jump from the type alone.
        assert host["type"] == "local"
        assert host["access"] is None

    def test_in_docker_it_uses_container_names(self):
        # The master-node shares cooja-net, so Docker's DNS answers — which is
        # exactly how it reaches cooja1..N today.
        host = local_host_document(is_docker=True)
        assert host["container_access"]["address_mode"] == "container_name"

    def test_on_the_host_it_uses_published_ports(self):
        # No Docker DNS available, so it goes to the loopback ports — again
        # what it already did.
        host = local_host_document(is_docker=False)
        assert host["container_access"]["address_mode"] == "published_port"

    def test_no_credential_is_stored(self):
        # The image default is resolved at connect time, so the bootstrap
        # never writes a password into the database.
        assert local_host_document(is_docker=True)["container_access"]["secret"] is None

    def test_discovery_targets_the_compose_label(self):
        discovery = local_host_document(is_docker=True)["discovery"]
        assert discovery["label_filter"] == DEFAULT_COOJA_LABEL
        assert discovery["endpoint"] == "socket"

    def test_starts_unmeasured_rather_than_assumed(self):
        """Nothing here claims the host works; the health cycle decides.

        This briefly asserted the opposite. The bootstrap used to mark the
        host ONLINE because nothing measured anything, and an earlier version
        marking it OFFLINE deadlocked the pool outright — observed in
        deployment as 60 simulations Waiting with 8 containers free.

        The protection is no longer the default. It is that a health cycle
        runs synchronously before the dispatcher places any work, so the first
        decision reads a measurement. See
        ``test_health.TestFirstCycle.test_an_unmeasured_pool_becomes_usable``.
        """
        host = local_host_document(is_docker=True)
        assert host["enabled"] is True
        assert host["status"] == "offline"
        assert host["last_check"] is None

    def test_marked_as_bootstrapped(self):
        host = local_host_document(is_docker=True)
        assert host["origin"] == "bootstrap"
        assert host["name"] == LOCAL_HOST_NAME


# ── The seeded resources ─────────────────────────────────────────────────────

class TestLocalResources:
    def test_docker_targets_match_the_previous_behaviour(self):
        host = local_host_document(is_docker=True)
        targets = [build_plan(host, r).target for r in docker_resources(3)]
        assert [(t.address, t.port) for t in targets] == [
            ("cooja1", 22), ("cooja2", 22), ("cooja3", 22),
        ]

    def test_host_targets_match_the_previous_behaviour(self):
        host = local_host_document(is_docker=False)
        targets = [build_plan(host, r).target for r in host_resources(3)]
        assert [(t.address, t.port) for t in targets] == [
            ("localhost", 2231), ("localhost", 2232), ("localhost", 2233),
        ]

    def test_no_plan_is_tunnelled(self):
        host = local_host_document(is_docker=True)
        assert not any(build_plan(host, r).is_tunnelled for r in docker_resources(3))

    def test_names_are_derived_when_the_hostname_is_loopback(self):
        # "localhost" repeated N times is not N distinct names, and the unique
        # index is on (host_id, name).
        assert [r["name"] for r in host_resources(3)] == ["cooja1", "cooja2", "cooja3"]

    def test_names_are_kept_when_they_identify_the_container(self):
        assert [r["name"] for r in docker_resources(2)] == ["cooja1", "cooja2"]

    def test_seeded_resources_wait_to_be_measured(self):
        # Nothing enters the pool on an assertion, including this one: the
        # container has to be running per Docker and answer the SSH probe.
        assert all(r["status"] == "offline" for r in docker_resources(2))
        assert all(r["enabled"] for r in docker_resources(2))

    def test_container_facts_are_not_claimed(self):
        # Nothing has looked at Docker yet, so the container state must not
        # assert that it is running.
        for resource in docker_resources(2):
            assert resource["container"]["state"] == "missing"
            assert resource["container"]["checked_at"] is None
            assert resource["probe"]["ssh_ok"] is False

    def test_remote_dir_is_carried(self):
        assert all(r["remote_dir"] == REMOTE_DIR for r in docker_resources(2))

    def test_host_id_is_attached_only_when_known(self):
        # In-memory use has no host _id yet; the startup upsert supplies it.
        assert "host_id" not in docker_resources(1)[0]
        with_id = local_resource_documents(
            hostnames=["cooja1"], ports=[22], remote_dir=REMOTE_DIR, host_id="HID")
        assert with_id[0]["host_id"] == "HID"

    def test_count_follows_the_environment(self):
        assert len(docker_resources(8)) == 8
        assert len(docker_resources(0)) == 0


# ── Persisting it, without undoing anything ──────────────────────────────────

class FakeMongo:
    def __init__(self, inserted=True):
        self.host_repo = MagicMock()
        self.resource_repo = MagicMock()
        self.infra_crypto_repo = MagicMock()
        self.host_repo.upsert_bootstrap.return_value = "HOST_ID"
        self.resource_repo.insert_seeded.return_value = inserted


class TestEnsureLocalHost:
    def test_the_host_goes_through_the_insert_only_upsert(self):
        mongo = FakeMongo()
        host_id = ensure_local_host(
            mongo, is_docker=True, hostnames=["cooja1"], ports=[22],
            remote_dir=REMOTE_DIR)

        assert host_id == "HOST_ID"
        # Not `insert` and not `update`: a restart must not re-enable a host an
        # operator disabled, nor undo its access configuration.
        mongo.host_repo.upsert_bootstrap.assert_called_once()
        mongo.host_repo.insert.assert_not_called()
        mongo.host_repo.update.assert_not_called()

    def test_every_described_container_is_offered(self):
        mongo = FakeMongo()
        ensure_local_host(
            mongo, is_docker=True,
            hostnames=["cooja1", "cooja2", "cooja3"], ports=[22, 22, 22],
            remote_dir=REMOTE_DIR)

        offered = [c.args[0]["name"] for c in mongo.resource_repo.insert_seeded.call_args_list]
        assert offered == ["cooja1", "cooja2", "cooja3"]

    def test_resources_carry_the_host_id(self):
        mongo = FakeMongo()
        ensure_local_host(
            mongo, is_docker=True, hostnames=["cooja1"], ports=[22],
            remote_dir=REMOTE_DIR)
        assert mongo.resource_repo.insert_seeded.call_args.args[0]["host_id"] == "HOST_ID"

    def test_no_pre_check_reads_the_collection(self):
        # insert_seeded is $setOnInsert, so a second run is a no-op per
        # resource and the decision belongs to the database. Reading first to
        # decide would be both redundant and racy: two master-nodes starting
        # together would each see "absent" and both try to create it.
        mongo = FakeMongo(inserted=False)
        ensure_local_host(
            mongo, is_docker=True, hostnames=["cooja1", "cooja2"], ports=[22, 22],
            remote_dir=REMOTE_DIR)

        mongo.resource_repo.find_by_host.assert_not_called()
        # Still offered, even though the repository reports nothing inserted.
        assert mongo.resource_repo.insert_seeded.call_count == 2

    def test_no_containers_configured_is_not_an_error(self):
        mongo = FakeMongo()
        assert ensure_local_host(
            mongo, is_docker=True, hostnames=[], ports=[],
            remote_dir=REMOTE_DIR) == "HOST_ID"
        mongo.resource_repo.insert_seeded.assert_not_called()


class TestPublishPublicKey:
    def test_only_public_material_is_published(self):
        mongo = FakeMongo()
        keystore = MagicMock()
        keystore.key_id = "b114dd6d8410a62a"
        keystore.public_key_b64 = "cHVibGlj"
        keystore.fingerprint = "SHA256:abc"
        keystore.retired_key_ids = ["old"]

        publish_public_key(mongo, keystore)

        kwargs = mongo.infra_crypto_repo.publish.call_args.kwargs
        assert kwargs == {
            "alg": "crypto_box_seal",
            "key_id": "b114dd6d8410a62a",
            "public_key": "cHVibGlj",
            "fingerprint": "SHA256:abc",
            "retired_key_ids": ["old"],
        }
        # The private key never leaves its mounted file.
        assert "private" not in repr(kwargs)
