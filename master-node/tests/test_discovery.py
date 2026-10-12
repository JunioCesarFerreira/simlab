"""Tests for discovering a host's Cooja containers.

Discovery is level 1 of availability: it answers what exists and whether it is
running, and it resolves the hop-2 address nobody has to maintain by hand.

The properties pinned here are the ones whose absence would be silent:

  * Docker being unreachable must not take a working host out of the pool;
  * a container with no resolvable address must not be guessed at;
  * a vanished container is reported, never decided upon.
"""
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.discovery import DEFAULT_LABEL_FILTER, discover_all, discover_host
from lib.dockerctl import DockerError, DockerUnavailable

HOST_ID = ObjectId("507f1f77bcf86cd799430011")

LOCAL_HOST = {
    "_id": HOST_ID,
    "name": "local",
    "type": "local",
    "container_access": {"username": "root", "port": 22,
                         "address_mode": "container_name"},
    "discovery": {"mode": "docker", "label_filter": DEFAULT_LABEL_FILTER,
                  "endpoint": "socket"},
}
REMOTE_HOST = {
    "_id": HOST_ID,
    "name": "lab-01",
    "type": "remote",
    "access": {"address": "192.168.0.42", "port": 22, "username": "simlab"},
    "container_access": {"username": "root", "port": 22,
                         "address_mode": "container_ip"},
    "discovery": {"mode": "docker", "label_filter": "simlab.type=cooja-remote",
                  "name_prefix": None, "endpoint": "ssh-cli"},
}


def inspect_doc(name="cooja1", state="running", ip="172.18.0.5"):
    return {
        "Id": "abc123",
        "Name": f"/{name}",
        "State": {"Status": state, "Running": state == "running"},
        "Config": {"Image": "cooja:v1.1", "Labels": {"simlab.type": "cooja"}},
        "NetworkSettings": {
            "Networks": {"cooja_net": {"IPAddress": ip}},
            "Ports": {"22/tcp": [{"HostIp": "127.0.0.1", "HostPort": "2231"}]},
        },
    }


def fake_mongo(sync=None) -> MagicMock:
    mongo = MagicMock()
    mongo.resource_repo.sync_discovered.return_value = sync or {
        "created": [], "updated": [], "missing": [], "missing_occupied": [],
    }
    return mongo


class FakeInspector:
    def __init__(self, containers=None, error=None):
        self.containers = containers or []
        self.error = error
        self.calls: list[tuple] = []

    def list_containers(self, label_filter=None, name_prefix=None):
        self.calls.append((label_filter, name_prefix))
        if self.error:
            raise self.error
        return self.containers


# ── The happy path ───────────────────────────────────────────────────────────

class TestDiscovery:
    def test_containers_are_synced_with_their_addresses(self):
        mongo = fake_mongo(sync={
            "created": ["cooja1"], "updated": [], "missing": [], "missing_occupied": []})
        inspector = FakeInspector([inspect_doc("cooja1")])

        report = discover_host(mongo, LOCAL_HOST, inspector=inspector)

        assert report.available is True
        assert report.created == ["cooja1"]
        containers, kwargs = (
            mongo.resource_repo.sync_discovered.call_args.args[1],
            mongo.resource_repo.sync_discovered.call_args.kwargs,
        )
        assert [c["name"] for c in containers] == ["cooja1"]
        # The address comes from inspection, not from a registration.
        assert kwargs["endpoints"] == {"cooja1": {"address": "cooja1", "port": 22}}

    def test_the_address_mode_follows_the_host(self):
        mongo = fake_mongo()
        discover_host(mongo, REMOTE_HOST, inspector=FakeInspector([inspect_doc()]))
        endpoints = mongo.resource_repo.sync_discovered.call_args.kwargs["endpoints"]
        # A remote host routes to the container's bridge address.
        assert endpoints == {"cooja1": {"address": "172.18.0.5", "port": 22}}

    def test_the_hosts_label_filter_is_used(self):
        inspector = FakeInspector([])
        discover_host(fake_mongo(), REMOTE_HOST, inspector=inspector)
        assert inspector.calls == [("simlab.type=cooja-remote", None)]

    def test_a_stopped_container_is_still_recorded(self):
        # It has to be seen in order to be marked offline rather than
        # vanishing along with the record of what ran on it.
        mongo = fake_mongo()
        discover_host(
            mongo, LOCAL_HOST, inspector=FakeInspector([inspect_doc(state="exited")]))
        containers = mongo.resource_repo.sync_discovered.call_args.args[1]
        assert containers[0]["state"] == "exited"

    def test_the_outcome_is_recorded_on_the_host(self):
        mongo = fake_mongo(sync={
            "created": ["cooja1"], "updated": ["cooja2"], "missing": [],
            "missing_occupied": []})
        discover_host(
            mongo, LOCAL_HOST,
            inspector=FakeInspector([inspect_doc("cooja1"), inspect_doc("cooja2")]))
        kwargs = mongo.host_repo.record_discovery.call_args.kwargs
        assert kwargs["count"] == 2
        assert "2 container(s)" in kwargs["message"]


# ── Unresolvable addresses ───────────────────────────────────────────────────

class TestUnresolvedAddress:
    def test_a_container_without_an_address_is_not_guessed_at(self):
        mongo = fake_mongo()
        # No bridge IP, and the host is configured for container_ip.
        inspector = FakeInspector([inspect_doc(ip="")])

        report = discover_host(mongo, REMOTE_HOST, inspector=inspector)

        assert report.unresolved == ["cooja1"]
        endpoints = mongo.resource_repo.sync_discovered.call_args.kwargs["endpoints"]
        assert endpoints == {}

    def test_the_container_facts_are_still_recorded(self):
        # The resource exists and its state is known; only its address is not,
        # which build_plan refuses rather than inventing.
        mongo = fake_mongo()
        discover_host(mongo, REMOTE_HOST, inspector=FakeInspector([inspect_doc(ip="")]))
        containers = mongo.resource_repo.sync_discovered.call_args.args[1]
        assert [c["name"] for c in containers] == ["cooja1"]


# ── Docker unavailable is not the host failing ───────────────────────────────

class TestUnavailable:
    @pytest.mark.parametrize("error", [
        DockerUnavailable("socket not mounted"),
        DockerError("daemon said no"),
    ])
    def test_the_inventory_is_left_untouched(self, error):
        # Wiping or reinterpreting the resources here would take a working
        # host out of the pool because of a missing socket — which is all the
        # master-node ever needed before this feature.
        mongo = fake_mongo()
        report = discover_host(
            mongo, LOCAL_HOST, inspector=FakeInspector(error=error))

        assert report.available is False
        assert str(error) in report.unavailable_reason
        mongo.resource_repo.sync_discovered.assert_not_called()

    def test_the_reason_is_recorded_for_the_operator(self):
        mongo = fake_mongo()
        discover_host(
            mongo, LOCAL_HOST,
            inspector=FakeInspector(error=DockerUnavailable("socket not mounted")))
        message = mongo.host_repo.record_discovery.call_args.kwargs["message"]
        assert "unavailable" in message and "socket not mounted" in message

    def test_an_unexpected_error_is_contained(self):
        mongo = fake_mongo()
        report = discover_host(
            mongo, LOCAL_HOST, inspector=FakeInspector(error=ValueError("odd")))
        assert report.available is False
        assert "ValueError" in report.unavailable_reason
        mongo.resource_repo.sync_discovered.assert_not_called()

    def test_a_remote_host_without_an_address_is_unavailable(self):
        mongo = fake_mongo()
        host = {**REMOTE_HOST, "access": {}}
        report = discover_host(mongo, host)
        assert report.available is False
        assert "no SSH address" in report.unavailable_reason


# ── Vanished containers are reported, not judged ─────────────────────────────

class TestVanished:
    def test_a_container_holding_a_simulation_is_only_reported(self):
        mongo = fake_mongo(sync={
            "created": [], "updated": [], "missing": ["cooja9"],
            "missing_occupied": ["cooja9"]})

        report = discover_host(
            mongo, LOCAL_HOST, inspector=FakeInspector([inspect_doc()]))

        # Whether that simulation died is for reconciliation, which can probe.
        assert report.missing_occupied == ["cooja9"]
        mongo.simulation_repo.mark_error.assert_not_called()


# ── Every host ───────────────────────────────────────────────────────────────

class TestDiscoverAll:
    def test_only_enabled_hosts_are_visited(self):
        mongo = fake_mongo()
        mongo.host_repo.find_enabled.return_value = [LOCAL_HOST, REMOTE_HOST]
        reports = discover_all(
            mongo, inspector_for=lambda host: FakeInspector([inspect_doc()]))
        assert [r.host_name for r in reports] == ["local", "lab-01"]

    def test_no_hosts_is_not_an_error(self):
        mongo = fake_mongo()
        mongo.host_repo.find_enabled.return_value = []
        assert discover_all(mongo) == []
