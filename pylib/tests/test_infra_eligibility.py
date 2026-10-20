"""Tests for the one rule that decides whose observation counts.

``operator_configured`` answers a single question — did a person, rather than
Docker, decide how this worker is reached? — and three decisions depend on the
answer: the health cycle probes such a worker regardless of Docker's silence,
the release must not evict it on a stale ``missing``, and the API tells the
page why the Docker column is not being used.

It lives in the shared library because those three live in three different
services, and they disagreed once. The release path had not been given the
exception, and on a live installation two hand-configured remote workers left
the pool on *every* completed simulation.
"""
from pylib.db.models.infra import operator_configured

DISCOVERED = {"name": "cooja1", "origin": "discovered", "connection": {}}
DOCKER_HOST = {"name": "local", "discovery": {"mode": "docker"},
               "container_access": {"address_mode": "container_ip"}}


class TestTheWorkersOwnSettings:
    def test_a_worker_with_its_own_port_is_configured(self):
        worker = {**DISCOVERED, "connection": {"port": 2223}}
        assert operator_configured(DOCKER_HOST, worker) is True

    def test_a_worker_with_its_own_credential_is_configured(self):
        worker = {**DISCOVERED, "connection": {"secret": {"ciphertext": "x"}}}
        assert operator_configured(DOCKER_HOST, worker) is True

    def test_transport_alone_does_not_count(self):
        """The mapper always writes it, so it is not a statement of intent."""
        worker = {**DISCOVERED, "connection": {"transport": "ssh"}}
        assert operator_configured(DOCKER_HOST, worker) is False

    def test_a_hand_registered_worker_is_configured(self):
        assert operator_configured(DOCKER_HOST, {**DISCOVERED, "origin": "manual"}) is True

    def test_a_discovered_worker_on_a_docker_host_is_not(self):
        assert operator_configured(DOCKER_HOST, DISCOVERED) is False


class TestTheHostsSettings:
    def test_a_host_that_lists_workers_by_hand(self):
        host = {**DOCKER_HOST, "discovery": {"mode": "manual"}}
        assert operator_configured(host, DISCOVERED) is True

    def test_a_host_that_states_addresses_explicitly(self):
        host = {**DOCKER_HOST, "container_access": {"address_mode": "explicit"}}
        assert operator_configured(host, DISCOVERED) is True

    def test_a_missing_host_record_leaves_only_the_workers_signals(self):
        # Recovery may release a worker whose host registration is gone.
        assert operator_configured({}, DISCOVERED) is False
        assert operator_configured({}, {**DISCOVERED, "origin": "manual"}) is True

    def test_empty_documents_do_not_raise(self):
        assert operator_configured({}, {}) is False
