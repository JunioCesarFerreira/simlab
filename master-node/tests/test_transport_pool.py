"""Tests for the hop-1 pool.

What this exists to prevent is a connection per worker. A health cycle asks
the same host for its Docker inventory and then once per container, and each
of those used to be its own SSH handshake: a pilot measured 3078 transport
opens for 344 simulations, and the host's sshd eventually started refusing
them.

Properties locked in:
  * a host is connected to at most once per turn, however many callers ask;
  * a local host is not connected to at all — there is no first hop;
  * a failed handshake is not retried once per worker;
  * a turn ends when the caller says so, and the next one starts clean;
  * a missing address is a configuration error, raised before any socket.
"""
from unittest.mock import MagicMock

import pytest

from lib.credentials import CredentialUnavailable, SecretOpener
from lib.transport.base import TransportUnreachable
from lib.transport.plan import PlanError
from lib.transport.pool import JumpPool, needs_jump

HOST_ID = "507f1f77bcf86cd799430011"

LOCAL_HOST = {"_id": "local-id", "name": "local", "type": "local"}
REMOTE_HOST = {
    "_id": HOST_ID,
    "name": "lab-01",
    "type": "remote",
    "access": {
        "address": "192.168.0.42", "port": 2222, "username": "simlab",
        "auth_method": "password",
        "secret": {"alg": "crypto_box_seal", "key_id": "0" * 16, "ciphertext": "x"},
        "host_key_fingerprint": "SHA256:abc",
    },
}
OTHER_HOST = {**REMOTE_HOST, "_id": "other-id", "name": "lab-02"}


class FakeKeystore:
    def open(self, envelope):
        return "hunter2"


class FakeSession:
    """Records that it was opened and closed, without a network."""

    instances: list["FakeSession"] = []

    def __init__(self, endpoint, credential, *, pinned_host_key=None,
                 on_learn_host_key=None, host_name="?"):
        self.endpoint = endpoint
        self.credential = credential
        self.pinned_host_key = pinned_host_key
        self.on_learn_host_key = on_learn_host_key
        self.host_name = host_name
        self.opens = 0
        self.closes = 0
        self.client = MagicMock(name=f"client:{host_name}")
        FakeSession.instances.append(self)

    def open(self):
        self.opens += 1

    def close(self):
        self.closes += 1


@pytest.fixture(autouse=True)
def _reset_instances():
    FakeSession.instances = []
    yield
    FakeSession.instances = []


def build_pool(session_factory=FakeSession, keystore=None) -> JumpPool:
    return JumpPool(
        SecretOpener(keystore or FakeKeystore()), session_factory=session_factory)


class FailingSession(FakeSession):
    def open(self):
        self.opens += 1
        raise TransportUnreachable("host 192.168.0.42:2222 is unreachable")


# ── Which hosts need a first hop ─────────────────────────────────────────────

class TestNeedsJump:
    def test_local_host_does_not(self):
        assert needs_jump(LOCAL_HOST) is False

    def test_remote_host_does(self):
        assert needs_jump(REMOTE_HOST) is True

    def test_a_host_without_a_type_is_treated_as_remote(self):
        # The conservative reading: assuming local would skip hop 1 and try to
        # reach a container address that only exists on the host.
        assert needs_jump({"name": "?"}) is True


# ── Reuse ────────────────────────────────────────────────────────────────────

class TestReuse:
    def test_a_local_host_is_never_connected_to(self):
        pool = build_pool()
        assert pool.session(LOCAL_HOST) is None
        assert FakeSession.instances == []

    def test_one_handshake_serves_every_caller(self):
        pool = build_pool()

        first = pool.session(REMOTE_HOST)
        second = pool.session(REMOTE_HOST)
        third = pool.session(REMOTE_HOST)

        assert first is second is third
        assert len(FakeSession.instances) == 1
        assert FakeSession.instances[0].opens == 1

    def test_different_hosts_get_different_sessions(self):
        pool = build_pool()

        a = pool.session(REMOTE_HOST)
        b = pool.session(OTHER_HOST)

        assert a is not b
        assert sorted(s.host_name for s in FakeSession.instances) == ["lab-01", "lab-02"]

    def test_the_session_carries_the_hosts_address_and_pin(self):
        pool = build_pool()

        session = pool.session(REMOTE_HOST)

        assert (session.endpoint.address, session.endpoint.port) == ("192.168.0.42", 2222)
        assert session.pinned_host_key == "SHA256:abc"
        assert session.credential.username == "simlab"

    def test_port_defaults_to_22(self):
        pool = build_pool()
        host = {**REMOTE_HOST, "access": {**REMOTE_HOST["access"], "port": None}}
        assert pool.session(host).endpoint.port == 22


# ── Failure ──────────────────────────────────────────────────────────────────

class TestFailure:
    def test_a_failed_handshake_is_not_retried_per_caller(self):
        """The behaviour that turned one down host into a burst of attempts."""
        pool = build_pool(session_factory=FailingSession)

        with pytest.raises(TransportUnreachable):
            pool.session(REMOTE_HOST)
        for _ in range(4):
            with pytest.raises(TransportUnreachable):
                pool.session(REMOTE_HOST)

        assert len(FakeSession.instances) == 1

    def test_every_caller_sees_the_same_error(self):
        pool = build_pool(session_factory=FailingSession)

        with pytest.raises(TransportUnreachable) as first:
            pool.session(REMOTE_HOST)
        with pytest.raises(TransportUnreachable) as second:
            pool.session(REMOTE_HOST)

        assert first.value is second.value

    def test_a_credential_problem_is_raised_as_itself(self):
        # The health cycle distinguishes this from unreachability: one is a
        # configuration fault against the worker, the other is the network.
        class Broken:
            def open(self, envelope):
                raise CredentialUnavailable("no key")

        pool = build_pool(keystore=Broken())
        with pytest.raises(CredentialUnavailable):
            pool.session(REMOTE_HOST)

    def test_a_missing_address_is_refused_before_connecting(self):
        pool = build_pool()
        host = {**REMOTE_HOST, "access": {"username": "simlab"}}

        with pytest.raises(PlanError, match="no SSH address"):
            pool.session(host)
        assert FakeSession.instances == []

    def test_a_new_turn_retries_a_host_that_failed(self):
        pool = build_pool(session_factory=FailingSession)
        with pytest.raises(TransportUnreachable):
            pool.session(REMOTE_HOST)

        pool.close_host(REMOTE_HOST)

        with pytest.raises(TransportUnreachable):
            pool.session(REMOTE_HOST)
        assert len(FakeSession.instances) == 2


# ── Lifetime ─────────────────────────────────────────────────────────────────

class TestLifetime:
    def test_close_host_closes_only_that_host(self):
        pool = build_pool()
        a = pool.session(REMOTE_HOST)
        b = pool.session(OTHER_HOST)

        pool.close_host(REMOTE_HOST)

        assert (a.closes, b.closes) == (1, 0)

    def test_a_closed_host_is_connected_to_again_next_turn(self):
        pool = build_pool()
        first = pool.session(REMOTE_HOST)
        pool.close_host(REMOTE_HOST)

        second = pool.session(REMOTE_HOST)

        assert second is not first
        assert len(FakeSession.instances) == 2

    def test_closing_an_unknown_host_is_harmless(self):
        build_pool().close_host(REMOTE_HOST)

    def test_close_releases_everything(self):
        pool = build_pool()
        a = pool.session(REMOTE_HOST)
        b = pool.session(OTHER_HOST)

        pool.close()

        assert (a.closes, b.closes) == (1, 1)
        assert pool.session(REMOTE_HOST) is not a

    def test_context_manager_closes(self):
        with build_pool() as pool:
            session = pool.session(REMOTE_HOST)
        assert session.closes == 1

    def test_a_host_without_an_id_is_keyed_by_name(self):
        # Hosts are always stored, but a caller may hand over a partial record
        # — keying on a missing _id would make every host the same host.
        pool = build_pool()
        pool.session({"name": "lab-01", "type": "remote",
                      "access": REMOTE_HOST["access"]})
        pool.session({"name": "lab-02", "type": "remote",
                      "access": REMOTE_HOST["access"]})
        assert len(FakeSession.instances) == 2


class TestFirstContact:
    """A key accepted on first contact has to be written down somewhere.

    The health cycle is usually the first thing to reach a host — it runs
    before anything is placed — and it had no callback at all, so the pin the
    next connection would be checked against was never recorded. Trust on
    first use with nothing recording the first use.
    """

    def test_the_session_is_given_a_way_to_report_what_it_learned(self):
        learned: list = []
        pool = JumpPool(
            SecretOpener(FakeKeystore()),
            on_learn=lambda host, fingerprint: learned.append(
                (host["name"], fingerprint)),
            session_factory=FakeSession,
        )

        session = pool.session(REMOTE_HOST)
        session.on_learn_host_key("SHA256:learned")

        assert learned == [("lab-01", "SHA256:learned")]

    def test_the_callback_names_the_host_it_came_from(self):
        # One pool serves every host in a cycle; a fingerprint with no host
        # attached could be written against the wrong registration.
        learned: list = []
        pool = JumpPool(
            SecretOpener(FakeKeystore()),
            on_learn=lambda host, fingerprint: learned.append(host["name"]),
            session_factory=FakeSession,
        )

        pool.session(REMOTE_HOST).on_learn_host_key("a")
        pool.session(OTHER_HOST).on_learn_host_key("b")

        assert learned == ["lab-01", "lab-02"]

    def test_the_stored_pin_still_reaches_the_session(self):
        pool = build_pool()
        assert pool.session(REMOTE_HOST).pinned_host_key == "SHA256:abc"

    def test_without_a_sink_nothing_is_reported(self):
        pool = build_pool()
        assert pool.session(REMOTE_HOST).on_learn_host_key is None
