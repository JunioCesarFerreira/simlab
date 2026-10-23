"""Tests for hop 1's name resolution.

The failure this exists for: Docker's embedded DNS kept the upstream servers
of the network the machine was on when the container started, every lookup
failed with ``EAI_AGAIN``, and every remote host went offline at once —
counted against hosts that were fine. Locked in here:

  * a resolver that does not answer is told apart from a name that does not
    exist;
  * a lookup is bounded, and a hung one is not started again per caller;
  * the last address that worked is used when DNS fails — only for a
    connection that has a pinned host key to check it against;
  * ``connect_host`` connects to the address it resolved, tries the next one
    when the first does not answer, and remembers the one that did.

Nothing here touches a network: ``getaddrinfo`` and paramiko are fakes.
"""
import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

from lib.credentials import SSHCredential
from lib.transport.base import TransportUnreachable
from lib.transport.plan import SSHEndpoint
from lib.transport.resolve import (
    HostResolver,
    NameResolutionFailed,
    Resolution,
    classify,
    is_ip_literal,
)
from lib.transport.session import HostSession, connect_host

NAME = "lab-01.example.org"
PIN = "SHA256:pinned"
CREDENTIAL = SSHCredential(username="simlab", password="k")


def answers(*addresses):
    """A ``getaddrinfo`` that returns these addresses, and counts its calls."""
    calls = []

    def getaddrinfo(host, port, family=0, type_=0):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addresses]

    getaddrinfo.calls = calls
    return getaddrinfo


def fails(code, text="lookup failed"):
    calls = []

    def getaddrinfo(host, port, family=0, type_=0):
        calls.append(host)
        raise socket.gaierror(code, text)

    getaddrinfo.calls = calls
    return getaddrinfo


RESOLVER_DOWN = fails(socket.EAI_AGAIN, "Temporary failure in name resolution")
NO_SUCH_NAME = fails(socket.EAI_NONAME, "Name or service not known")


# ── Telling the two failures apart ───────────────────────────────────────────

class TestClassification:
    def test_a_resolver_that_does_not_answer_is_the_resolvers_fault(self):
        error = classify(NAME, socket.gaierror(socket.EAI_AGAIN, "Temporary failure"))

        assert error.resolver_fault is True
        assert "master-node" in str(error)

    def test_a_name_that_does_not_exist_is_the_hosts_configuration(self):
        error = classify(NAME, socket.gaierror(socket.EAI_NONAME, "Name or service not known"))

        assert error.resolver_fault is False
        assert "check the address" in str(error)

    def test_a_timeout_is_the_resolvers_fault(self):
        assert classify(NAME, TimeoutError()).resolver_fault is True

    def test_it_is_still_an_unreachable_host_to_older_callers(self):
        # Everything that only knows "could not get there" keeps working.
        assert issubclass(NameResolutionFailed, TransportUnreachable)

    def test_addresses_are_recognised(self):
        assert is_ip_literal("192.168.0.42")
        assert is_ip_literal("::1")
        assert is_ip_literal("[fe80::1]")
        assert not is_ip_literal(NAME)


# ── Lookups ──────────────────────────────────────────────────────────────────

class TestLookup:
    def test_an_address_needs_no_lookup(self):
        getaddrinfo = answers("10.0.0.1")
        resolver = HostResolver(getaddrinfo=getaddrinfo)

        result = resolver.resolve("192.168.0.42", 22, allow_cached=False)

        assert result.addresses == ("192.168.0.42",)
        assert result.source == "literal"
        assert getaddrinfo.calls == []

    def test_a_name_is_resolved_to_every_distinct_address(self):
        resolver = HostResolver(getaddrinfo=answers("10.0.0.1", "10.0.0.1", "10.0.0.2"))

        result = resolver.resolve(NAME, 22, allow_cached=False)

        assert result.addresses == ("10.0.0.1", "10.0.0.2")
        assert result.source == "dns"

    def test_a_lookup_that_hangs_is_bounded(self):
        release = threading.Event()

        def hangs(*_args):
            release.wait(5)
            raise socket.gaierror(socket.EAI_AGAIN, "late")

        resolver = HostResolver(timeout=0.05, getaddrinfo=hangs)
        try:
            with pytest.raises(NameResolutionFailed) as caught:
                resolver.lookup(NAME)
            assert caught.value.resolver_fault is True
        finally:
            release.set()

    def test_a_hung_lookup_is_not_started_again_by_the_next_caller(self):
        release = threading.Event()
        calls = []

        def hangs(host, *_args):
            calls.append(host)
            release.wait(5)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 22))]

        resolver = HostResolver(timeout=0.05, getaddrinfo=hangs)
        try:
            for _ in range(3):
                with pytest.raises(NameResolutionFailed):
                    resolver.lookup(NAME)
            assert calls == [NAME]
        finally:
            release.set()

    def test_an_interrupt_is_not_mistaken_for_a_dns_failure(self):
        # The first health cycle runs on the main thread.
        def interrupted(*_args):
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            HostResolver(getaddrinfo=interrupted).lookup(NAME)

    def test_many_names_share_one_timeout(self):
        def mixed(host, port, *_args):
            if host == "down.example":
                raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure")
            if host == "gone.example":
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.9", port))]

        resolver = HostResolver(getaddrinfo=mixed)

        out = resolver.lookup_many(["up.example", "down.example", "gone.example", "10.0.0.1"])

        assert out["up.example"] == ("10.0.0.9",)
        assert out["10.0.0.1"] == ("10.0.0.1",)
        assert out["down.example"].resolver_fault is True
        assert out["gone.example"].resolver_fault is False


# ── The last address that worked ─────────────────────────────────────────────

class TestFallback:
    def test_a_pinned_host_is_reached_at_its_last_address_while_dns_is_down(self):
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)
        resolver.remember(NAME, "143.107.0.10")

        result = resolver.resolve(NAME, 22, allow_cached=True)

        assert result.addresses == ("143.107.0.10",)
        assert result.from_cache
        assert result.dns_error.resolver_fault is True

    def test_without_a_pin_an_old_address_is_not_trusted(self):
        # It could belong to another machine by now, and nothing would notice.
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)
        resolver.remember(NAME, "143.107.0.10")

        with pytest.raises(NameResolutionFailed):
            resolver.resolve(NAME, 22, allow_cached=False)

    def test_nothing_remembered_means_the_failure_is_raised(self):
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)

        with pytest.raises(NameResolutionFailed):
            resolver.resolve(NAME, 22, allow_cached=True)

    def test_a_fresh_answer_wins_over_the_remembered_one(self):
        resolver = HostResolver(getaddrinfo=answers("10.0.0.2"))
        resolver.remember(NAME, "10.0.0.1")

        assert resolver.resolve(NAME, 22, allow_cached=True).addresses == ("10.0.0.2",)

    def test_names_are_remembered_case_insensitively(self):
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)
        resolver.remember("Lab-01.Example.org", "10.0.0.1")

        assert resolver.known(NAME) == "10.0.0.1"

    def test_a_stored_address_does_not_overwrite_a_fresher_one(self):
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)
        resolver.remember(NAME, "10.0.0.2")

        assert resolver.remember_if_unknown(NAME, "10.0.0.1") is False
        assert resolver.known(NAME) == "10.0.0.2"

    def test_an_address_is_never_remembered_for_an_address(self):
        resolver = HostResolver()
        resolver.remember("192.168.0.42", "10.0.0.1")

        assert resolver.known("192.168.0.42") is None


# ── connect_host ─────────────────────────────────────────────────────────────

@pytest.fixture
def clients():
    created: list[MagicMock] = []

    def factory():
        client = MagicMock(name=f"client{len(created)}")
        created.append(client)
        return client

    with patch("lib.transport.session.paramiko.SSHClient", side_effect=factory):
        yield created


class TestConnectHost:
    def test_it_connects_to_the_address_it_resolved(self, clients):
        resolver = HostResolver(getaddrinfo=answers("10.0.0.1"))

        connect_host(SSHEndpoint(NAME, 2222), CREDENTIAL, resolver=resolver)

        assert clients[0].connect.call_args.args[0] == "10.0.0.1"
        assert clients[0].connect.call_args.kwargs["port"] == 2222

    def test_the_timeout_also_bounds_the_wait_for_the_banner(self, clients):
        # A host whose sshd is throttling accepts the TCP connection and then
        # says nothing; paramiko's own banner wait is 15 s.
        connect_host(SSHEndpoint("10.0.0.1", 22), CREDENTIAL, timeout=8.0,
                     resolver=HostResolver())

        kwargs = clients[0].connect.call_args.kwargs
        assert kwargs["timeout"] == 8.0
        assert kwargs["banner_timeout"] == 8.0

    def test_a_failed_lookup_is_raised_before_any_connection(self, clients):
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)

        with pytest.raises(NameResolutionFailed) as caught:
            connect_host(SSHEndpoint(NAME, 22), CREDENTIAL, resolver=resolver)

        assert caught.value.resolver_fault is True
        assert clients == []

    def test_a_pinned_host_is_reached_on_its_last_address(self, clients):
        resolver = HostResolver(getaddrinfo=RESOLVER_DOWN)
        resolver.remember(NAME, "10.0.0.7")
        seen = []

        connect_host(SSHEndpoint(NAME, 22), CREDENTIAL, pinned_host_key=PIN,
                     resolver=resolver, on_connected=lambda r, a: seen.append((r, a)))

        assert clients[0].connect.call_args.args[0] == "10.0.0.7"
        assert seen[0][0].from_cache and seen[0][1] == "10.0.0.7"

    def test_the_next_address_is_tried_when_one_does_not_answer(self, clients):
        resolver = HostResolver(getaddrinfo=answers("fe80::1", "10.0.0.1"))

        def connect(address, **_kwargs):
            if address == "fe80::1":
                raise OSError("Network is unreachable")

        with patch("lib.transport.session.paramiko.SSHClient") as factory:
            first, second = MagicMock(), MagicMock()
            first.connect.side_effect = connect
            factory.side_effect = [first, second]

            client = connect_host(SSHEndpoint(NAME, 22), CREDENTIAL, resolver=resolver)

        assert client is second
        first.close.assert_called_once()
        assert resolver.known(NAME) == "10.0.0.1"

    def test_no_address_answering_is_unreachable_and_says_where(self, clients):
        resolver = HostResolver(getaddrinfo=answers("10.0.0.1"))

        with patch("lib.transport.session.paramiko.SSHClient") as factory:
            factory.return_value.connect.side_effect = OSError("timed out")
            with pytest.raises(TransportUnreachable) as caught:
                connect_host(SSHEndpoint(NAME, 22), CREDENTIAL, resolver=resolver)

        assert not isinstance(caught.value, NameResolutionFailed)
        assert "10.0.0.1" in str(caught.value) and "timed out" in str(caught.value)

    def test_the_address_that_worked_is_remembered(self, clients):
        resolver = HostResolver(getaddrinfo=answers("10.0.0.3"))

        connect_host(SSHEndpoint(NAME, 22), CREDENTIAL, resolver=resolver)

        assert resolver.known(NAME) == "10.0.0.3"

    def test_a_session_keeps_how_it_was_reached(self, clients):
        resolver = HostResolver(getaddrinfo=answers("10.0.0.4"))
        session = HostSession(SSHEndpoint(NAME, 22), CREDENTIAL, resolver=resolver)

        session.open()

        assert session.connected_address == "10.0.0.4"
        assert isinstance(session.resolution, Resolution)
        assert session.resolution.source == "dns"
