"""Tests for the watch on the master-node's own resolver.

A restart is the only repair for a container whose DNS servers went stale,
and also the one thing here that can interrupt work — so what is locked in
is mostly when it does *not* happen:

  * only a resolver that answers for none of the hosts' names counts as
    broken; a name that does not exist is an answer;
  * not before it has lasted ``restart_after_sec``;
  * not while anything is placed;
  * not twice within ``restart_guard_sec`` — the guard against a loop when
    the machine itself has no DNS;
  * never when self-restart is off.

The resolver is fed by a fake ``getaddrinfo`` and the clock is a variable.
"""
import socket
from datetime import datetime, timedelta, timezone

from lib.dns_watch import ResolverWatch, watched_names
from lib.transport.resolve import HostResolver

T0 = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)


def remote(name, address, **over):
    host = {"_id": name, "name": name, "type": "remote",
            "access": {"address": address, "port": 22}}
    host.update(over)
    return host


HOSTS = [remote("vm1", "andromeda.example"), remote("vm2", "andromeda.example"),
         remote("lab", "lab.example"), remote("raw", "192.168.0.42"),
         {"_id": "local", "name": "local", "type": "local"}]


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def getaddrinfo_where(down=(), missing=()):
    def getaddrinfo(host, port, *_args):
        if host in down or "*" in down:
            raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
        if host in missing:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port))]
    return getaddrinfo


def watch(down=(), missing=(), clock=None, **kwargs):
    return ResolverWatch(
        HostResolver(getaddrinfo=getaddrinfo_where(down, missing)),
        clock=clock or Clock(), **kwargs)


class TestWhatIsWatched:
    def test_only_names_used_to_reach_remote_hosts(self):
        # An address needs no resolver, and a local host has no first hop.
        assert watched_names(HOSTS) == ["andromeda.example", "lab.example"]

    def test_nothing_registered_by_name_is_never_broken(self):
        w = watch(down=("*",))

        status = w.observe([remote("raw", "192.168.0.42")])

        assert status.names == []
        assert not status.broken


class TestBroken:
    def test_every_name_without_an_answer_is_broken(self):
        status = watch(down=("*",)).observe(HOSTS)

        assert status.broken
        assert set(status.failing) == {"andromeda.example", "lab.example"}
        assert status.to_document()["ok"] is False

    def test_one_name_failing_while_others_resolve_is_that_names_problem(self):
        status = watch(down=("lab.example",)).observe(HOSTS)

        assert not status.broken
        assert "no answer for lab.example" in status.message

    def test_a_name_that_does_not_exist_is_an_answer(self):
        # The resolver answered; restarting would not change what it says.
        status = watch(missing=("andromeda.example", "lab.example")).observe(HOSTS)

        assert not status.broken
        assert set(status.missing) == {"andromeda.example", "lab.example"}

    def test_the_failure_keeps_its_start_across_cycles(self):
        clock = Clock()
        w = watch(down=("*",), clock=clock)
        w.observe(HOSTS)
        clock.advance(90)

        assert w.observe(HOSTS).failing_since == T0

    def test_an_answer_clears_it(self):
        down = {"*"}
        w = ResolverWatch(HostResolver(getaddrinfo=getaddrinfo_where(down=down)))
        w.observe(HOSTS)
        down.clear()

        status = w.observe(HOSTS)

        assert not status.broken and status.failing_since is None


class TestRestartDecision:
    def _broken_for(self, seconds, **kwargs):
        clock = Clock()
        w = watch(down=("*",), clock=clock, self_restart=True,
                  restart_after_sec=300, restart_guard_sec=3600, **kwargs)
        w.observe(HOSTS)
        clock.advance(seconds)
        w.observe(HOSTS)
        return w, clock

    def test_it_restarts_once_the_failure_has_lasted(self):
        w, _ = self._broken_for(301)

        decision = w.restart_decision(idle=True, last_restart_at=None)

        assert decision.reason is not None
        assert "DNS" in decision.reason

    def test_not_before(self):
        w, _ = self._broken_for(120)

        decision = w.restart_decision(idle=True, last_restart_at=None)

        assert decision.reason is None
        assert "lasts" in decision.waiting

    def test_not_while_a_simulation_is_placed(self):
        w, _ = self._broken_for(301)

        decision = w.restart_decision(idle=False, last_restart_at=None)

        assert decision.reason is None
        assert "no simulation is placed" in decision.waiting

    def test_not_again_soon_after_the_last_restart(self):
        # It did not help, so the machine's DNS is the likely cause; another
        # restart every five minutes would be a loop.
        w, clock = self._broken_for(301)

        decision = w.restart_decision(
            idle=True, last_restart_at=clock.now - timedelta(minutes=10))

        assert decision.reason is None
        assert "already restarted" in decision.waiting

    def test_again_once_the_guard_has_passed(self):
        w, clock = self._broken_for(301)

        decision = w.restart_decision(
            idle=True, last_restart_at=clock.now - timedelta(hours=2))

        assert decision.reason is not None

    def test_never_when_switched_off(self):
        clock = Clock()
        w = watch(down=("*",), clock=clock, self_restart=False, restart_after_sec=0)
        w.observe(HOSTS)
        clock.advance(3600)

        decision = w.restart_decision(idle=True, last_restart_at=None)

        assert decision.reason is None and decision.waiting is None

    def test_never_for_a_working_resolver(self):
        w = watch(self_restart=True, restart_after_sec=0)
        w.observe(HOSTS)

        assert w.restart_decision(idle=True, last_restart_at=None).reason is None
