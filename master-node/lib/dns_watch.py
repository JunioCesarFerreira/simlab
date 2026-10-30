"""Watching the master-node's own DNS resolver, and restarting to repair it.

Why a process would restart itself
----------------------------------
Inside a container on a user-defined network, names go to Docker's embedded
resolver (127.0.0.11), which forwards everything it does not know to the
upstream servers of the machine. It copies that list from the machine's
``resolv.conf`` **when the container starts** and never reads it again. Move
the machine to another network — a laptop going from home to the lab — and
the container keeps asking servers it can no longer reach: every lookup fails
with "Temporary failure in name resolution", while ``ssh`` from a terminal on
the same machine works, because the machine itself did pick up the new
servers.

``docker restart`` fixes it; nothing inside the container can. So once the
resolver has failed for long enough, the master-node exits and lets its
restart policy (``unless-stopped``) start it again with the current servers.

Why it is guarded
-----------------
* **Only a resolver that answers nothing.** Every name the hosts are
  registered by must fail with a resolver fault. A name that does not exist
  is an answer, and one failing name among several that resolve is that
  name's problem.
* **Only after it has lasted.** ``restart_after_sec``: a blip is not worth a
  restart.
* **Only with nothing placed.** A reserved or busy worker anywhere means a
  simulation could be interrupted. The hosts already reached keep working on
  their last known address in the meantime (``lib.transport.resolve``), so
  waiting is cheap.
* **At most once per ``restart_guard_sec``.** If the machine itself has no
  DNS, a restart does not help, and restarting every five minutes would be a
  loop. The last restart is stored in the database, so the guard survives the
  restart it is guarding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from lib.transport.pool import needs_jump
from lib.transport.resolve import (
    DEFAULT_RESOLVE_TIMEOUT,
    HostResolver,
    NameResolutionFailed,
    is_ip_literal,
)
from pylib.timeutil import as_aware, utc_now

log = logging.getLogger(__name__)

#: Five minutes of a resolver answering nothing before restarting over it.
DEFAULT_RESTART_AFTER_SEC = 300.0
#: And no more than one restart an hour for it.
DEFAULT_RESTART_GUARD_SEC = 3600.0

#: Exit status asking the restart policy for a new container: EX_TEMPFAIL.
RESTART_EXIT_CODE = 75


@dataclass
class ResolverStatus:
    """One look at the resolver, through the names the hosts are reached by."""
    checked_at: datetime
    #: Names looked up. Empty when every host is reached by address, and then
    #: there is nothing to judge — and nothing that DNS could break.
    names: list[str] = field(default_factory=list)
    #: Name → why the resolver did not answer for it.
    failing: dict[str, str] = field(default_factory=dict)
    #: Name → why it does not exist. An answer, so not the resolver's fault.
    missing: dict[str, str] = field(default_factory=dict)
    failing_since: Optional[datetime] = None

    @property
    def broken(self) -> bool:
        """Every name failed, and every failure was the resolver's."""
        return bool(self.names) and len(self.failing) == len(self.names)

    @property
    def message(self) -> str:
        if not self.names:
            return "no host is registered by name"
        if self.broken:
            return (
                f"the DNS resolver did not answer for any of {len(self.names)} "
                f"host name(s): {next(iter(self.failing.values()))}"
            )
        resolved = len(self.names) - len(self.failing) - len(self.missing)
        parts = [f"{resolved} of {len(self.names)} host name(s) resolved"]
        if self.failing:
            parts.append(f"no answer for {', '.join(sorted(self.failing))}")
        if self.missing:
            parts.append(f"not found: {', '.join(sorted(self.missing))}")
        return "; ".join(parts)

    def to_document(self) -> dict[str, Any]:
        return {
            "ok": not self.broken,
            "checked_at": self.checked_at,
            "names": list(self.names),
            "failing": dict(self.failing),
            "missing": dict(self.missing),
            "failing_since": self.failing_since,
            "message": self.message,
        }


@dataclass(frozen=True)
class RestartDecision:
    #: Set when the master-node should restart now, saying why.
    reason: Optional[str] = None
    #: Set when it would, but something holds it back — shown to operators.
    waiting: Optional[str] = None


def watched_names(hosts: Iterable[dict]) -> list[str]:
    """The names the master-node has to resolve to reach these hosts."""
    names = set()
    for host in hosts:
        if not needs_jump(host):
            continue
        address = str((host.get("access") or {}).get("address") or "").strip()
        if address and not is_ip_literal(address):
            names.add(address)
    return sorted(names)


class ResolverWatch:
    """Looks up the hosts' names every cycle and keeps the verdict."""

    def __init__(
        self,
        resolver: HostResolver,
        *,
        clock: Callable[[], datetime] = utc_now,
        timeout: float = DEFAULT_RESOLVE_TIMEOUT,
        self_restart: bool = False,
        restart_after_sec: float = DEFAULT_RESTART_AFTER_SEC,
        restart_guard_sec: float = DEFAULT_RESTART_GUARD_SEC,
    ):
        self._resolver = resolver
        self._clock = clock
        self._timeout = timeout
        self.self_restart = self_restart
        self._restart_after = restart_after_sec
        self._restart_guard = restart_guard_sec
        self._failing_since: Optional[datetime] = None
        self.status: Optional[ResolverStatus] = None

    @property
    def broken(self) -> bool:
        return self.status is not None and self.status.broken

    def observe(self, hosts: Iterable[dict]) -> ResolverStatus:
        names = watched_names(hosts)
        now = self._clock()
        status = ResolverStatus(checked_at=now, names=names)
        if names:
            answers = self._resolver.lookup_many(names, timeout=self._timeout)
            for name, answer in answers.items():
                if isinstance(answer, NameResolutionFailed):
                    target = status.failing if answer.resolver_fault else status.missing
                    target[name] = answer.reason

        was_broken = self.broken
        if status.broken:
            if self._failing_since is None:
                self._failing_since = now
            status.failing_since = self._failing_since
            if not was_broken:
                log.error(
                    "DNS: %s. Hosts reached by name are not being judged until "
                    "it answers; those with a known address are still reached "
                    "on it.", status.message,
                )
        else:
            if was_broken:
                log.info("DNS: the resolver answers again (%s)", status.message)
            self._failing_since = None
        self.status = status
        return status

    def restart_decision(
        self, *, idle: bool, last_restart_at: Optional[datetime]
    ) -> RestartDecision:
        """Whether to restart now to get a working resolver, and if not why."""
        status = self.status
        if not self.self_restart or status is None or not status.broken:
            return RestartDecision()
        now = self._clock()
        failing_for = (now - as_aware(status.failing_since or now)).total_seconds()
        if failing_for < self._restart_after:
            return RestartDecision(waiting=(
                f"restarting the master-node if it lasts "
                f"{int(self._restart_after - failing_for)} s more"))
        if last_restart_at is not None:
            since = (now - as_aware(last_restart_at)).total_seconds()
            if since < self._restart_guard:
                return RestartDecision(waiting=(
                    f"already restarted {int(since)} s ago for this; a restart "
                    "did not help, so the machine's own DNS is the likely "
                    "cause — not restarting again for "
                    f"{int(self._restart_guard - since)} s"))
        if not idle:
            return RestartDecision(waiting=(
                "will restart the master-node to renew its DNS servers once no "
                "simulation is placed"))
        return RestartDecision(reason=(
            f"the DNS resolver has not answered for {int(failing_for)} s "
            f"({status.message}); restarting so Docker gives the container "
            "the machine's current DNS servers"))
