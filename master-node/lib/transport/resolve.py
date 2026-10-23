"""Name resolution for hop 1, with a memory.

Paramiko resolves a host's name itself, inside ``connect``, and reports a
failed lookup as one more ``OSError``. That made two different events look
identical in the health cycle:

* **the master-node cannot resolve anything** — its resolver is broken or
  stale. Observed in deployment: Docker's embedded DNS copies the machine's
  upstream servers when the container *starts* and never updates them, so
  after the machine moved to another network every lookup failed with
  ``EAI_AGAIN`` while ``ssh`` from a terminal on the same machine worked;
* **the host's name does not exist** — a typo, or a record that was removed.

The first is the master-node's problem and says nothing about the host; the
second is the host's configuration. Both were counted as the host being
unreachable, and every remote host went offline at once.

Resolving here, before paramiko sees the address, lets the failure be
classified (:class:`NameResolutionFailed`) and bounded by a timeout, and lets
the last address that *worked* stand in when the lookup fails. That fallback
is safe only because the host key is pinned: an address that now belongs to
another machine fails the pin instead of being trusted, so it is offered only
to a connection that has a pin to check.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from lib.transport.base import TransportUnreachable

log = logging.getLogger(__name__)

#: How long one lookup may take. glibc's own default is 5 s per server and
#: two attempts; a resolver that has not answered in this long is not going
#: to, and the health cycle has other hosts to measure.
DEFAULT_RESOLVE_TIMEOUT = 5.0

#: ``getaddrinfo`` errors that mean "this name has no address" — an answer,
#: from a resolver that works. Everything else (``EAI_AGAIN``, ``EAI_FAIL``,
#: ``EAI_SYSTEM``, a timeout) means the resolver itself did not answer.
_NO_SUCH_NAME = {
    code for code in (
        getattr(socket, "EAI_NONAME", None),
        getattr(socket, "EAI_NODATA", None),
        getattr(socket, "EAI_ADDRFAMILY", None),
    ) if code is not None
}


class NameResolutionFailed(TransportUnreachable):
    """A host's name could not be turned into an address.

    A ``TransportUnreachable``, so every caller that only knows "could not get
    there" keeps treating it that way. ``resolver_fault`` is what a caller
    that knows better reads:

    * ``True`` — the resolver did not answer (timeout, ``EAI_AGAIN``,
      ``EAI_FAIL``). Nothing was learned about the host.
    * ``False`` — the resolver answered that the name does not exist. The
      host's configured address is wrong, or the record is gone.
    """

    def __init__(self, name: str, *, resolver_fault: bool, reason: str):
        self.name = name
        self.resolver_fault = resolver_fault
        self.reason = reason
        if resolver_fault:
            text = (
                f"cannot resolve {name}: the master-node's DNS resolver did "
                f"not answer ({reason}) — a problem on the master-node, not "
                "on the host"
            )
        else:
            text = (
                f"{name} does not resolve ({reason}) — check the address "
                "configured for this host"
            )
        super().__init__(text)


@dataclass(frozen=True)
class Resolution:
    """Where a connection to ``name`` should go, and how that was decided."""
    name: str
    addresses: tuple[str, ...]
    #: ``literal`` (already an address), ``dns`` (a fresh lookup) or
    #: ``cache`` (the lookup failed; the last address that worked).
    source: str
    #: Why the cache was used. Set only when ``source == "cache"``.
    dns_error: Optional[NameResolutionFailed] = None

    @property
    def from_cache(self) -> bool:
        return self.source == "cache"


def is_ip_literal(address: str) -> bool:
    try:
        ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return False
    return True


def classify(name: str, error: BaseException) -> NameResolutionFailed:
    """What a failed lookup says, and about whom."""
    if isinstance(error, NameResolutionFailed):
        return error
    if isinstance(error, (FutureTimeout, TimeoutError)):
        return NameResolutionFailed(
            name, resolver_fault=True, reason="no answer within the lookup timeout")
    if isinstance(error, socket.gaierror):
        code = error.args[0] if error.args else None
        reason = error.args[1] if len(error.args) > 1 else str(error)
        return NameResolutionFailed(
            name, resolver_fault=code not in _NO_SUCH_NAME, reason=str(reason))
    return NameResolutionFailed(name, resolver_fault=True, reason=str(error))


class HostResolver:
    """Resolves host names with a timeout and remembers what worked.

    Thread-safe: the health cycle checks hosts concurrently, and runners on
    other threads connect through the same instance.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_RESOLVE_TIMEOUT,
        getaddrinfo: Callable[..., list] = socket.getaddrinfo,
    ):
        self._timeout = timeout
        self._getaddrinfo = getaddrinfo
        self._lock = threading.Lock()
        self._known: dict[str, str] = {}
        # One lookup per name at a time. ``getaddrinfo`` cannot be cancelled,
        # so a timed-out lookup keeps its thread until the resolver gives up;
        # without this every caller would start another one behind it.
        self._inflight: dict[str, Future] = {}

    # ── the cache ──────────────────────────────────────────────────────────

    def remember(self, name: str, address: str) -> None:
        """Record the address a connection to ``name`` succeeded on."""
        if not name or not address or is_ip_literal(name):
            return
        with self._lock:
            self._known[name.lower()] = address

    def remember_if_unknown(self, name: str, address: str) -> bool:
        """Seed from a stored address without overwriting a fresher one."""
        if not name or not address or is_ip_literal(name):
            return False
        with self._lock:
            if name.lower() in self._known:
                return False
            self._known[name.lower()] = address
            return True

    def known(self, name: str) -> Optional[str]:
        with self._lock:
            return self._known.get(name.lower())

    # ── lookups ────────────────────────────────────────────────────────────

    def lookup(self, name: str, port: int = 22, *, timeout: Optional[float] = None
               ) -> tuple[str, ...]:
        """Every address ``name`` has, in the resolver's order.

        Raises :class:`NameResolutionFailed`. Never uses the cache.
        """
        if is_ip_literal(name):
            return (name.strip("[]"),)
        future = self._start(name, port)
        try:
            return future.result(timeout=self._timeout if timeout is None else timeout)
        except Exception as error:
            # Not BaseException: the first health cycle runs on the main
            # thread, and a Ctrl+C there is not a DNS failure.
            raise classify(name, error) from None

    def lookup_many(self, names: Iterable[str], *, timeout: Optional[float] = None
                    ) -> dict[str, tuple[str, ...] | NameResolutionFailed]:
        """Look several names up at once, under one shared timeout."""
        wanted = sorted({n for n in names if n})
        futures = {n: self._start(n, 22) for n in wanted if not is_ip_literal(n)}
        deadline = time.monotonic() + (self._timeout if timeout is None else timeout)
        out: dict[str, tuple[str, ...] | NameResolutionFailed] = {}
        for name in wanted:
            if name not in futures:
                out[name] = (name.strip("[]"),)
                continue
            try:
                out[name] = futures[name].result(
                    timeout=max(0.0, deadline - time.monotonic()))
            except Exception as error:
                out[name] = classify(name, error)
        return out

    def resolve(self, name: str, port: int, *, allow_cached: bool) -> Resolution:
        """Where to connect for ``name``: a fresh answer, or the last good one.

        ``allow_cached`` is the caller saying the connection will check a
        pinned host key; without one, an old address could belong to anyone
        and is not offered.
        """
        if is_ip_literal(name):
            return Resolution(name, (name.strip("[]"),), "literal")
        try:
            return Resolution(name, self.lookup(name, port), "dns")
        except NameResolutionFailed as error:
            cached = self.known(name) if allow_cached else None
            if cached is None:
                raise
            log.warning(
                "%s; connecting to its last known address %s (the host key is "
                "pinned, so a different machine there would be refused)",
                error, cached,
            )
            return Resolution(name, (cached,), "cache", dns_error=error)

    def _start(self, name: str, port: int) -> Future:
        key = name.lower()
        with self._lock:
            future = self._inflight.get(key)
            if future is not None:
                return future
            future = Future()
            self._inflight[key] = future
        threading.Thread(
            target=self._run, args=(key, name, port, future),
            daemon=True, name=f"resolve-{name}",
        ).start()
        return future

    def _run(self, key: str, name: str, port: int, future: Future) -> None:
        try:
            infos = self._getaddrinfo(name, port, 0, socket.SOCK_STREAM)
            addresses: list[str] = []
            for info in infos:
                address = str(info[4][0])
                if address not in addresses:
                    addresses.append(address)
            if not addresses:
                raise socket.gaierror(
                    getattr(socket, "EAI_NONAME", -2), "no address returned")
            future.set_result(tuple(addresses))
        except BaseException as error:  # noqa: BLE001 - handed to the waiter
            future.set_exception(error)
        finally:
            with self._lock:
                if self._inflight.get(key) is future:
                    del self._inflight[key]


#: The process's resolver. Shared so an address learned by the health cycle
#: serves a runner's reconnection, and the other way round.
DEFAULT_RESOLVER = HostResolver()
