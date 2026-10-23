"""One hop-1 session per host, lent to everything a cycle asks of it.

A health cycle touches the same host several times: once for Docker's
inventory, then once per worker to probe it. Each of those opened its own SSH
connection to the host, so a host with four workers paid five handshakes every
thirty seconds. Measured on a pilot: 3078 transport opens for 344 simulations,
and 41 opens in three minutes while nothing was running — which is what made
the host's sshd start turning arrivals away ("Error reading SSH protocol
banner" is what a client sees when sshd's MaxStartups refuses it).

This makes hop 1 a per-host resource with a lifetime the caller controls: open
it once, lend it to the inventory and to each probe, close it when the host's
turn is over. Hop 2 is untouched — every container still gets its own
authenticated channel, because that is the connection a simulation runs on.

A *failed* handshake is remembered too, and re-raised for the rest of the
host's turn. Retrying it once per worker is precisely the behaviour that turns
one unreachable host into a burst of connection attempts against it. The
caller can ask for it (``failure``) to tell *why* the host could not be
reached — a resolver that did not answer is not the host's fault.

Hosts are checked concurrently, each by one thread, so every map here is
guarded; a host's turn itself is never shared between threads.
"""
import logging
import threading
from typing import Any, Callable, Optional

from lib.credentials import SecretOpener, resolve_host_credential
from lib.transport.plan import PlanError, SSHEndpoint
from lib.transport.resolve import HostResolver, Resolution
from lib.transport.session import CONNECT_TIMEOUT, HostSession
from pylib.db.models.infra import EnumHostType

log = logging.getLogger(__name__)


def _key(host: dict[str, Any]) -> str:
    """Identify a host by its record, falling back to its name."""
    host_id = host.get("_id")
    return str(host_id) if host_id is not None else f"name:{host.get('name')}"


def needs_jump(host: dict[str, Any]) -> bool:
    """Whether reaching this host's containers goes through the host itself.

    A local host's containers are reached directly, so there is no hop 1 to
    share and the pool stays out of the way.
    """
    return (host.get("type") or EnumHostType.REMOTE.value) != EnumHostType.LOCAL.value


class JumpPool:
    """Host sessions, opened at most once per host per turn.

    ``session`` is the whole interface: it answers ``None`` for a host that
    needs no first hop, so callers do not have to ask what kind of host they
    were given.
    """

    def __init__(
        self,
        opener: Optional[SecretOpener] = None,
        *,
        on_learn: Optional[Callable[[dict, str], None]] = None,
        session_factory: Optional[Callable[..., HostSession]] = None,
        connect_timeout: float = CONNECT_TIMEOUT,
        resolver: Optional[HostResolver] = None,
        on_connected: Optional[Callable[[dict, Resolution, str], None]] = None,
    ):
        self._opener = opener
        # Called with (host, fingerprint) when a host's key is accepted on
        # first contact. Without it, a host first reached by the health cycle
        # — which is the usual case, since the cycle runs before anything is
        # placed — was never pinned at all: trust on first use with nothing
        # recording the first use.
        self._on_learn = on_learn
        self._session_factory = session_factory or HostSession
        # The health cycle passes a shorter one than a simulation's: a check
        # that waits twenty seconds for a host that is down delays every
        # other host's turn and the housekeeping after them.
        self._connect_timeout = connect_timeout
        self._resolver = resolver
        # Called with (host, resolution, address) once a session is open, so
        # the address that worked can be kept for when DNS does not.
        self._on_connected = on_connected
        self._lock = threading.Lock()
        self._sessions: dict[str, HostSession] = {}
        self._failures: dict[str, Exception] = {}

    # ── lending ────────────────────────────────────────────────────────────

    def session(self, host: dict[str, Any]) -> Optional[HostSession]:
        """An open session on this host, or ``None`` if it needs no hop 1.

        Raises whatever opening it raised — the same error object for every
        caller in this turn, so an unreachable host is reported once and
        attempted once.
        """
        if not needs_jump(host):
            return None

        key = _key(host)
        with self._lock:
            failure = self._failures.get(key)
            existing = self._sessions.get(key)
        if failure is not None:
            raise failure
        if existing is not None:
            return existing

        try:
            session = self._open(host)
        except Exception as error:
            with self._lock:
                self._failures[key] = error
            raise
        with self._lock:
            self._sessions[key] = session
        return session

    def failure(self, host: dict[str, Any]) -> Optional[Exception]:
        """Why opening this host's session failed in this turn, if it did."""
        with self._lock:
            return self._failures.get(_key(host))

    def resolution(self, host: dict[str, Any]) -> Optional[Resolution]:
        """How this turn's session found its address, if one is open."""
        with self._lock:
            session = self._sessions.get(_key(host))
        return getattr(session, "resolution", None)

    def _open(self, host: dict[str, Any]) -> HostSession:
        access = host.get("access") or {}
        address = access.get("address")
        name = str(host.get("name", "?"))
        if not address:
            raise PlanError(f"host {name!r} has no SSH address configured")

        credential = resolve_host_credential(access, self._opener, host_name=name)
        on_learn = None
        if self._on_learn is not None:
            def on_learn(fingerprint: str, host=host) -> None:
                self._on_learn(host, fingerprint)
        session = self._session_factory(
            SSHEndpoint(str(address), int(access.get("port") or 22)),
            credential,
            pinned_host_key=access.get("host_key_fingerprint"),
            on_learn_host_key=on_learn,
            host_name=name,
            timeout=self._connect_timeout,
            resolver=self._resolver,
        )
        session.open()
        log.debug("Host session open: %s (%s)", name, session.endpoint)
        address = getattr(session, "connected_address", None)
        resolution = getattr(session, "resolution", None)
        if self._on_connected is not None and address and resolution is not None:
            try:
                self._on_connected(host, resolution, address)
            except Exception:
                # Bookkeeping. The session is open and works; losing the
                # address only costs the fallback until the next success.
                log.exception("Could not record the address of host %s", name)
        return session

    # ── lifetime ───────────────────────────────────────────────────────────

    def close_host(self, host: dict[str, Any]) -> None:
        """End this host's turn: close its session and forget its failure."""
        key = _key(host)
        with self._lock:
            self._failures.pop(key, None)
            session = self._sessions.pop(key, None)
        if session is not None:
            session.close()

    def close(self) -> None:
        with self._lock:
            self._failures.clear()
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def __enter__(self) -> "JumpPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
