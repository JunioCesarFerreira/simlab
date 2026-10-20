"""SSH transport, with an optional jump through the host.

The two-hop path: connect to the host, open a ``direct-tcpip`` channel to the
container's address, and hand that channel to a second client as its socket.
From there SCP and command execution are byte-for-byte what the master-node
already did — which is why no container port has to be published on the LAN.

The local host takes the same code path with ``jump=None`` and behaves exactly
as before.
"""
import logging
import shlex
import time
from typing import Callable, Iterator, Optional

import paramiko
from paramiko import SSHClient
from paramiko.ssh_exception import NoValidConnectionsError, SSHException
from scp import SCPClient, SCPException

from lib.credentials import SSHCredential
from lib.transport.base import (
    ExecResult,
    ProbeResult,
    TransportError,
    TransportUnreachable,
)
from lib.transport.hostkey import TunnelledHostKeyPolicy
from lib.transport.plan import ConnectionPlan
from lib.transport.session import KEEPALIVE_INTERVAL, connect_host

log = logging.getLogger(__name__)

# Per-run artifacts a Cooja run leaves inside the container. Unchanged from the
# single-hop implementation: build/cooja holds the mtype*.cooja libraries
# compiled for each simulation, whose names are random per run.
GENERATED_ARTIFACTS = ("build/cooja", "COOJA.testlog", "COOJA.log")

#: Printed by the probe when the JVM exited zero. A marker rather than
#: parsing the version text, because what matters is that it *ran*.
JAVA_OK_MARK = "SIMLABJAVA-ok"

#: Where the Cooja image keeps its runtime. The same path the launch uses, so
#: the probe answers for the binary that will actually run the simulation.
JAVA_BIN = "/opt/java/openjdk/bin/java"

CONNECT_TIMEOUT = 20.0

# How long a single SCP receive may stall before the transfer is abandoned.
# The library's default is 10 seconds, and it is not a budget for the whole
# file: it is how long one chunk may fail to arrive. Over a tunnel to another
# machine that is far too tight. Measured on a three-host pilot, a ~780 KiB
# testlog took a median of 3s from a remote host but up to 79s when the link
# was busy — and the 79s transfer *succeeded*, because no single gap reached
# ten. Two concurrent downloads through the same host did hit a ten-second
# gap, and two completed simulations lost their results.
SCP_SOCKET_TIMEOUT = 45.0
# Attempts per file. A stalled link is the one failure here worth retrying:
# the simulation has already run and the file is sitting on the worker, so
# giving up on the first timeout throws away minutes of work for seconds of
# network trouble.
TRANSFER_ATTEMPTS = 3
TRANSFER_BACKOFF_SEC = 2.0


def _is_missing_file(error: BaseException) -> bool:
    """Whether this failure means the file is not there.

    The one case not worth retrying. Established against a live container:
    a missing path comes back as ``SCPException: scp: <path>: No such file or
    directory``, while a stall surfaces either as ``SCPException: Error
    receiving, socket.timeout`` or as a bare ``TimeoutError``, depending on
    where in the transfer it happened. So the rule is to retry everything
    except an explicit "not there" — erring toward retrying, because the cost
    of a needless attempt is seconds and the cost of not retrying is a lost
    simulation.
    """
    return "no such file" in str(error).lower()


def _describe_error(error: BaseException) -> str:
    """Name the failure even when it carries no message.

    A bare ``TimeoutError`` stringifies to nothing, so the log line for the
    one failure this code exists to handle read ``failed:`` and said nothing
    at all.
    """
    text = str(error).strip()
    return text or type(error).__name__


def build_probe_command(jar: str, java_bin: str = JAVA_BIN) -> str:
    """The shell that answers "is there a Cooja build, and can Java run it?".

    Extracted so it can be *executed* in a test rather than have its output
    simulated. That distinction caught the defect this version fixes.

    The previous command was ``if java -version 2>&1 | head -n1; then echo
    ok; fi``, and ``if`` on a pipeline reads the status of its **last**
    element. ``head`` exits zero after reading the error message of a Java
    that does not exist, so the mark was printed for a container with no
    runtime at all: measured on a real shell, the probe returned
    ``java_ok=True, ready=True, java_version=None`` with the binary missing.

    So the status is taken from Java itself, by capturing its output into a
    variable first: for ``V=$(cmd)`` the exit status is the command's, and
    nothing is interposed. The first line is printed either way, because a
    runtime that fails still says something worth recording.
    """
    return (
        f"sha256sum {shlex.quote(jar)} 2>/dev/null | cut -d' ' -f1; "
        f"if V=$({shlex.quote(java_bin)} -version 2>&1); then "
        f'printf "%s\n" "$V" | head -n 1; echo {JAVA_OK_MARK}; '
        f'else printf "%s\n" "$V" | head -n 1; fi'
    )


def build_cleanup_command(remote_dir: str, remote_files: Optional[list[str]] = None) -> str:
    """Compose the per-run cleanup, refusing anything that escapes remote_dir.

    Names come from GridFS documents, so a crafted file name must not be able
    to delete files shipped in the image.
    """
    targets = list(GENERATED_ARTIFACTS)
    for name in remote_files or []:
        if name.startswith("/") or ".." in name:
            log.warning("Skipping unsafe cleanup target: %s", name)
            continue
        targets.append(name)
    quoted = " ".join(shlex.quote(t) for t in targets)
    return f"cd {shlex.quote(remote_dir)} && rm -rf -- {quoted}"


class SSHTransport:
    """One SSH connection to one container, optionally through its host."""

    def __init__(
        self,
        plan: ConnectionPlan,
        target_credential: SSHCredential,
        jump_credential: Optional[SSHCredential] = None,
        *,
        jump_client: Optional[SSHClient] = None,
        pinned_host_key: Optional[str] = None,
        on_learn_host_key: Optional[callable] = None,
        connect_timeout: float = CONNECT_TIMEOUT,
        scp_timeout: float = SCP_SOCKET_TIMEOUT,
        transfer_attempts: int = TRANSFER_ATTEMPTS,
    ):
        if plan.is_tunnelled and jump_credential is None and jump_client is None:
            raise TransportError(
                f"{plan.describe()} needs a host credential for the first hop"
            )
        self.plan = plan
        self._target_credential = target_credential
        self._jump_credential = jump_credential
        # A hop 1 someone else owns: its host key was pinned and its handshake
        # paid when it was opened, and closing this transport must leave it
        # running, because the next container on the same host rides it too.
        self._borrowed_jump = jump_client
        self._pinned_host_key = pinned_host_key
        self._on_learn_host_key = on_learn_host_key
        self._connect_timeout = connect_timeout
        self._scp_timeout = scp_timeout
        self._transfer_attempts = max(1, transfer_attempts)
        self._jump: Optional[SSHClient] = None
        self._client: Optional[SSHClient] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def open(self) -> None:
        if self._client is not None:
            return
        sock = None
        try:
            if self.plan.is_tunnelled:
                jump = self._borrowed_jump
                if jump is None:
                    self._jump = self._connect_jump()
                    jump = self._jump
                sock = self._open_tunnel(jump)
            self._client = self._connect_target(sock)
        except BaseException:
            # A half-open pair would leak a file descriptor and, worse, leave a
            # reservation looking live while nothing is connected.
            self.close()
            raise
        log.info(
            "Transport open: %s%s",
            self.plan.describe(),
            " (on the host's shared session)" if self._borrowed_jump else "",
        )

    def _connect_jump(self) -> SSHClient:
        assert self.plan.jump is not None
        return connect_host(
            self.plan.jump,
            self._jump_credential,
            pinned_host_key=self._pinned_host_key,
            on_learn_host_key=self._remember_host_key,
            timeout=self._connect_timeout,
        )

    def _remember_host_key(self, fingerprint: str) -> None:
        """Keep a key learned on first contact, for this object's reconnects.

        The policy is built fresh for every connection, from
        ``_pinned_host_key``. Leaving that at ``None`` after learning meant a
        reconnection — and the follower reconnects, that is the point of it —
        started as a first contact again and would accept *any* key. The pin
        has to live as long as the transport that learned it.
        """
        self._pinned_host_key = fingerprint
        if self._on_learn_host_key is not None:
            self._on_learn_host_key(fingerprint)

    def _open_tunnel(self, jump: SSHClient):
        transport = jump.get_transport()
        if transport is None:  # pragma: no cover - defensive
            raise TransportError(f"no transport on the jump to {self.plan.jump}")
        try:
            return transport.open_channel(
                "direct-tcpip",
                dest_addr=(self.plan.target.address, self.plan.target.port),
                src_addr=("127.0.0.1", 0),
            )
        except SSHException as e:
            # "Administratively prohibited" is sshd refusing to open a TCP
            # forward, and it is the one failure here an operator cannot
            # diagnose from the message alone: the second hop is a
            # direct-tcpip channel, so the host's sshd must allow forwarding.
            # Upstream OpenSSH defaults to yes; several distributions ship it
            # as no, Alpine among them.
            hint = ""
            if "administratively prohibited" in str(e).lower():
                hint = (
                    " — the host's sshd is refusing to forward. Set "
                    "`AllowTcpForwarding yes` in its sshd_config and reload "
                    "it; the second hop into the container needs it."
                )
            raise TransportUnreachable(
                f"host {self.plan.jump} could not reach container "
                f"{self.plan.target}: {e}{hint}"
            ) from None

    def _connect_target(self, sock) -> SSHClient:
        client = paramiko.SSHClient()
        # Container keys are ephemeral and the tunnel is already authenticated
        # by hop 1 — see lib/transport/hostkey.py for why this is not pinned.
        client.set_missing_host_key_policy(TunnelledHostKeyPolicy())
        try:
            client.connect(
                self.plan.target.address,
                port=self.plan.target.port,
                timeout=self._connect_timeout,
                sock=sock,
                **self._target_credential.connect_kwargs(),
            )
        except (NoValidConnectionsError, OSError) as e:
            client.close()
            raise TransportUnreachable(
                f"container {self.plan.target} is unreachable: {e}"
            ) from None
        except SSHException as e:
            client.close()
            raise TransportError(
                f"SSH to container {self.plan.target} failed: {e}"
            ) from None
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(KEEPALIVE_INTERVAL)
        return client

    def close(self) -> None:
        # ``_borrowed_jump`` is deliberately absent: it belongs to whoever
        # lent it, and other containers on that host are still using it.
        for attr in ("_client", "_jump"):
            client = getattr(self, attr, None)
            if client is not None:
                try:
                    client.close()
                except Exception:  # pragma: no cover - best effort
                    log.debug("Error closing %s for %s", attr, self.plan.describe())
                setattr(self, attr, None)

    def __enter__(self) -> "SSHTransport":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def _active(self) -> SSHClient:
        if self._client is None:
            raise TransportError(f"transport for {self.plan.describe()} is not open")
        return self._client

    # ── files ──────────────────────────────────────────────────────────────

    def _scp(self) -> SCPClient:
        return SCPClient(
            self._active.get_transport(), socket_timeout=self._scp_timeout)

    def _transfer(self, describe: str, move: Callable[[SCPClient], None]) -> None:
        """Run one transfer, retrying a stalled link.

        Each attempt gets a fresh SCP channel: a timed-out one is in an
        unknown state, and reusing it would fail for the wrong reason.
        """
        last: BaseException | None = None
        for attempt in range(1, self._transfer_attempts + 1):
            try:
                with self._scp() as scp:
                    move(scp)
                if attempt > 1:
                    log.info("%s succeeded on attempt %d", describe, attempt)
                return
            except (SCPException, SSHException, OSError, TimeoutError) as error:
                last = error
                if _is_missing_file(error) or attempt == self._transfer_attempts:
                    break
                log.warning(
                    "%s failed (attempt %d of %d): %s; retrying in %.0fs",
                    describe, attempt, self._transfer_attempts,
                    _describe_error(error), TRANSFER_BACKOFF_SEC,
                )
                time.sleep(TRANSFER_BACKOFF_SEC)
        raise TransportError(
            f"{describe} failed after {attempt} attempt(s): "
            f"{_describe_error(last) if last else 'unknown'}"
        ) from None

    def put_files(self, local_paths: list[str], remote_names: list[str]) -> None:
        if len(local_paths) != len(remote_names):
            raise TransportError(
                f"{len(local_paths)} local files for {len(remote_names)} remote names"
            )
        for local, name in zip(local_paths, remote_names):
            remote = f"{self.plan.remote_dir}/{name}"
            log.info("Sending %s to %s", local, remote)
            # Per file, not per batch: a retry of the whole batch would resend
            # everything that already arrived.
            self._transfer(
                f"sending {local} to {self.plan.target}:{remote}",
                lambda scp, local=local, remote=remote: scp.put(local, remote),
            )

    def get_file(self, remote_name: str, local_path: str) -> None:
        remote = (
            remote_name if remote_name.startswith("/")
            else f"{self.plan.remote_dir}/{remote_name}"
        )
        self._transfer(
            f"fetching {remote}",
            lambda scp: scp.get(remote, local_path),
        )

    # ── commands ───────────────────────────────────────────────────────────

    def exec_command(self, command: str, timeout: Optional[float] = None) -> ExecResult:
        try:
            _, stdout, stderr = self._active.exec_command(command, timeout=timeout)
            out = stdout.read().decode("utf-8", errors="ignore")
            err = stderr.read().decode("utf-8", errors="ignore")
            return ExecResult(stdout.channel.recv_exit_status(), out, err)
        except SSHException as e:
            raise TransportError(f"command failed on {self.plan.target}: {e}") from None

    def stream_command(
        self, command: str, timeout: Optional[float] = None
    ) -> Iterator[tuple[str, str]]:
        """Run a command, yielding output as it arrives.

        Kept for interactive probes and short commands. The simulation itself
        no longer uses this: tying a multi-hour JVM to a channel makes a
        dropped tunnel fatal, so it is launched detached instead.
        """
        try:
            _, stdout, stderr = self._active.exec_command(
                command, get_pty=True, timeout=timeout)
        except SSHException as e:
            raise TransportError(f"command failed on {self.plan.target}: {e}") from None

        chan = stdout.channel
        chan.settimeout(5.0)
        while not chan.exit_status_ready():
            if chan.recv_ready():
                data = chan.recv(4096).decode("utf-8", errors="ignore")
                if data:
                    yield "stdout", data
            if chan.recv_stderr_ready():
                data = chan.recv_stderr(4096).decode("utf-8", errors="ignore")
                if data:
                    yield "stderr", data
            time.sleep(0.1)
        remaining_out = stdout.read().decode("utf-8", errors="ignore")
        if remaining_out:
            yield "stdout", remaining_out
        remaining_err = stderr.read().decode("utf-8", errors="ignore")
        if remaining_err:
            yield "stderr", remaining_err

    # ── readiness and hygiene ──────────────────────────────────────────────

    def probe(self) -> ProbeResult:
        """Check that this container can actually run a simulation.

        Being open is not enough to answer that: a container can accept SSH
        while its Cooja build is missing, or while the JVM that launches it
        cannot run. Each fact is reported separately so the GUI can say which
        one failed instead of only that something did.

        Readiness asks for all three. The Java version used to be *collected
        and ignored* — a container with a jar and a broken runtime was put in
        the pool, and the failure surfaced at launch, as a simulation marked
        in error for what was a configuration problem.
        """
        jar = f"{self.plan.remote_dir}/build/libs/cooja.jar"
        command = build_probe_command(jar)
        try:
            result = self.exec_command(command, timeout=30)
        except TransportUnreachable as e:
            return ProbeResult(reachable=False, message=str(e))
        except TransportError as e:
            return ProbeResult(reachable=True, message=str(e))

        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        sha = next((l for l in lines if len(l) == 64 and all(
            c in "0123456789abcdef" for c in l)), None)
        java = next((l for l in lines if "version" in l.lower()), None)
        java_ok = JAVA_OK_MARK in lines
        if sha is None:
            return ProbeResult(
                reachable=True,
                cooja_present=False,
                java_version=java,
                java_ok=java_ok,
                message=f"cooja.jar not found at {jar}",
            )
        if not java_ok:
            # The jar is there and nothing can start it. Saying so here is the
            # difference between a worker an operator can fix and a run that
            # fails for no stated reason.
            return ProbeResult(
                reachable=True,
                cooja_present=True,
                java_version=java,
                java_ok=False,
                cooja_jar_sha256=sha,
                message=(
                    "cooja.jar is there but the JVM did not run "
                    "(/opt/java/openjdk/bin/java)"
                ),
            )
        return ProbeResult(
            reachable=True,
            cooja_present=True,
            java_version=java,
            java_ok=True,
            cooja_jar_sha256=sha,
            message="ready",
        )

    def cleanup(self, remote_names: Optional[list[str]] = None) -> bool:
        """Remove per-run files. Best-effort: never aborts the simulation flow."""
        command = build_cleanup_command(self.plan.remote_dir, remote_names)
        try:
            result = self.exec_command(command, timeout=60)
        except TransportError as e:
            log.warning("Remote cleanup failed on %s: %s", self.plan.describe(), e)
            return False
        if not result.ok:
            log.warning(
                "Remote cleanup exited with status %s on %s: %s",
                result.exit_status, self.plan.describe(), result.stderr.strip(),
            )
            return False
        log.info("Remote cleanup done in %s", self.plan.remote_dir)
        return True
