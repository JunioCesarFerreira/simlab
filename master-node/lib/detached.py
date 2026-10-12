"""Running Cooja detached from the SSH channel, and following it.

Why
---
Cooja used to run through ``exec_command(get_pty=True)``, which ties the JVM's
lifetime to the channel: closing the channel sends SIGHUP and kills the
simulation. That was survivable while every target was a container on a local
bridge. It is not once the connection crosses a LAN and a tunnel for hours, and
it makes the specification's requirement — that losing contact is not the same
as the run failing — impossible to honour.

So the JVM is started in its own session, writes its output to a file, and
records both its PID and its exit code. Following it is then a matter of
polling, which survives the connection dropping and being re-established.

What the remote shell actually does
-----------------------------------
Each decision below came out of measuring the Cooja image rather than reasoning
about POSIX:

* **``setsid`` alone does not detach from the channel.** sshd keeps the
  channel's stdout open until the process it spawned exits, so the launch call
  blocked for the entire run — a 12-second job took 12 seconds to "start".
  Wrapping it so the intermediate shell exits first (``( setsid … & ) &``)
  makes the launch return in milliseconds.
* **``$$`` is not the shell's own PID under dash**, which is ``/bin/sh`` here:
  it is inherited from the parent, so it named a process that had already
  exited. The PID recorded is ``$!`` of the backgrounded work, which is the
  JVM itself — and therefore also the right thing for a later liveness probe.
* ``wait $!`` then ``$?`` captures the exit code, which the previous
  implementation discarded entirely.
* Liveness is ``kill -0``, guarded against an empty PID: ``[ -d /proc/ ]``
  with an empty value is a false positive.
* ``tail -c +N`` reads the log incrementally, so the live output the old
  implementation streamed is preserved.

The portable wrapper is used rather than ``setsid --fork``: both work here, but
``--fork`` is a util-linux flag that busybox does not have, and a remote host
is not guaranteed to be Ubuntu.
"""
import logging
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# Per-run bookkeeping files, alongside the simulation inside remote_dir.
OUT_FILE = "run.out"
PID_FILE = "run.pid"
EXIT_FILE = "run.exit"
RUN_ARTIFACTS = (OUT_FILE, PID_FILE, EXIT_FILE)

__all__ = [
    "OUT_FILE", "PID_FILE", "EXIT_FILE", "RUN_ARTIFACTS", "STATUS_PREFIX",
    "DetachedRun", "DetachedRunError", "RunHandle", "RunOutcome", "RunProgress",
    "build_launch_command", "build_poll_command", "build_liveness_command",
    "parse_poll_output", "is_running",
]

# Prefix of the status line, chosen so it cannot be confused with Cooja output.
STATUS_PREFIX = "SIMLABRUN"
_STATUS_RE = re.compile(rf"^{STATUS_PREFIX} (\S+) (\d) (\d+)$")

DEFAULT_POLL_INTERVAL_SEC = 5.0
# Grace between asking the JVM to stop and killing it outright.
TERMINATE_GRACE_SEC = 10.0
# Backoff bounds for re-establishing a dropped connection.
RECONNECT_MIN_SEC = 2.0
RECONNECT_MAX_SEC = 30.0


class DetachedRunError(RuntimeError):
    """The run could not be started, or its bookkeeping is unreadable."""


@dataclass(frozen=True)
class RunHandle:
    """What is needed to follow, adopt or stop a detached run."""
    pid: int
    remote_dir: str


@dataclass(frozen=True)
class RunProgress:
    """One observation of a detached run."""
    finished: bool
    alive: bool
    exit_code: Optional[int]
    output: str
    log_size: int


@dataclass(frozen=True)
class RunOutcome:
    """How following a run ended."""
    completed: bool
    exit_code: Optional[int] = None
    timed_out: bool = False
    lost_contact: bool = False
    elapsed_sec: float = 0.0
    message: str = ""


@dataclass
class _Follow:
    """Mutable state of one following loop, kept out of the class surface."""
    offset: int = 1              # tail -c is 1-indexed
    disconnected_since: Optional[float] = None
    backoff: float = RECONNECT_MIN_SEC
    reconnects: int = 0
    warned: set = field(default_factory=set)


def build_launch_command(
    command: str,
    *,
    remote_dir: str,
    out_file: str = OUT_FILE,
    pid_file: str = PID_FILE,
    exit_file: str = EXIT_FILE,
) -> str:
    """Compose the shell that starts ``command`` detached and records it.

    ``rm -f`` first: a stale exit file from a previous run would make the very
    first poll report the new run as already finished.
    """
    inner = (
        f"{command} > {out_file} 2>&1 & "
        f"echo $! > {pid_file}; "
        f"wait $!; "
        f"echo $? > {exit_file}"
    )
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"rm -f {out_file} {pid_file} {exit_file} && "
        # The subshell is what lets sshd close the channel: without it the
        # launch call blocks for the whole run.
        f"( setsid /bin/sh -c {shlex.quote(inner)} < /dev/null > /dev/null 2>&1 & ) & "
        f"echo {STATUS_PREFIX}-launched"
    )


def build_poll_command(
    offset: int,
    *,
    remote_dir: str,
    out_file: str = OUT_FILE,
    pid_file: str = PID_FILE,
    exit_file: str = EXIT_FILE,
    max_chunk: int = 64 * 1024,
) -> str:
    """One round trip that answers "finished? alive?" and returns new output.

    The status comes first on its own line, so only the first line is ever
    parsed and Cooja's output cannot be mistaken for it. Batched into a single
    command because, over a tunnel, round trips are the expensive part.
    """
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"S=$([ -f {exit_file} ] && cat {exit_file} 2>/dev/null || echo -); "
        f"P=$([ -f {pid_file} ] && cat {pid_file} 2>/dev/null || echo -); "
        # The -n guard matters: `kill -0 ''` and `[ -d /proc/ ]` both succeed.
        f'A=$(if [ -n "$P" ] && [ "$P" != - ] && kill -0 "$P" 2>/dev/null; '
        f'then echo 1; else echo 0; fi); '
        f"N=$(wc -c < {out_file} 2>/dev/null || echo 0); "
        f'echo "{STATUS_PREFIX} ${{S:--}} $A ${{N:-0}}"; '
        f"tail -c +{offset} {out_file} 2>/dev/null | head -c {max_chunk}"
    )


def parse_poll_output(stdout: str) -> RunProgress:
    """Split the status line from the log bytes that follow it."""
    status_line, _, output = stdout.partition("\n")
    match = _STATUS_RE.match(status_line.strip())
    if match is None:
        raise DetachedRunError(
            f"unexpected status line from the remote shell: {status_line[:120]!r}"
        )
    raw_exit, alive, size = match.groups()
    exit_code: Optional[int] = None
    finished = raw_exit != "-"
    if finished:
        try:
            exit_code = int(raw_exit)
        except ValueError:
            # The file exists but holds something unexpected. Treat the run as
            # finished — it clearly ended — with an unknown status.
            log.warning("Exit file held %r, not a number", raw_exit)
    return RunProgress(
        finished=finished,
        alive=alive == "1",
        exit_code=exit_code,
        output=output,
        log_size=int(size),
    )


class DetachedRun:
    """Starts a command in its own session and follows it to completion.

    Depends only on a transport that can ``exec_command``, plus ``close`` and
    ``open`` for reconnecting — so it can be driven in tests with neither a
    network nor a container.
    """

    def __init__(
        self,
        transport: Any,
        *,
        remote_dir: Optional[str] = None,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        exec_timeout_sec: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport
        self._remote_dir = remote_dir or getattr(
            getattr(transport, "plan", None), "remote_dir", ".")
        self._poll_interval = poll_interval_sec
        self._exec_timeout = exec_timeout_sec
        self._sleep = sleep
        self._clock = clock
        self._handle: Optional[RunHandle] = None

    @property
    def handle(self) -> Optional[RunHandle]:
        return self._handle

    # ── starting ───────────────────────────────────────────────────────────

    def launch(self, command: str) -> RunHandle:
        """Start the command detached and read back its PID.

        The PID is read in a second call rather than printed by the launch:
        the launch's own shell exits immediately by design, so it cannot know
        the PID the inner shell recorded.
        """
        result = self._transport.exec_command(
            build_launch_command(command, remote_dir=self._remote_dir),
            timeout=self._exec_timeout,
        )
        if not result.ok:
            raise DetachedRunError(
                f"could not start the run (exit {result.exit_status}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        pid = self._read_pid()
        self._handle = RunHandle(pid=pid, remote_dir=self._remote_dir)
        log.info("Started detached run, pid=%s in %s", pid, self._remote_dir)
        return self._handle

    def _read_pid(self, attempts: int = 10, delay: float = 0.3) -> int:
        """Wait briefly for the inner shell to record its PID.

        The launch returns before the inner shell has necessarily written the
        file, so this is a short poll rather than a single read.
        """
        command = (
            f"cd {shlex.quote(self._remote_dir)} && "
            f"cat {PID_FILE} 2>/dev/null || true"
        )
        for attempt in range(attempts):
            result = self._transport.exec_command(command, timeout=self._exec_timeout)
            raw = result.stdout.strip()
            if raw.isdigit():
                return int(raw)
            if attempt < attempts - 1:
                self._sleep(delay)
        raise DetachedRunError(
            "the run did not record a PID; it may have failed to start "
            f"(last read: {raw!r})"
        )

    # ── following ──────────────────────────────────────────────────────────

    def poll(self, offset: int = 1) -> RunProgress:
        """One observation. Raises the transport's error if unreachable."""
        result = self._transport.exec_command(
            build_poll_command(offset, remote_dir=self._remote_dir),
            timeout=self._exec_timeout,
        )
        if not result.ok and not result.stdout.strip():
            raise DetachedRunError(
                f"polling failed (exit {result.exit_status}): {result.stderr.strip()}"
            )
        return parse_poll_output(result.stdout)

    def wait(
        self,
        *,
        timeout_sec: Optional[float] = None,
        on_output: Optional[Callable[[str], None]] = None,
    ) -> RunOutcome:
        """Follow the run until it ends, the timeout expires, or contact is lost.

        A dropped connection is retried with backoff rather than treated as a
        failed run — that distinction is the whole reason this module exists.
        The timeout still applies while disconnected, so an unreachable host
        cannot hold a container indefinitely.
        """
        if self._handle is None:
            raise DetachedRunError("nothing has been launched yet")

        started = self._clock()
        state = _Follow()

        while True:
            elapsed = self._clock() - started
            if timeout_sec and elapsed >= timeout_sec:
                return self._on_timeout(elapsed, state)

            try:
                progress = self.poll(state.offset)
            except Exception as error:               # transport or shell failure
                if not self._handle_disconnect(error, state, elapsed):
                    return RunOutcome(
                        completed=False, lost_contact=True, elapsed_sec=elapsed,
                        message=f"lost contact with the run: {error}",
                    )
                continue

            self._note_reconnected(state)
            if progress.output:
                state.offset += len(progress.output.encode("utf-8", errors="ignore"))
                if on_output is not None:
                    on_output(progress.output)

            if progress.finished:
                log.info(
                    "Detached run finished with exit code %s after %.0fs",
                    progress.exit_code, elapsed,
                )
                return RunOutcome(
                    completed=True, exit_code=progress.exit_code, elapsed_sec=elapsed)

            if not progress.alive:
                # The process is gone but no exit file was written: it was
                # killed from outside, or the container was restarted.
                return RunOutcome(
                    completed=False, elapsed_sec=elapsed,
                    message="the run process disappeared without recording an exit code",
                )

            self._sleep(self._poll_interval)

    def _handle_disconnect(self, error: Exception, state: _Follow, elapsed: float) -> bool:
        """Try to re-establish the connection. False means give up."""
        if state.disconnected_since is None:
            state.disconnected_since = self._clock()
            log.warning(
                "Lost contact while following the run (%s); retrying — the run "
                "itself is detached and is assumed to still be going", error,
            )
        try:
            self._transport.close()
            self._transport.open()
        except Exception as reopen_error:
            log.debug("Reconnect attempt failed: %s", reopen_error)
        self._sleep(state.backoff)
        state.backoff = min(state.backoff * 2, RECONNECT_MAX_SEC)
        state.reconnects += 1
        return True

    def _note_reconnected(self, state: _Follow) -> None:
        if state.disconnected_since is not None:
            log.info(
                "Contact re-established after %.0fs and %d attempt(s); "
                "the run was unaffected",
                self._clock() - state.disconnected_since, state.reconnects,
            )
            state.disconnected_since = None
            state.backoff = RECONNECT_MIN_SEC
            state.reconnects = 0

    def _on_timeout(self, elapsed: float, state: _Follow) -> RunOutcome:
        log.warning("Detached run exceeded its timeout after %.0fs; stopping it", elapsed)
        stopped = self.terminate()
        return RunOutcome(
            completed=False,
            timed_out=True,
            lost_contact=state.disconnected_since is not None,
            elapsed_sec=elapsed,
            message=(
                "the simulation exceeded SIM_TIMEOUT_SEC"
                if stopped else
                "the simulation exceeded SIM_TIMEOUT_SEC and could not be stopped"
            ),
        )

    # ── stopping ───────────────────────────────────────────────────────────

    def terminate(self, grace_sec: float = TERMINATE_GRACE_SEC) -> bool:
        """Ask the run to stop, then insist. Best-effort.

        Signals the process group, not just the process: the JVM may have
        spawned children, and leaving them behind would hold the container's
        resources after the simulation is over.
        """
        if self._handle is None:
            return False
        pid = self._handle.pid
        command = (
            f"kill -TERM -{pid} 2>/dev/null || kill -TERM {pid} 2>/dev/null; "
            f"sleep {max(1, int(grace_sec))}; "
            f"kill -KILL -{pid} 2>/dev/null || kill -KILL {pid} 2>/dev/null; "
            "true"
        )
        try:
            self._transport.exec_command(
                command, timeout=self._exec_timeout + grace_sec)
            return True
        except Exception as error:
            log.warning("Could not stop the run (pid %s): %s", pid, error)
            return False


def build_liveness_command(pid: int, *, remote_dir: str) -> str:
    """Ask whether a recorded PID is still running.

    The same guarded ``kill -0`` the poll uses: an empty or non-numeric value
    must read as dead, because ``kill -0 ''`` succeeds and would report a
    finished run as alive.
    """
    return (
        f"cd {shlex.quote(remote_dir)} 2>/dev/null; "
        f'P={shlex.quote(str(pid))}; '
        f'if [ -n "$P" ] && [ "$P" != - ] && kill -0 "$P" 2>/dev/null; '
        f"then echo {STATUS_PREFIX}-alive; else echo {STATUS_PREFIX}-dead; fi"
    )


def is_running(transport: Any, pid: Optional[int], *, remote_dir: str) -> bool:
    """Whether the run with this PID is still going.

    Raises whatever the transport raises when the container cannot be reached:
    "unreachable" and "not running" are different answers, and the caller has
    to tell them apart — treating the first as the second is what would turn a
    network blip into a destroyed simulation.
    """
    if not pid:
        return False
    result = transport.exec_command(
        build_liveness_command(int(pid), remote_dir=remote_dir), timeout=30)
    return f"{STATUS_PREFIX}-alive" in result.stdout
