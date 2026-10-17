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
# The process *group*, which is what a signal has to reach to take the JVM's
# children with it. Recorded separately because it is not the pid: `$!` is the
# command, and the group is led by the shell that supervises it.
PGID_FILE = "run.pgid"
RUN_ARTIFACTS = (OUT_FILE, PID_FILE, EXIT_FILE, PGID_FILE)

__all__ = [
    "OUT_FILE", "PID_FILE", "EXIT_FILE", "PGID_FILE", "RUN_ARTIFACTS",
    "STATUS_PREFIX",
    "DetachedRun", "DetachedRunError", "LaunchUnconfirmed", "RunHandle", "RunOutcome", "RunProgress",
    "build_launch_command", "build_poll_command", "build_inspect_command",
    "build_live_check_command", "assert_no_live_run", "LiveRunPresent",
    "parse_poll_output", "parse_inspect_output", "inspect_run", "is_running",
    "RunState",
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


class LaunchUnconfirmed(DetachedRunError):
    """The launch may have started something, and that something is still up.

    ``launch`` sends a shell that backgrounds the JVM and then reads the PID
    the shell recorded. Those are two round trips, and the second one failing
    says nothing about the first: the JVM may be running perfectly well with
    nobody holding its pid. Reported as a plain failure, the container went
    back to the pool with a simulation on it — "error while launching" is not
    evidence that no launch occurred.

    Raised only when the group could not be stopped either. A confirmed stop
    turns this back into an ordinary start-up failure, which it then is.

    Covers three moments, because all three say the same thing about the JVM:
    the launch command was sent and contact was lost, the shell answered but
    the pid file never appeared, and the pid read itself lost contact. Only the
    second was classified before, so a dropped connection around the launch
    released the worker.
    """


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
    #: Whether stopping the run was confirmed. ``None`` when no stop was
    #: attempted. ``False`` is the case that matters: something may still be
    #: running in the container, so handing it to the next simulation would
    #: have two JVMs competing and silently corrupt the newcomer's timing.
    stopped: Optional[bool] = None


# How many consecutive polls must report the process gone *and* no exit code
# before the run is called failed, and how long to wait between those looks.
#
# There is a real gap between the two facts. `kill -0` succeeds for a zombie,
# so the pid stops existing at the exact moment the supervising shell reaps it
# in `wait $!` — and `echo $? > run.exit` runs microseconds to milliseconds
# later, longer when the machine is loaded. A poll landing in that gap sees
# "gone, with no exit code" for a run that has just finished perfectly well.
#
# Measured on a real container: 4 of 25 one-second runs passed through that
# state when polled every ~45ms, which puts the window around 5–10ms. At the
# production interval that is roughly one simulation in seven hundred — and it
# cost generation 0 of pilot 6acf822dbe51fa7df2345a83 a completed simulation.
VANISHED_CONFIRMATIONS = 3
VANISHED_RECHECK_SEC = 1.0


@dataclass
class _Follow:
    """Mutable state of one following loop, kept out of the class surface."""
    offset: int = 1              # tail -c is 1-indexed
    disconnected_since: Optional[float] = None
    backoff: float = RECONNECT_MIN_SEC
    reconnects: int = 0
    warned: set = field(default_factory=set)
    #: Consecutive polls that found the process gone without an exit code.
    vanished: int = 0


def build_launch_command(
    command: str,
    *,
    remote_dir: str,
    out_file: str = OUT_FILE,
    pid_file: str = PID_FILE,
    exit_file: str = EXIT_FILE,
    pgid_file: str = PGID_FILE,
) -> str:
    """Compose the shell that starts ``command`` detached and records it.

    ``rm -f`` first: a stale exit file from a previous run would make the very
    first poll report the new run as already finished.

    Two identifiers are written, and the difference matters. ``run.pid`` is
    ``$!``, the command itself — what a liveness check asks about. ``run.pgid``
    is ``$$``, the supervising shell, which ``setsid`` has just made the
    leader of a new session and process group; the JVM and everything it
    spawns belong to that group. Signalling the pid as though it were a group
    does nothing at all: verified on a container, ``kill -TERM -<pid>`` answers
    "No such process" and every child survives, while the same signal to the
    group takes them all.
    """
    inner = (
        f"echo $$ > {pgid_file}; "
        f"{command} > {out_file} 2>&1 & "
        f"echo $! > {pid_file}; "
        f"wait $!; "
        f"echo $? > {exit_file}"
    )
    return (
        f"cd {shlex.quote(remote_dir)} && "
        f"rm -f {out_file} {pid_file} {exit_file} {pgid_file} && "
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
        vanished_confirmations: int = VANISHED_CONFIRMATIONS,
        vanished_recheck_sec: float = VANISHED_RECHECK_SEC,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport
        self._remote_dir = remote_dir or getattr(
            getattr(transport, "plan", None), "remote_dir", ".")
        self._poll_interval = poll_interval_sec
        self._exec_timeout = exec_timeout_sec
        self._vanished_confirmations = max(1, vanished_confirmations)
        self._vanished_recheck_sec = vanished_recheck_sec
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
        try:
            result = self._transport.exec_command(
                build_launch_command(command, remote_dir=self._remote_dir),
                timeout=self._exec_timeout,
            )
        except Exception as error:
            # The command was *sent*. Whether it arrived is exactly what we do
            # not know: a connection that drops after sshd accepted it leaves
            # a JVM running with nobody holding its pid. Reported as a
            # transport failure, this released the worker (R05).
            if self._stop_by_group():
                raise
            raise LaunchUnconfirmed(
                f"contact was lost while starting the run ({error}), and "
                "nothing could be stopped, so something may be executing in "
                f"{self._remote_dir}"
            ) from error
        if not result.ok:
            # The remote shell answered and said no. Nothing started.
            raise DetachedRunError(
                f"could not start the run (exit {result.exit_status}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

        try:
            pid = self._read_pid()
        except Exception:
            # The launch command itself succeeded, so the inner shell ran and
            # may have started the JVM. Before calling this a failed start,
            # stop whatever it left: the group file is written *before* the
            # command, so there is normally something to signal, and its
            # absence means the inner shell never got that far.
            #
            # Every failure, not just `DetachedRunError`: a transport error
            # while reading the pid says just as little about the JVM, and
            # catching only the one class let it through to the generic
            # handler, which released the worker.
            if self._stop_by_group():
                raise
            raise LaunchUnconfirmed(
                "the run recorded no PID and the process group could not be "
                "stopped, so something may still be executing in "
                f"{self._remote_dir}"
            ) from None
        self._handle = RunHandle(pid=pid, remote_dir=self._remote_dir)
        log.info("Started detached run, pid=%s in %s", pid, self._remote_dir)
        return self._handle

    def attach(self, pid: int) -> RunHandle:
        """Take over following a run that is already going.

        Used after a restart, to pick up a simulation that outlived the
        master-node. Nothing is started and nothing is reset: the run is
        simply followed from where it is, which is the whole point of having
        recorded its PID.
        """
        self._handle = RunHandle(pid=int(pid), remote_dir=self._remote_dir)
        log.info("Attached to running pid %s in %s", pid, self._remote_dir)
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

    def _stop_by_group(self, grace_sec: float = TERMINATE_GRACE_SEC) -> bool:
        """Stop what the launch started, knowing only its process group.

        Used when the PID could not be read. Returns true when nothing is
        left — including when no group was ever recorded, which means the
        inner shell never reached its first statement and therefore started
        nothing.
        """
        grace = max(1, int(grace_sec))
        command = (
            f"cd {shlex.quote(self._remote_dir)} 2>/dev/null; "
            f"G=$(cat {PGID_FILE} 2>/dev/null); "
            f'if [ -z "$G" ]; then echo {STATUS_PREFIX}-gone; exit 0; fi; '
            f'kill -TERM -"$G" 2>/dev/null; sleep {grace}; '
            f'kill -KILL -"$G" 2>/dev/null; sleep 1; '
            f'if kill -0 -"$G" 2>/dev/null; '
            f"then echo {STATUS_PREFIX}-alive; else echo {STATUS_PREFIX}-gone; fi"
        )
        try:
            result = self._transport.exec_command(
                command, timeout=self._exec_timeout + grace_sec + 2)
        except Exception as error:
            log.warning("Could not stop the launched group: %s", error)
            return False
        return f"{STATUS_PREFIX}-gone" in (result.stdout or "")

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
                if state.vanished:
                    # It had not died; the exit code was simply a moment behind
                    # the pid. Worth a line, because this is the race that used
                    # to be reported as a failed simulation.
                    log.info(
                        "The exit code for this run arrived %d look(s) after "
                        "the process was reaped", state.vanished,
                    )
                log.info(
                    "Detached run finished with exit code %s after %.0fs",
                    progress.exit_code, elapsed,
                )
                return RunOutcome(
                    completed=True, exit_code=progress.exit_code, elapsed_sec=elapsed)

            if not progress.alive:
                # Gone, with no exit code — but not necessarily dead. The pid
                # stops existing the moment the supervising shell reaps it,
                # and the exit file is written just after, so this is also
                # what a *successful* finish looks like for a few
                # milliseconds. Look again before throwing the run away.
                state.vanished += 1
                if state.vanished >= self._vanished_confirmations:
                    return RunOutcome(
                        completed=False, elapsed_sec=elapsed,
                        message=(
                            "the run process disappeared without recording an "
                            f"exit code (confirmed over {state.vanished} looks)"
                        ),
                    )
                log.warning(
                    "The run process is gone with no exit code yet "
                    "(look %d of %d); checking again in %.0fs",
                    state.vanished, self._vanished_confirmations,
                    self._vanished_recheck_sec,
                )
                self._sleep(self._vanished_recheck_sec)
                continue

            state.vanished = 0
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
            stopped=stopped,
            message=(
                "the simulation exceeded SIM_TIMEOUT_SEC"
                if stopped else
                "the simulation exceeded SIM_TIMEOUT_SEC and could not be stopped"
            ),
        )

    # ── stopping ───────────────────────────────────────────────────────────

    def terminate(self, grace_sec: float = TERMINATE_GRACE_SEC) -> bool:
        """Stop the run and its children, and say whether they are gone.

        Signals the process *group*, read from ``run.pgid``, because the JVM
        spawns children and leaving them behind would keep using the
        container after the simulation is over. The group is led by the
        supervising shell, not by the recorded pid — signalling the pid as a
        group does nothing, which is what the previous version did.

        Returns true only when nothing is left alive. The old version
        returned true whenever the command *executed*, so a caller could not
        tell "stopped" from "tried to stop", and the container was released
        either way. A run launched before this file existed has no
        ``run.pgid``; the pid is then used as a fallback, which is the old
        behaviour and all that is available.
        """
        if self._handle is None:
            return False
        pid = self._handle.pid
        quoted_dir = shlex.quote(self._remote_dir)
        grace = max(1, int(grace_sec))
        command = (
            f"cd {quoted_dir} 2>/dev/null; "
            # The group if it was recorded, the pid if this run predates it.
            f"G=$(cat {PGID_FILE} 2>/dev/null); "
            f'[ -n "$G" ] || G={shlex.quote(str(pid))}; '
            f'kill -TERM -"$G" 2>/dev/null || kill -TERM {pid} 2>/dev/null; '
            f"sleep {grace}; "
            f'kill -KILL -"$G" 2>/dev/null || kill -KILL {pid} 2>/dev/null; '
            "sleep 1; "
            # Report what survived, so the caller is not left guessing.
            f'if kill -0 -"$G" 2>/dev/null || kill -0 {pid} 2>/dev/null; '
            f"then echo {STATUS_PREFIX}-alive; else echo {STATUS_PREFIX}-gone; fi"
        )
        try:
            result = self._transport.exec_command(
                command, timeout=self._exec_timeout + grace_sec + 2)
        except Exception as error:
            log.warning("Could not stop the run (pid %s): %s", pid, error)
            return False

        if f"{STATUS_PREFIX}-gone" in result.stdout:
            return True
        log.warning(
            "Asked the run (pid %s) to stop, but something is still alive: %s",
            pid, result.stdout.strip()[:120] or "no answer",
        )
        return False


@dataclass
class RunState:
    """What a container says about a run nobody is following.

    Four answers, not two. "Is the process alive?" was all this used to ask,
    and a dead process was taken for a failed simulation — so a run that
    finished perfectly well while the master-node was down had its result
    thrown away, and the next job overwrote the artifacts that proved it.

    * ``alive``      — still going; adopt it and follow from here.
    * ``finished``   — ``run.exit`` is there, so it ended on its own terms.
      The exit code says how, and the testlog is still on disk: collectable.
    * neither, with artifacts — ended without recording an exit code. The
      testlog may still be worth collecting; see :attr:`collectable`.
    * neither, nothing — interrupted with nothing to show.

    "Unreachable" is not among them: it is the absence of an answer, and the
    caller must keep it separate, because treating it as "not running" is what
    would turn a network blip into a destroyed simulation.
    """
    alive: bool
    finished: bool
    exit_code: Optional[int] = None
    testlog_bytes: int = 0

    @property
    def collectable(self) -> bool:
        """Whether there is a result worth fetching before giving up.

        A finished run with a testlog, or — more cautiously — one that
        vanished but left a testlog behind. In both cases the bytes are on the
        container and the alternative is to delete them.
        """
        return self.testlog_bytes > 0 and not self.alive


def build_inspect_command(
    pid: int,
    *,
    remote_dir: str,
    exit_file: str = EXIT_FILE,
    testlog: str = "COOJA.testlog",
) -> str:
    """One round trip that answers alive, finished, and what is left behind.

    The guarded ``kill -0`` is the poll's: an empty or non-numeric value must
    read as dead, because ``kill -0 ''`` succeeds and would report a finished
    run as alive.
    """
    return (
        f"cd {shlex.quote(remote_dir)} 2>/dev/null; "
        f'P={shlex.quote(str(pid))}; '
        f'A=$(if [ -n "$P" ] && [ "$P" != - ] && kill -0 "$P" 2>/dev/null; '
        f"then echo 1; else echo 0; fi); "
        f"S=$([ -f {exit_file} ] && cat {exit_file} 2>/dev/null || echo -); "
        f"N=$(wc -c < {testlog} 2>/dev/null || echo 0); "
        f'echo "{STATUS_PREFIX} ${{S:--}} $A ${{N:-0}}"'
    )


def parse_inspect_output(stdout: str) -> RunState:
    """Read the status line ``build_inspect_command`` prints."""
    match = _STATUS_RE.match(stdout.strip().splitlines()[0].strip() if stdout.strip() else "")
    if match is None:
        raise DetachedRunError(
            f"unexpected status line from the remote shell: {stdout[:120]!r}")
    raw_exit, alive, size = match.groups()
    exit_code: Optional[int] = None
    finished = raw_exit != "-"
    if finished:
        try:
            exit_code = int(raw_exit)
        except ValueError:
            log.warning("Exit file held %r, not a number", raw_exit)
    return RunState(
        alive=alive == "1",
        finished=finished,
        exit_code=exit_code,
        testlog_bytes=int(size),
    )


class LiveRunPresent(DetachedRunError):
    """A run is still executing in a directory something is about to disturb.

    Raised by :func:`assert_no_live_run`. It means the database and the
    container disagree: the reservation says this worker is ours and nothing
    has been started on it, and the container says a JVM is running. Neither
    cleaning the directory nor uploading into it is safe, so the caller must
    treat the worker as needing a person rather than as free.
    """


def build_live_check_command(*, remote_dir: str, pid_file: str = PID_FILE) -> str:
    """One round trip: is the pid recorded in this directory still alive?

    The same guarded ``kill -0`` as everywhere: an empty or non-numeric value
    must read as dead, because ``kill -0 ''`` succeeds.
    """
    return (
        f"cd {shlex.quote(remote_dir)} 2>/dev/null; "
        f"P=$(cat {pid_file} 2>/dev/null); "
        f'if [ -n "$P" ] && [ "$P" != - ] && kill -0 "$P" 2>/dev/null; '
        f"then echo {STATUS_PREFIX}-live; else echo {STATUS_PREFIX}-clear; fi"
    )


def assert_no_live_run(transport: Any, *, remote_dir: Optional[str] = None) -> None:
    """Refuse to disturb a directory in which a run is still executing.

    The run directory is shared by every attempt that lands on a worker, and
    the first thing a new run does is a defensive ``rm -rf`` of the previous
    one's artifacts. That is correct when the previous run is over and
    destructive when it is not — and "is it over?" was never asked: the
    reservation was taken as proof, which it is not. An attempt whose lease
    expired while it was uploading, a JVM that outlived a release, a worker
    handed on by a recovery pass: in each case two attempts share one
    directory and the newcomer deletes the other's testlog.

    One extra round trip buys the check. It is deliberately a *refusal* and
    not a cleanup: if something is alive in there, no automatic action is
    safe, and the caller takes the worker out of the pool instead.

    Silent when the container cannot answer — this guards against a *known*
    live process, and an unreachable container is handled by the transport
    errors that follow.
    """
    target = remote_dir or getattr(
        getattr(transport, "plan", None), "remote_dir", ".")
    try:
        result = transport.exec_command(
            build_live_check_command(remote_dir=target), timeout=30)
    except Exception as error:
        log.warning("Could not check %s for a live run: %s", target, error)
        return
    if f"{STATUS_PREFIX}-live" in (result.stdout or ""):
        raise LiveRunPresent(
            f"a process recorded in {target} is still running, so the "
            "directory must not be cleaned or written to"
        )


def inspect_run(transport: Any, pid: Optional[int], *, remote_dir: str) -> RunState:
    """Ask the container about a run, in one round trip.

    Raises whatever the transport raises when the container cannot be reached:
    unreachable is the absence of an answer, not an answer.
    """
    if not pid:
        return RunState(alive=False, finished=False)
    result = transport.exec_command(
        build_inspect_command(int(pid), remote_dir=remote_dir), timeout=30)
    return parse_inspect_output(result.stdout)


def is_running(transport: Any, pid: Optional[int], *, remote_dir: str) -> bool:
    """Whether the run with this PID is still going.

    Kept as the narrow question, for callers that only need that. Anything
    deciding what to *do* about a run should use :func:`inspect_run`, because
    "not alive" alone cannot tell a finished run from an interrupted one.
    """
    return inspect_run(transport, pid, remote_dir=remote_dir).alive
