"""Tests for running Cooja detached from the SSH channel.

The behaviours here were each derived from measuring the Cooja image, and each
one would be invisible in a unit test that only checked the happy path — so
the shape of the remote shell is asserted explicitly:

  * the launch must not block the channel (the subshell wrapper);
  * the PID recorded must be the work's, not ``$$``;
  * an empty PID must not read as alive;
  * a dropped connection is retried, never reported as a failed run;
  * the timeout stops the process instead of abandoning it.
"""
import pytest

from lib.detached import (
    EXIT_FILE,
    OUT_FILE,
    PID_FILE,
    STATUS_PREFIX,
    DetachedRun,
    DetachedRunError,
    RunHandle,
    build_launch_command,
    build_poll_command,
    parse_poll_output,
)
from lib.transport.base import ExecResult

REMOTE_DIR = "/opt/contiki-ng/tools/cooja"
JAVA = "/opt/java/openjdk/bin/java -jar build/libs/cooja.jar --no-gui simulation.csc"


def status(exit_code="-", alive=1, size=0, log="") -> str:
    line = f"{STATUS_PREFIX} {exit_code} {alive} {size}"
    return f"{line}\n{log}"


class FakeTransport:
    """Replays scripted results and records what was asked."""

    def __init__(self, results=None):
        self.commands: list[str] = []
        self.results = list(results or [])
        self.default = ExecResult(0, status(), "")
        self.opened = 0
        self.closed = 0
        self.fail_next = 0

    def exec_command(self, command, timeout=None) -> ExecResult:
        self.commands.append(command)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("channel closed")
        if self.results:
            nxt = self.results.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return self.default

    def open(self):
        self.opened += 1

    def close(self):
        self.closed += 1


def build_run(transport, **kwargs) -> DetachedRun:
    return DetachedRun(
        transport,
        remote_dir=REMOTE_DIR,
        poll_interval_sec=0,
        sleep=lambda _s: None,
        clock=kwargs.pop("clock", None) or _steady_clock(),
        **kwargs,
    )


def _steady_clock():
    state = {"t": 0.0}

    def clock():
        state["t"] += 1.0
        return state["t"]

    return clock


# ── The launch shell ─────────────────────────────────────────────────────────

class TestLaunchCommand:
    def test_wraps_in_a_subshell_so_the_channel_can_close(self):
        command = build_launch_command(JAVA, remote_dir=REMOTE_DIR)
        # Measured: `setsid` alone leaves sshd holding the channel open until
        # the process exits, so the launch call blocked for the entire run.
        assert "( setsid /bin/sh -c " in command
        assert command.rstrip().endswith(f"echo {STATUS_PREFIX}-launched")
        assert "& ) &" in command

    def test_records_the_work_pid_not_the_shell(self):
        command = build_launch_command(JAVA, remote_dir=REMOTE_DIR)
        # `$$` is inherited from the parent under dash, so it named a process
        # that had already exited. `$!` is the backgrounded work itself.
        assert f"echo $! > {PID_FILE}" in command
        assert "$$" not in command

    def test_captures_the_exit_code(self):
        command = build_launch_command(JAVA, remote_dir=REMOTE_DIR)
        assert "wait $!" in command
        assert f"echo $? > {EXIT_FILE}" in command

    def test_clears_stale_bookkeeping_first(self):
        # A leftover exit file would make the first poll report the new run as
        # already finished.
        command = build_launch_command(JAVA, remote_dir=REMOTE_DIR)
        assert f"rm -f {OUT_FILE} {PID_FILE} {EXIT_FILE}" in command

    def test_detaches_from_the_sessions_streams(self):
        command = build_launch_command(JAVA, remote_dir=REMOTE_DIR)
        assert "< /dev/null > /dev/null 2>&1" in command

    def test_the_remote_dir_is_quoted(self):
        command = build_launch_command(JAVA, remote_dir="/opt/dir with space")
        assert "'/opt/dir with space'" in command

    def test_output_goes_to_a_file_not_the_channel(self):
        command = build_launch_command(JAVA, remote_dir=REMOTE_DIR)
        assert f"{JAVA} > {OUT_FILE} 2>&1 &" in command


# ── The poll shell ───────────────────────────────────────────────────────────

class TestPollCommand:
    def test_status_comes_before_the_log(self):
        command = build_poll_command(1, remote_dir=REMOTE_DIR)
        # Only the first line is parsed, so Cooja output can never be mistaken
        # for the status.
        assert command.index(f'echo "{STATUS_PREFIX}') < command.index("tail -c")

    def test_an_empty_pid_cannot_read_as_alive(self):
        command = build_poll_command(1, remote_dir=REMOTE_DIR)
        # `kill -0 ''` and `[ -d /proc/ ]` both succeed, which reported a dead
        # run as alive.
        assert '[ -n "$P" ]' in command
        assert '[ "$P" != - ]' in command

    def test_reads_the_log_from_the_given_offset(self):
        assert "tail -c +4096" in build_poll_command(4096, remote_dir=REMOTE_DIR)

    def test_the_chunk_is_bounded(self):
        # A burst of output must not be pulled into memory in one go.
        assert "head -c 1024" in build_poll_command(
            1, remote_dir=REMOTE_DIR, max_chunk=1024)


class TestParsePoll:
    def test_running(self):
        progress = parse_poll_output(status(alive=1, size=10, log="hello"))
        assert progress.finished is False
        assert progress.alive is True
        assert progress.exit_code is None
        assert progress.output == "hello"
        assert progress.log_size == 10

    def test_finished_with_a_code(self):
        progress = parse_poll_output(status(exit_code="0", alive=0, size=5))
        assert progress.finished is True and progress.exit_code == 0

    def test_finished_with_a_failure_code(self):
        assert parse_poll_output(status(exit_code="137", alive=0)).exit_code == 137

    def test_gone_without_an_exit_code(self):
        progress = parse_poll_output(status(exit_code="-", alive=0))
        # Killed from outside, or the container restarted.
        assert progress.finished is False and progress.alive is False

    def test_a_non_numeric_exit_file_still_counts_as_finished(self):
        progress = parse_poll_output(status(exit_code="oops", alive=0))
        assert progress.finished is True and progress.exit_code is None

    def test_multiline_output_is_preserved(self):
        progress = parse_poll_output(status(log="a\nb\nc"))
        assert progress.output == "a\nb\nc"

    def test_garbage_is_refused_rather_than_guessed(self):
        with pytest.raises(DetachedRunError, match="unexpected status line"):
            parse_poll_output("bash: setsid: command not found\n")


# ── Launching ────────────────────────────────────────────────────────────────

class TestLaunch:
    def test_reads_the_pid_back(self):
        transport = FakeTransport([
            ExecResult(0, f"{STATUS_PREFIX}-launched", ""),
            ExecResult(0, "4242\n", ""),
        ])
        handle = build_run(transport).launch(JAVA)
        assert handle == RunHandle(pid=4242, remote_dir=REMOTE_DIR)

    def test_waits_for_the_pid_file_to_appear(self):
        # The launch returns before the inner shell has necessarily written it.
        transport = FakeTransport([
            ExecResult(0, "launched", ""),
            ExecResult(0, "", ""),
            ExecResult(0, "", ""),
            ExecResult(0, "77\n", ""),
        ])
        assert build_run(transport).launch(JAVA).pid == 77

    def test_a_run_that_never_starts_is_reported(self):
        transport = FakeTransport([ExecResult(0, "launched", "")])
        transport.default = ExecResult(0, "", "")
        with pytest.raises(DetachedRunError, match="did not record a PID"):
            build_run(transport).launch(JAVA)

    def test_a_failing_launch_surfaces_the_stderr(self):
        transport = FakeTransport([ExecResult(127, "", "setsid: not found")])
        with pytest.raises(DetachedRunError, match="setsid: not found"):
            build_run(transport).launch(JAVA)


# ── Following ────────────────────────────────────────────────────────────────

class TestWait:
    def _started(self, results) -> tuple[DetachedRun, FakeTransport]:
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", ""), *results])
        run = build_run(transport)
        run.launch(JAVA)
        return run, transport

    def test_completes_and_reports_the_exit_code(self):
        run, _ = self._started([
            ExecResult(0, status(alive=1, log="working"), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        outcome = run.wait(timeout_sec=1000)
        assert outcome.completed is True and outcome.exit_code == 0

    def test_output_is_streamed_as_it_arrives(self):
        run, _ = self._started([
            ExecResult(0, status(log="first"), ""),
            ExecResult(0, status(log="second"), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        seen: list[str] = []
        run.wait(timeout_sec=1000, on_output=seen.append)
        # Chunk by chunk, preserving the live log the old implementation had.
        assert seen == ["first", "second"]

    def test_the_offset_advances_so_nothing_is_read_twice(self):
        run, transport = self._started([
            ExecResult(0, status(log="12345"), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        run.wait(timeout_sec=1000)
        polls = [c for c in transport.commands if "tail -c" in c]
        assert "tail -c +1 " in polls[0]
        assert "tail -c +6 " in polls[1]

    def test_a_disappeared_process_is_not_a_success(self):
        """Still a failure — but only once it has been confirmed."""
        gone = ExecResult(0, status(exit_code="-", alive=0), "")
        run, transport = self._started([gone, gone, gone])

        outcome = run.wait(timeout_sec=1000)

        assert outcome.completed is False
        assert "disappeared" in outcome.message
        assert "3 looks" in outcome.message

    def test_an_exit_code_arriving_late_is_not_a_failure(self):
        """The race that cost a completed simulation.

        ``kill -0`` succeeds for a zombie, so the pid stops existing exactly
        when the supervising shell reaps it in ``wait $!`` — and the exit file
        is written just after. For a few milliseconds a *successful* finish is
        indistinguishable from a process that vanished, and the first look was
        taken as final. Measured on a real container: 4 of 25 one-second runs
        passed through that state.
        """
        run, _ = self._started([
            ExecResult(0, status(exit_code="-", alive=0), ""),   # reaped
            ExecResult(0, status(exit_code="0", alive=0), ""),   # exit code lands
        ])

        outcome = run.wait(timeout_sec=1000)

        assert outcome.completed is True
        assert outcome.exit_code == 0

    def test_a_late_non_zero_exit_code_is_still_reported(self):
        run, _ = self._started([
            ExecResult(0, status(exit_code="-", alive=0), ""),
            ExecResult(0, status(exit_code="137", alive=0), ""),
        ])

        outcome = run.wait(timeout_sec=1000)

        assert outcome.completed is True and outcome.exit_code == 137

    def test_output_written_just_before_the_end_is_not_lost(self):
        run, _ = self._started([
            ExecResult(0, status(exit_code="-", alive=0, log="12345"), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        seen: list[str] = []

        run.wait(timeout_sec=1000, on_output=seen.append)

        assert seen == ["12345"]

    def test_a_process_that_recovers_resets_the_count(self):
        # Only *consecutive* sightings count: anything else would accumulate
        # across a long run and fail it for no reason.
        run, _ = self._started([
            ExecResult(0, status(exit_code="-", alive=0), ""),
            ExecResult(0, status(alive=1), ""),
            ExecResult(0, status(exit_code="-", alive=0), ""),
            ExecResult(0, status(alive=1), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])

        outcome = run.wait(timeout_sec=1000)

        assert outcome.completed is True

    def test_one_look_is_allowed_for_a_caller_that_wants_it(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", ""),
            ExecResult(0, status(exit_code="-", alive=0), ""),
        ])
        run = build_run(transport, vanished_confirmations=1)
        run.launch(JAVA)

        outcome = run.wait(timeout_sec=1000)

        assert outcome.completed is False

    def test_waiting_before_launching_is_refused(self):
        with pytest.raises(DetachedRunError, match="nothing has been launched"):
            build_run(FakeTransport()).wait(timeout_sec=10)


class TestReconnection:
    def test_a_dropped_connection_is_retried_not_failed(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", ""),
            RuntimeError("channel closed"),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        run = build_run(transport)
        run.launch(JAVA)

        outcome = run.wait(timeout_sec=1000)

        # The run is detached, so losing the connection says nothing about it.
        assert outcome.completed is True
        assert outcome.lost_contact is False
        assert transport.closed >= 1 and transport.opened >= 1

    def test_several_drops_in_a_row_are_tolerated(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", ""),
            RuntimeError("drop 1"), RuntimeError("drop 2"), RuntimeError("drop 3"),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        run = build_run(transport)
        run.launch(JAVA)
        assert run.wait(timeout_sec=1000).completed is True

    def test_output_already_read_is_not_re_delivered_after_a_drop(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", ""),
            ExecResult(0, status(log="abc"), ""),
            RuntimeError("drop"),
            ExecResult(0, status(log="def"), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        run = build_run(transport)
        run.launch(JAVA)
        seen: list[str] = []
        run.wait(timeout_sec=1000, on_output=seen.append)
        assert seen == ["abc", "def"]


class TestTimeout:
    def test_the_timeout_stops_the_run(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", "")])
        transport.default = ExecResult(0, status(alive=1), "")
        run = build_run(transport)
        run.launch(JAVA)

        outcome = run.wait(timeout_sec=3)

        assert outcome.timed_out is True and outcome.completed is False
        # Abandoning it would leave a JVM holding the container's resources.
        assert any("kill -KILL" in c for c in transport.commands)

    def test_the_timeout_applies_while_disconnected(self):
        # Otherwise an unreachable host would hold a container indefinitely.
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", "")])
        run = build_run(transport)
        run.launch(JAVA)
        transport.fail_next = 100

        outcome = run.wait(timeout_sec=4)

        assert outcome.timed_out is True
        assert outcome.lost_contact is True

    def test_no_timeout_means_no_deadline(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "100\n", ""),
            ExecResult(0, status(alive=1), ""),
            ExecResult(0, status(exit_code="0", alive=0), ""),
        ])
        run = build_run(transport)
        run.launch(JAVA)
        # SIM_TIMEOUT_SEC=0 has always meant "do not time out".
        assert run.wait(timeout_sec=None).completed is True


class TestTerminate:
    def test_signals_the_group_then_insists(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "4242\n", "")])
        run = build_run(transport)
        run.launch(JAVA)

        assert run.terminate(grace_sec=1) is True

        command = transport.commands[-1]
        # The group, because the JVM may have children that would otherwise
        # keep holding the container's resources.
        assert "kill -TERM -4242" in command
        assert "kill -KILL -4242" in command

    def test_terminating_nothing_is_a_no_op(self):
        assert build_run(FakeTransport()).terminate() is False

    def test_a_failure_to_stop_is_reported_not_raised(self):
        transport = FakeTransport([
            ExecResult(0, "launched", ""), ExecResult(0, "1\n", "")])
        run = build_run(transport)
        run.launch(JAVA)
        transport.fail_next = 1
        assert run.terminate() is False
