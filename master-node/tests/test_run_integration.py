"""Executor, runner and ``DetachedRun`` all real, with only the container faked.

Why this file exists
--------------------
``test_executor_contract.py`` already drives the real executor and the real
runner, and it found what it was built for. But it replaces ``DetachedRun``
with a ``MagicMock``, and a ``MagicMock`` answers anything: ``handle()`` is a
valid call on one and a ``TypeError`` on the real object, whose ``handle`` is a
property. So the retry over a fresh session — the whole point of separating a
collection failure from a failed run — was dead code that passed its tests
(R23).

The stand-in here is a *container*, not a Python object: it answers the shells
``lib.detached`` composes and refuses anything it does not recognise. Every
decision between the three layers is then real, and the only thing simulated
is the thing that genuinely cannot be in a unit test — a JVM on the other end
of an SSH connection.
"""
import dataclasses
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.detached import PGID_FILE, PID_FILE, STATUS_PREFIX
from lib.scheduler.runner import SimulationRunner
from lib.transport.base import ExecResult, TransportError
from pylib.timeutil import utc_now

SIM_OID = ObjectId("507f1f77bcf86cd799430003")
RES_ID = ObjectId("507f1f77bcf86cd799430021")
GEN_OID = ObjectId("507f1f77bcf86cd799430031")
HOST = {"_id": ObjectId("507f1f77bcf86cd799430011"), "name": "host-a", "type": "local"}
RESOURCE = {"_id": RES_ID, "name": "cooja1",
            "remote_dir": "/opt/contiki-ng/tools/cooja"}
ATTEMPT = "attempt-A"
REMOTE_DIR = "/opt/contiki-ng/tools/cooja"


def sim(**over) -> dict:
    base = {
        "_id": SIM_OID, "status": "Waiting", "generation_id": GEN_OID,
        "experiment_id": ObjectId("507f1f77bcf86cd799430041"),
    }
    base.update(over)
    return base


class UnexpectedCommand(AssertionError):
    """The container was asked something this fake does not model.

    Raised rather than answered. A fake that tolerates anything cannot tell
    you that the code under test is talking nonsense, which is the failure
    mode this whole file is a reaction to.
    """


class ScriptedContainer:
    """Answers the real remote shells for one run.

    State, not a script: ``alive``, ``exit_code`` and ``testlog`` describe the
    container at the moment it is asked, so the same object serves a run that
    finishes, one that is still going and one that vanished without recording
    a status.
    """

    def __init__(
        self,
        *,
        pid: int = 4242,
        pid_readable: bool = True,
        alive: bool = False,
        exit_code=0,
        testlog: str = "Starting Cooja\nTest ended\n",
        group_alive: bool = False,
        fetch_failures: int = 0,
        live_in_dir: bool = False,
        fail_on: Optional[str] = None,
        fail_open_at: tuple[int, ...] = (),
        launch_ok: bool = True,
    ):
        self.pid = pid
        self.pid_readable = pid_readable
        self.alive = alive
        self.exit_code = exit_code
        self.testlog = testlog
        self.group_alive = group_alive
        self.fetch_failures = fetch_failures
        self.live_in_dir = live_in_dir
        #: A substring; every command containing it loses contact. This is how
        #: a connection that drops *after* sshd accepted something is
        #: represented — the one case `LaunchUnconfirmed` exists for.
        self.fail_on = fail_on
        #: 1-based indices of `open()` calls that fail. The retry opens a
        #: session of its own, so `(2,)` is "the second connection never
        #: came up" — the case where the preservation copy cannot be made.
        self.fail_open_at = set(fail_open_at)
        #: The remote shell answering "no". Distinct from losing contact:
        #: nothing started, so this is an ordinary start-up failure.
        self.launch_ok = launch_ok

        self.commands: list[str] = []
        self.fetches = 0
        self.launches = 0
        self.terminations = 0
        self.cleanups: list[tuple] = []
        self.uploads: list[tuple] = []
        self.preserved: list[str] = []
        self.opens = 0
        self.closes = 0
        self.plan = MagicMock()
        self.plan.remote_dir = REMOTE_DIR
        self.plan.describe.return_value = "host-a/cooja1"

    # ── the transport surface the code actually uses ──────────────────────

    def exec_command(self, command: str, timeout=None) -> ExecResult:
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            raise TransportError("contact lost")
        status = (
            f"{STATUS_PREFIX} "
            f"{'-' if self.exit_code is None else self.exit_code} "
            f"{1 if self.alive else 0} {len(self.testlog)}"
        )

        if f"{STATUS_PREFIX}-launched" in command:
            self.launches += 1
            if not self.launch_ok:
                return ExecResult(127, "", "setsid: not found")
            return ExecResult(0, f"{STATUS_PREFIX}-launched", "")
        if f"{STATUS_PREFIX}-live" in command:
            # The pre-upload guard: is anything running in this directory?
            mark = "-live" if self.live_in_dir else "-clear"
            return ExecResult(0, f"{STATUS_PREFIX}{mark}", "")
        if "kill -TERM" in command:
            self.terminations += 1
            mark = "-alive" if self.group_alive else "-gone"
            return ExecResult(0, f"{STATUS_PREFIX}{mark}", "")
        # Order matters, and getting it wrong was instructive: the poll command
        # also contains `cat run.pid`, so matching that first answered a poll
        # with a bare pid. The parse failed, `wait` read it as a lost
        # connection, and the reconnect loop ran until the simulation timeout
        # — a hanging suite instead of a failing test. The specific shells
        # come before the general ones.
        if "tail -c +" in command:
            return ExecResult(0, f"{status}\n{self.testlog}", "")
        if "wc -c < COOJA.testlog" in command:
            return ExecResult(0, status, "")
        if f"cat {PID_FILE}" in command or f"cat {PGID_FILE}" in command:
            return ExecResult(
                0, f"{self.pid}\n" if self.pid_readable else "", "")
        if "cp -f COOJA.testlog" in command:
            if not self.testlog:
                return ExecResult(1, "", "")
            self.preserved.append(command)
            return ExecResult(0, "kept", "")
        if "rm -rf --" in command:
            return ExecResult(0, "", "")
        raise UnexpectedCommand(command)

    def get_file(self, remote: str, local: str) -> None:
        self.fetches += 1
        if self.fetches <= self.fetch_failures:
            raise OSError("session dropped")
        Path(local).write_text(self.testlog)

    def put_files(self, local, remote) -> None:
        self.uploads.append((tuple(local), tuple(remote)))

    def cleanup(self, names=None) -> bool:
        self.cleanups.append(tuple(names or ()))
        return True

    def open(self) -> None:
        self.opens += 1
        if self.opens in self.fail_open_at:
            raise TransportError("could not reach the container")

    def close(self) -> None:
        self.closes += 1

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture(scope="module")
def entry():
    """The entrypoint module, loaded by path — `master-node.py` is not a name."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "master_node_entry", root / "master-node.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["master_node_entry"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wire(entry, monkeypatch, tmp_path):
    """A runner → real executor → real ``DetachedRun`` → scripted container."""
    # A short timeout on purpose. A fake that answers a poll with something
    # unparseable sends `wait` into its reconnect loop, and that loop only
    # ends at the timeout — so the default hour would turn a mistake in this
    # file into a hanging suite rather than a failing test.
    monkeypatch.setattr(
        entry, "SET", dataclasses.replace(
            entry.SET, local_log_dir=str(tmp_path), run_poll_interval_sec=0,
            sim_timeout_sec=5))
    monkeypatch.setattr(
        entry.cooja_files, "convert_cooja_log_to_csv",
        lambda log_path, csv_path: Path(csv_path).write_text("node,time\n1,0\n"))
    monkeypatch.setattr(
        entry.statistics, "evaluate_config", lambda df, cfg, log: {"latency": 1.0})
    monkeypatch.setattr(entry.rpl_dodag, "analyze_dodag_full", lambda path: None)

    def build(container: ScriptedContainer, *, heartbeat=3600.0,
              budget=3600.0, **repo):
        mongo = MagicMock()
        mongo.simulation_repo.mark_running.return_value = True
        mongo.simulation_repo.mark_done.return_value = repo.pop("mark_done", True)
        mongo.simulation_repo.mark_error.return_value = True
        mongo.simulation_repo.requeue.return_value = True
        mongo.simulation_repo.update.return_value = True
        mongo.resource_repo.mark_busy.return_value = repo.pop("mark_busy", True)
        mongo.resource_repo.is_held_by.return_value = True
        mongo.resource_repo.touch_reservation.return_value = repo.pop(
            "lease", True)
        mongo.resource_repo.quarantine.return_value = True
        mongo.resource_repo.release.return_value = {
            "_id": RES_ID, "status": "available"}
        mongo.fs_handler.upload_file.side_effect = (
            lambda path, name: f"blob-{name}")
        assert not repo, f"unused: {sorted(repo)}"

        monkeypatch.setattr(
            "lib.scheduler.runner.build_transport",
            MagicMock(return_value=container))
        runner = SimulationRunner(
            mongo,
            execute=entry.run_cooja_simulation,
            prepare=lambda s, repo_: (True, ["tmp/a.csc"], ["simulation.csc"]),
            # Effectively off unless a test asks for it: a heartbeat firing
            # during an unrelated test would only add noise.
            lease_heartbeat_sec=heartbeat,
            prepare_budget_sec=budget,
        )
        return runner, mongo

    return build


# ── R23 ─────────────────────────────────────────────────────────────────────

class TestCollectionRetryWithTheRealObject:
    """A dropped download must open a second session and collect again.

    Reproduces R23. ``DetachedRun.handle`` is a property, and the executor
    called it: building ``CollectionFailed`` raised ``TypeError`` before the
    exception existed, the runner saw a generic error, and the retry it had
    been given was never reached. With a ``MagicMock`` in place of
    ``DetachedRun`` the same code passed.
    """

    def test_a_dropped_download_is_collected_on_a_second_session(self, wire):
        container = ScriptedContainer(fetch_failures=1)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.fetches == 2
        assert container.opens == 2, "the retry must not reuse the dead session"
        mongo.simulation_repo.mark_done.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_the_pid_survives_into_the_failure(self, wire):
        """Without the pid there is nothing to attach to, and no retry.

        The property call failed *while building* the exception, so the pid it
        was meant to carry never got there.
        """
        container = ScriptedContainer(fetch_failures=1)
        runner, _ = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        # The second collection attaches to the recorded pid and inspects it,
        # rather than launching anything again.
        assert container.launches == 1
        assert any("wc -c < COOJA.testlog" in c for c in container.commands)

    def test_a_second_failure_preserves_the_result_remotely(self, wire):
        """R18: exhaustion must not mean the bytes are lost.

        Leaving them under ``COOJA.testlog`` is not preservation — that name
        is exactly what the next run on this worker deletes. They are copied
        to a name the cleanup does not know, and where they are is written
        down.
        """
        container = ScriptedContainer(fetch_failures=5)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.preserved, "the testlog was not copied aside"
        assert ATTEMPT in container.preserved[0]
        recorded = [
            c.args[1] for c in mongo.simulation_repo.update.call_args_list
            if "unfetched_result" in c.args[1]
        ]
        assert recorded, "nothing recorded where the result is"
        assert recorded[0]["unfetched_result"]["remote_pid"] == 4242
        assert recorded[0]["unfetched_result"]["remote_file"] is not None


# ── R03 ─────────────────────────────────────────────────────────────────────

class TestResumeAfterALongStop:
    """The master-node being down is not the simulation running."""

    def test_a_run_that_finished_during_the_stop_is_collected(self, wire):
        """R03: the deadline was applied before anything was asked.

        A simulation that finished minutes in, with the master-node away for
        hours, was failed by the clock — and the result sitting on the
        container was never looked for.
        """
        container = ScriptedContainer(alive=False, exit_code=0)
        runner, mongo = wire(container)
        old = sim(start_time=utc_now().replace(year=2020))

        runner.resume(old, HOST, RESOURCE, ATTEMPT, 4242)

        mongo.simulation_repo.mark_done.assert_called_once()
        assert container.terminations == 0, "nothing should have been killed"
        assert container.fetches == 1

    def test_a_run_still_going_past_its_deadline_is_stopped(self, wire):
        """The timeout still applies — to a run that is actually running."""
        container = ScriptedContainer(alive=True, exit_code=None)
        runner, mongo = wire(container)
        old = sim(start_time=utc_now().replace(year=2020))

        runner.resume(old, HOST, RESOURCE, ATTEMPT, 4242)

        assert container.terminations == 1
        mongo.simulation_repo.mark_done.assert_not_called()
        mongo.simulation_repo.mark_error.assert_called_once()

    def test_a_run_that_cannot_be_stopped_quarantines_the_worker(self, wire):
        container = ScriptedContainer(
            alive=True, exit_code=None, group_alive=True)
        runner, mongo = wire(container)
        old = sim(start_time=utc_now().replace(year=2020))

        runner.resume(old, HOST, RESOURCE, ATTEMPT, 4242)

        mongo.resource_repo.quarantine.assert_called_once()
        mongo.resource_repo.release.assert_not_called()


class TestTestlogWithoutAnExitCode:
    """The stated policy: keep the bytes, do not declare a result."""

    def test_the_artifacts_are_kept_and_the_result_is_not_published(self, wire):
        container = ScriptedContainer(
            alive=False, exit_code=None, testlog="Starting Cooja\n")
        runner, mongo = wire(container)

        runner.resume(sim(), HOST, RESOURCE, ATTEMPT, 4242)

        mongo.simulation_repo.mark_done.assert_not_called()
        mongo.simulation_repo.mark_error.assert_called_once()
        kwargs = mongo.simulation_repo.mark_error.call_args.kwargs
        # Stored, referenced, and deliberately *not* in `network_metrics`:
        # the objective aggregation reads that field, and an unconfirmed run
        # must not enter a mean without the evaluation above it knowing.
        assert kwargs["log_id"] == "blob-sim_result.log"
        assert kwargs["partial_metrics"] == {"latency": 1.0}

    def test_nothing_is_kept_when_there_is_nothing_there(self, wire):
        container = ScriptedContainer(alive=False, exit_code=None, testlog="")
        runner, mongo = wire(container)

        runner.resume(sim(), HOST, RESOURCE, ATTEMPT, 4242)

        mongo.simulation_repo.mark_error.assert_called_once()
        assert mongo.simulation_repo.mark_error.call_args.kwargs["log_id"] is None
        assert container.fetches == 0


# ── Garantia 1, on the real path ────────────────────────────────────────────

class TestExclusivityOnTheRealPath:
    def test_a_live_run_in_the_directory_stops_the_upload(self, wire):
        """Two attempts must never share a run directory.

        The first thing a run does is ``rm -rf`` the previous one's
        artifacts, and the reservation is not evidence that the previous one
        is over.
        """
        container = ScriptedContainer(live_in_dir=True)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.uploads == []
        assert container.cleanups == [], "a live directory must not be cleaned"
        mongo.resource_repo.quarantine.assert_called_once()
        mongo.resource_repo.release.assert_not_called()
        # Nothing of ours was sent, so this simulation has nothing to lose:
        # the worker is unusable, the work is not.
        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_refused_lease_stops_before_any_remote_effect(self, wire):
        """The reservation is gone: nothing may be sent, nothing published."""
        container = ScriptedContainer()
        runner, mongo = wire(container, lease=False)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.uploads == []
        assert container.launches == 0
        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_an_unreadable_pid_with_a_live_group_quarantines(self, wire):
        """R05.4: "the launch failed" is not "nothing was launched"."""
        container = ScriptedContainer(pid_readable=False, group_alive=True)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.quarantine.assert_called_once()
        mongo.resource_repo.release.assert_not_called()

    def test_an_unreadable_pid_with_a_dead_group_is_a_plain_failure(self, wire):
        container = ScriptedContainer(pid_readable=False, group_alive=False)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.quarantine.assert_not_called()
        mongo.resource_repo.release.assert_called_once()
        mongo.simulation_repo.mark_error.assert_called_once()

    def test_losing_ownership_with_a_surviving_run_quarantines(self, wire):
        """R05.2: ownership lost *and* the run would not stop.

        The pool already believes this worker is free, so the only thing that
        keeps a second simulation off a container with a live JVM is taking
        it out administratively. ``terminate()``'s answer used to be
        discarded.
        """
        container = ScriptedContainer(group_alive=True)
        runner, mongo = wire(container, mark_busy=False)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.requeue.assert_called_once()
        mongo.resource_repo.quarantine.assert_called_once()
        mongo.resource_repo.release.assert_not_called()

    def test_losing_ownership_with_a_stopped_run_only_re_queues(self, wire):
        container = ScriptedContainer(group_alive=False)
        runner, mongo = wire(container, mark_busy=False)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.requeue.assert_called_once()
        mongo.resource_repo.quarantine.assert_not_called()


# ── R24 ─────────────────────────────────────────────────────────────────────

class TestGenerationClosingIsSeparate:
    def test_a_published_result_survives_a_failed_generation_close(self, wire):
        """R24: administrative bookkeeping must not undo a result.

        ``_publish`` wrote ``mark_done`` and then closed the generation. A
        failure of that second write propagated out and the generic handler
        marked the finished simulation in error — an experiment's outcome
        changed retroactively by a failure that had nothing to do with the
        run.
        """
        container = ScriptedContainer()
        runner, mongo = wire(container)
        mongo.generation_repo.close_if_finished.side_effect = RuntimeError(
            "generation write failed")

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_done.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()
        mongo.resource_repo.release.assert_called_once()

    def test_a_failed_publish_is_repeated_without_a_second_upload(self, wire):
        """A Mongo failure after the upload is a publish failure.

        Repeating the *collection* would re-download and re-upload the same
        bytes, leaving two blobs for one simulation. Only the write repeats.
        """
        container = ScriptedContainer()
        runner, mongo = wire(container)
        calls = {"n": 0}

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise RuntimeError("mongo gone")
            return True

        mongo.simulation_repo.mark_done.side_effect = flaky

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert calls["n"] == 4, "three failures, then the repeat"
        assert container.fetches == 1, "nothing was downloaded twice"
        assert mongo.fs_handler.upload_file.call_count == 2, "log + csv, once"


# ── R05: every lost confirmation is a possibly-active run ─────────────────

class TestLostConfirmationAroundTheLaunch:
    """"Error while launching" is not "nothing was launched".

    The launch is two round trips. Only one failure mode was classified as
    unconfirmed — the shell answering without writing a pid — so a connection
    that dropped around either trip fell through to the generic transport
    handler and **released the worker**. The command may well have been
    accepted; a JVM may well be running with nobody holding its pid.
    """

    def test_contact_lost_sending_the_launch_holds_the_worker(self, wire):
        container = ScriptedContainer(
            fail_on=f"{STATUS_PREFIX}-launched", group_alive=True)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.quarantine.assert_called_once()
        mongo.resource_repo.release.assert_not_called()

    def test_contact_lost_reading_the_pid_holds_it_too(self, wire):
        container = ScriptedContainer(
            fail_on=f"cat {PID_FILE}", group_alive=True)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.quarantine.assert_called_once()
        mongo.resource_repo.release.assert_not_called()

    def test_a_confirmed_stop_after_lost_contact_is_a_plain_failure(self, wire):
        """The distinction the whole mechanism rests on.

        Contact was lost, so the launch is unconfirmed — but the group was
        then found gone, which *is* evidence that nothing is running. Holding
        the worker on every network blip would be a bad trade; holding it
        when nothing could be confirmed is the point.
        """
        container = ScriptedContainer(
            fail_on=f"{STATUS_PREFIX}-launched", group_alive=False)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.quarantine.assert_not_called()
        mongo.resource_repo.release.assert_called_once()

    def test_the_hold_records_what_to_ask_about(self, wire):
        """A later pass has to be able to probe rather than infer, and the
        resource registration is mutable — so the directory is recorded."""
        container = ScriptedContainer(pid_readable=False, group_alive=True)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        held = mongo.resource_repo.quarantine.call_args.kwargs["held_for"]
        assert held["kind"] == "termination_unknown"
        assert held["attempt_id"] == ATTEMPT
        assert held["remote_dir"] == RESOURCE["remote_dir"]

    def test_a_shell_that_answers_no_is_still_a_plain_failure(self, wire):
        """The remote shell refusing is different: nothing started."""
        container = ScriptedContainer(launch_ok=False)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.quarantine.assert_not_called()
        mongo.resource_repo.release.assert_called_once()

    def test_a_failed_quarantine_write_still_records_the_hold(self, wire):
        """The protection has to outlive this process.

        Being disabled kept the dispatcher off the worker but not the
        reconciliation, which found a terminal simulation and freed it with no
        probe. The hold is in the document, so that pass refuses.
        """
        container = ScriptedContainer(pid_readable=False, group_alive=True)
        runner, mongo = wire(container)
        mongo.resource_repo.quarantine.side_effect = RuntimeError("mongo gone")

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.hold.assert_called_once()
        assert mongo.resource_repo.hold.call_args.args[2]["kind"] == (
            "termination_unknown")
        mongo.resource_repo.release.assert_not_called()


# ── R18: nothing is released until the result is accounted for ───────────

class TestThePreservationGatesTheRelease:
    def test_a_second_session_that_never_opens_holds_the_worker(self, wire):
        """The copy is made *inside* the second session. When it does not
        open there is no copy, and the original testlog is still under the
        name the next run deletes — so the worker may not be handed on.
        """
        container = ScriptedContainer(fetch_failures=1, fail_open_at=(2,))
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.preserved == []
        mongo.resource_repo.release.assert_not_called()
        held = mongo.resource_repo.quarantine.call_args.kwargs["held_for"]
        assert held["kind"] == "collection_pending"

    def test_a_preserved_copy_lets_the_worker_go(self, wire):
        container = ScriptedContainer(fetch_failures=5)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.preserved, "the testlog was not copied aside"
        mongo.resource_repo.release.assert_called_once()

    def test_an_unrecorded_unconfirmed_result_is_not_cleaned_up(self, wire):
        """The blobs exist and the simulation does not reference them.

        This path was handled as a successful return: the caller went on to
        clear the remote artifacts and release the worker, so the blobs were
        orphaned *and* the evidence on the container was deleted — for a run
        that had produced a result. No crash required.
        """
        container = ScriptedContainer(
            alive=False, exit_code=None, testlog="Starting Cooja\n")
        runner, mongo = wire(container)
        mongo.simulation_repo.mark_error.side_effect = RuntimeError("mongo gone")

        runner.resume(sim(), HOST, RESOURCE, ATTEMPT, 4242)

        assert container.cleanups == [], "the remote evidence was cleared"
        mongo.resource_repo.release.assert_not_called()
        held = mongo.resource_repo.quarantine.call_args.kwargs["held_for"]
        assert held["kind"] == "collection_pending"

    def test_the_recorded_location_includes_the_directory(self, wire):
        container = ScriptedContainer(fetch_failures=5)
        runner, mongo = wire(container)

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        recorded = [
            c.args[1]["unfetched_result"]
            for c in mongo.simulation_repo.update.call_args_list
            if "unfetched_result" in c.args[1]
        ]
        assert recorded and recorded[0]["remote_dir"] == RESOURCE["remote_dir"]


# ── R06: the reservation is held for the whole of the slow part ──────────

class TestTheLeaseIsHeldThroughout:
    def test_it_beats_while_the_upload_runs(self, wire):
        """Renewing before and after proves ownership at two instants; the
        sweep acts on what is true in between."""
        container = ScriptedContainer()
        runner, mongo = wire(container, heartbeat=0.01)
        beats: list[int] = []
        mongo.resource_repo.touch_reservation.side_effect = (
            lambda *a: beats.append(1) or True)

        def slow_upload(local, remote):
            time.sleep(0.08)

        container.put_files = slow_upload

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        # Two direct renewals plus at least one heartbeat during the upload.
        assert len(beats) > 2, beats

    def test_a_refusal_mid_upload_stops_before_the_launch(self, wire):
        container = ScriptedContainer()
        runner, mongo = wire(container, heartbeat=0.01)
        calls = {"n": 0}

        def renew(*_a):
            calls["n"] += 1
            return calls["n"] <= 1          # the first, then gone

        mongo.resource_repo.touch_reservation.side_effect = renew

        def slow_upload(local, remote):
            time.sleep(0.08)

        container.put_files = slow_upload

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.launches == 0
        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()


class TestThePreparationBudget:
    """A heartbeat that renews for ever claims more than it delivers.

    "A slow upload does not lose its worker" and "the worker is never lost"
    are different statements. A transfer that stopped making progress would
    hold its container until the simulation timeout, with the reservation
    looking healthy the whole time.
    """

    @pytest.mark.parametrize("ttl", [0.3, 4.0, 600.0])
    def test_dispatcher_heartbeat_is_shorter_than_the_configured_ttl(self, entry, monkeypatch, ttl):
        configured = dataclasses.replace(entry.SET, reservation_ttl_sec=ttl)
        monkeypatch.setattr(entry, "SET", configured)
        constructor = MagicMock()
        monkeypatch.setattr(entry, "SimulationRunner", constructor)
        monkeypatch.setattr(entry, "Dispatcher", MagicMock())
        monkeypatch.setattr(entry, "Thread", MagicMock())

        entry.build_dispatcher(MagicMock(), MagicMock(), MagicMock())

        assert 0 < constructor.call_args.kwargs["lease_heartbeat_sec"] < ttl

    def test_an_expired_upload_is_interrupted_before_the_worker_is_released(self, wire):
        container = ScriptedContainer()
        runner, mongo = wire(container, heartbeat=0.01, budget=0.05)
        interrupted = threading.Event()
        order = []

        def close():
            order.append("interrupt")
            interrupted.set()

        def upload(local, remote):
            assert interrupted.wait(2), "budget did not close the transport"
            # A transport may take time to unwind after cancellation. The
            # reservation still needs to be protected during this interval.
            before = mongo.resource_repo.touch_reservation.call_count
            time.sleep(0.08)
            assert mongo.resource_repo.touch_reservation.call_count > before
            order.append("upload_stopped")
            raise TransportError("channel closed")

        container.close = close
        container.put_files = upload
        mongo.resource_repo.release.side_effect = lambda *a, **kw: order.append("release")

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert order.index("upload_stopped") < order.index("release")
        assert container.launches == 0
        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_preparation_that_eventually_returns_gives_the_worker_up(self, wire):
        container = ScriptedContainer()
        runner, mongo = wire(container, heartbeat=0.01, budget=0.05)

        def never_ending_upload(local, remote):
            time.sleep(0.4)

        container.put_files = never_ending_upload

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.launches == 0
        mongo.simulation_repo.requeue.assert_called_once()
        # Released, not held: nothing was launched, so there is nothing
        # unaccounted for on the container.
        mongo.resource_repo.release.assert_called_once()

    @pytest.mark.parametrize("operation", ["is_held_by", "touch_reservation"])
    def test_unknown_ownership_does_not_allow_cleanup_or_upload(self, wire, operation):
        container = ScriptedContainer()
        runner, mongo = wire(container)
        getattr(mongo.resource_repo, operation).side_effect = RuntimeError("database offline")

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.cleanups == []
        assert container.uploads == []
        assert container.launches == 0
        mongo.simulation_repo.requeue.assert_called_once()

    def test_a_preparation_inside_the_budget_is_untouched(self, wire):
        container = ScriptedContainer()
        runner, mongo = wire(container, heartbeat=0.01, budget=5.0)

        def slow_upload(local, remote):
            time.sleep(0.08)

        container.put_files = slow_upload

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.launches == 1
        mongo.simulation_repo.mark_done.assert_called_once()

    def test_the_budget_can_be_switched_off(self, wire):
        container = ScriptedContainer()
        runner, mongo = wire(container, heartbeat=0.01, budget=0)

        def slow_upload(local, remote):
            time.sleep(0.08)

        container.put_files = slow_upload

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        assert container.launches == 1


class TestTheOccupancyIsMeasuredOnTheRealPath:
    """What a speed-aware policy is given, measured where it is produced.

    The number is how long the *container* was held — connection, upload,
    run, collection — because that is what decides throughput. Measuring the
    simulation's own ``start_time``/``end_time`` would leave out the two
    transfers, which is most of what makes a remote worker slower.

    Advisory by construction: it only chooses between equally loaded hosts,
    so every failure of it has to be survivable.
    """

    def test_a_successful_run_records_how_long_the_container_was_held(
            self, wire):
        runner, mongo = wire(ScriptedContainer())

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.record_duration.assert_called_once()
        resource_id, seconds = mongo.resource_repo.record_duration.call_args.args
        assert resource_id == RES_ID
        assert seconds > 0

    def test_a_failed_run_records_nothing(self, wire):
        """A run that aborted in five seconds would otherwise make the worst
        worker look like the best."""
        # The launch is never confirmed, so nothing is published.
        runner, mongo = wire(ScriptedContainer(
            fail_on=f"cat {PID_FILE}", group_alive=False))

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_done.assert_not_called()
        mongo.resource_repo.record_duration.assert_not_called()

    def test_a_resumed_run_records_nothing(self, wire):
        """Adopting a JVM that was already running measures the remainder of
        somebody else's run; recording it would make a container look fast
        for finishing work it did not start."""
        runner, mongo = wire(ScriptedContainer(alive=False, exit_code=0))

        runner.resume(sim(), HOST, RESOURCE, ATTEMPT, pid=4242)

        mongo.simulation_repo.mark_done.assert_called_once()
        mongo.resource_repo.record_duration.assert_not_called()

    def test_a_failure_to_record_it_does_not_fail_the_simulation(self, wire):
        """The trade stated as a test: losing the statistic costs a slightly
        worse placement, and must never cost a result."""
        runner, mongo = wire(ScriptedContainer())
        mongo.resource_repo.record_duration.side_effect = RuntimeError(
            "mongo is down")

        runner.run(sim(), HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_done.assert_called_once()
        mongo.resource_repo.release.assert_called()
