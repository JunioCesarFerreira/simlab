"""The contract between the executor and the runner, with both functions real.

This file exists because of a defect the previous stage shipped. The runner
gained a quarantine for a run that could not be stopped and a re-queue for an
attempt that lost its container — and neither was ever reached, because
``run_cooja_simulation`` caught both control exceptions in its own blanket
``except Exception``, marked the simulation in error and returned normally.

The tests that were supposed to cover it replaced the executor with a lambda
that raised the exception straight at the runner. They passed without touching
the real path.

So here the executor is **not** substituted: ``SimulationRunner`` is wired to
the real ``run_cooja_simulation``, and only the edges — the transport, the
detached process and the database — are stand-ins. What is asserted is which
database calls the pair makes between them.
"""
import dataclasses
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.detached import RunOutcome
from lib.scheduler.runner import SimulationRunner

SIM_OID = ObjectId("507f1f77bcf86cd799430003")
RES_ID = ObjectId("507f1f77bcf86cd799430021")
GEN_OID = ObjectId("507f1f77bcf86cd799430031")
SIM = {"_id": SIM_OID, "status": "Waiting", "generation_id": GEN_OID,
       "experiment_id": ObjectId("507f1f77bcf86cd799430041")}
HOST = {"_id": ObjectId("507f1f77bcf86cd799430011"), "name": "host-a", "type": "local"}
RESOURCE = {"_id": RES_ID, "name": "cooja1"}
ATTEMPT = "attempt-A"


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
def wired(entry, monkeypatch, tmp_path):
    """A runner driving the real executor, with the edges replaced.

    What counts as an edge: the transport, the detached process, the database,
    and the pure file/metric helpers — each tested on its own elsewhere, and
    none of them able to run here because no Cooja produced a testlog. The
    executor's own control flow, which is what the defect lived in, is real.
    """
    monkeypatch.setattr(
        entry, "SET", dataclasses.replace(entry.SET, local_log_dir=str(tmp_path)))
    monkeypatch.setattr(
        entry.cooja_files, "convert_cooja_log_to_csv",
        lambda log_path, csv_path: Path(csv_path).write_text("node,time\n1,0\n"))
    monkeypatch.setattr(
        entry.statistics, "evaluate_config", lambda df, cfg, log: {"latency": 1.0})
    monkeypatch.setattr(
        entry.rpl_dodag, "analyze_dodag_full", lambda path: None)

    def build(*, outcome=None, mark_busy=True, held=True):
        mongo = MagicMock()
        mongo.simulation_repo.mark_running.return_value = True
        mongo.simulation_repo.mark_done.return_value = True
        mongo.simulation_repo.mark_error.return_value = True
        mongo.simulation_repo.requeue.return_value = True
        mongo.resource_repo.mark_busy.return_value = mark_busy
        mongo.resource_repo.is_held_by.return_value = held
        mongo.resource_repo.release.return_value = {
            "_id": RES_ID, "status": "available"}

        detached = MagicMock()
        detached.launch.return_value = MagicMock(pid=4242)
        detached.wait.return_value = outcome or RunOutcome(
            completed=True, exit_code=0)
        detached.terminate.return_value = False
        monkeypatch.setattr(entry, "DetachedRun", lambda *a, **k: detached)

        transport = MagicMock()
        transport.__enter__ = lambda s: s
        transport.__exit__ = lambda s, *a: None
        transport.plan.describe.return_value = "host-a/cooja1"
        transport.plan.remote_dir = "/opt/cooja"
        transport.get_file.side_effect = (
            lambda remote, local: Path(local).write_text("testlog\n"))
        monkeypatch.setattr(
            "lib.scheduler.runner.build_transport", lambda *a, **k: transport)

        runner = SimulationRunner(
            mongo,
            execute=entry.run_cooja_simulation,
            prepare=lambda sim, repo: (True, [], []),
        )
        return runner, mongo, transport, detached

    return build


class TestTerminationUnconfirmed:
    """A run that could not be stopped must take its worker out of the pool."""

    def _unstoppable(self, wired):
        return wired(outcome=RunOutcome(
            completed=False, timed_out=True, stopped=False,
            message="exceeded SIM_TIMEOUT_SEC and could not be stopped"))

    def test_the_worker_is_disabled(self, wired):
        runner, mongo, _, _ = self._unstoppable(wired)

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.set_enabled.assert_called_once_with(RES_ID, False)

    def test_the_simulation_is_marked_once(self, wired):
        runner, mongo, _, _ = self._unstoppable(wired)

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        # Once, by the runner. Both layers used to mark it.
        assert mongo.simulation_repo.mark_error.call_count == 1

    def test_the_run_artifacts_are_left_in_place(self, wired):
        """`run.pgid` is the only handle on a process that may be alive.

        The executor used to clear them in a `finally`, which ran on exactly
        this path.
        """
        runner, _, transport, _ = self._unstoppable(wired)

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        cleared = [c.args[0] for c in transport.cleanup.call_args_list if c.args]
        assert not any("run.pgid" in names for names in cleared)


class TestLostOwnership:
    """An attempt that lost its container returns the work to the queue."""

    def test_a_refused_promotion_re_queues(self, wired):
        runner, mongo, _, _ = wired(mark_busy=False)

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_refused_publication_re_queues(self, wired):
        runner, mongo, _, _ = wired()
        mongo.simulation_repo.mark_done.return_value = False

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_container_already_handed_on_is_not_written_to(self, wired):
        """Checked before the upload, so nothing lands in someone else's run."""
        runner, mongo, transport, _ = wired(held=False)

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        transport.put_files.assert_not_called()
        mongo.simulation_repo.requeue.assert_called_once()


class TestOrdinaryOutcomes:
    def test_a_completed_run_is_published_by_the_runner(self, wired):
        runner, mongo, _, _ = wired()

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_done.assert_called_once()
        assert mongo.simulation_repo.mark_done.call_args.kwargs[
            "attempt_id"] == ATTEMPT
        mongo.generation_repo.close_if_finished.assert_called_once_with(GEN_OID)
        mongo.resource_repo.release.assert_called_once()

    def test_the_executor_writes_no_simulation_state(self, wired):
        """One owner. The executor reports; the runner decides."""
        runner, mongo, _, _ = wired()

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        # mark_running and mark_done, both from the runner — never twice.
        assert mongo.simulation_repo.mark_running.call_count == 1
        assert mongo.simulation_repo.mark_done.call_count == 1

    def test_an_incomplete_run_is_a_plain_failure(self, wired):
        runner, mongo, _, _ = wired(outcome=RunOutcome(
            completed=False, message="the simulation did not complete"))

        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_error.assert_called_once()
        mongo.resource_repo.set_enabled.assert_not_called()
        mongo.simulation_repo.requeue.assert_not_called()


class TestStartupRecoveryOrder:
    """What the clock may decide, and what it may not.

    `recover_stuck_simulations` used to run *before* the reconciliation and
    purely by elapsed time. The reconciliation then found a simulation already
    terminal and freed its container without probing — so a JVM that had
    survived the master-node, which is the whole point of detached execution,
    could keep running on a container handed to the next simulation.
    """

    def _mongo(self, *, holder):
        mongo = MagicMock()
        mongo.simulation_repo.find_running_before.return_value = [
            {"_id": SIM_OID, "generation_id": GEN_OID, "start_time": None}]
        mongo.resource_repo.find_by_simulation.return_value = holder
        return mongo

    def test_a_simulation_a_container_still_holds_is_left_alone(self, entry):
        mongo = self._mongo(holder={"_id": RES_ID, "name": "cooja1"})

        entry.recover_abandoned_simulations(mongo)

        # The reconciliation owns it: it has a container to ask.
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_simulation_nothing_holds_is_failed(self, entry):
        mongo = self._mongo(holder=None)

        entry.recover_abandoned_simulations(mongo)

        mongo.simulation_repo.mark_error.assert_called_once()
        assert "no container holding it" in \
            mongo.simulation_repo.mark_error.call_args.args[1]

    def test_an_unanswerable_check_leaves_the_simulation_alone(self, entry):
        # Cannot tell whether anything holds it: failing it would be the same
        # guess that caused the defect.
        mongo = self._mongo(holder=None)
        mongo.resource_repo.find_by_simulation.side_effect = RuntimeError("mongo")

        entry.recover_abandoned_simulations(mongo)

        mongo.simulation_repo.mark_error.assert_not_called()

    def test_the_timeout_can_disable_it(self, entry, monkeypatch):
        monkeypatch.setattr(
            entry, "SET", dataclasses.replace(entry.SET, sim_timeout_sec=0))
        mongo = self._mongo(holder=None)

        entry.recover_abandoned_simulations(mongo)

        mongo.simulation_repo.find_running_before.assert_not_called()
