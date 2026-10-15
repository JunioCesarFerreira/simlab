"""Tests for executing one placed simulation.

The contract with the dispatcher is the subject here: a container is returned
on every exit path, exactly once, guarded by the attempt that holds it — and
capacity is signalled afterwards so a held job wakes up.

Getting this wrong is how a pool silently shrinks: each failure that skips the
release costs one container until the process restarts.
"""
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from lib.scheduler.runner import (
    LostOwnership,
    SimulationRunner,
    TerminationUnconfirmed,
)
from lib.transport import TransportError, TransportUnreachable

SIM_OID = ObjectId("507f1f77bcf86cd799430003")
GEN_OID = ObjectId("507f1f77bcf86cd799430004")
RES_ID = ObjectId("507f1f77bcf86cd799430021")
ATTEMPT = "8f14e45fceea167a5a36dedd4bea2543"

SIM = {"_id": SIM_OID, "status": "Waiting", "generation_id": GEN_OID}
HOST = {"_id": ObjectId("507f1f77bcf86cd799430011"), "name": "host-a", "type": "local"}
RESOURCE = {"_id": RES_ID, "name": "cooja1"}


class FakeMongo:
    def __init__(self):
        self.simulation_repo = MagicMock()
        self.resource_repo = MagicMock()
        self.generation_repo = MagicMock()
        self.resource_repo.release.return_value = {"_id": RES_ID, "status": "available"}
        self.generation_repo.any_simulation_active.return_value = False
        self.generation_repo.all_simulations_done.return_value = True


def build_runner(
    mongo,
    *,
    prepare=None,
    execute=None,
    on_release=None,
    cleanup_local=None,
) -> SimulationRunner:
    return SimulationRunner(
        mongo,
        # MagicMock tolerates the on_started keyword the real executor takes.
        execute=execute or MagicMock(),
        prepare=prepare or (lambda sim, repo: (True, ["tmp/a.c"], ["a.c"])),
        on_release=on_release,
        cleanup_local=cleanup_local,
    )


@pytest.fixture
def transport():
    """Patch build_transport, handing out a context-manager mock."""
    with patch("lib.scheduler.runner.build_transport") as factory:
        instance = MagicMock()
        instance.__enter__.return_value = instance
        instance.__exit__.return_value = False
        factory.return_value = instance
        yield instance


# ── The happy path ───────────────────────────────────────────────────────────

class TestSuccessfulRun:
    def test_sends_files_then_executes(self, transport):
        mongo = FakeMongo()
        execute = MagicMock()
        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        # A defensive cleanup first, in case a previous run died before its
        # own; then the inputs; then Cooja; then the run's own artifacts, now
        # that the result is published and nothing needs them.
        assert transport.cleanup.call_args_list[0].args == ()
        transport.put_files.assert_called_once_with(["tmp/a.c"], ["a.c"])
        execute.assert_called_once()
        assert execute.call_args.args[0] is SIM
        assert "run.pgid" in transport.cleanup.call_args_list[-1].args[0]

    def test_the_container_becomes_busy_when_the_run_actually_starts(self, transport):
        """`reserved` -> `busy` is driven by the execution, not by the plan.

        The state then means what it says, and the transition carries the
        remote PID in one atomic write — which is what a later liveness probe
        needs in order to ask whether the run is still going.
        """
        mongo = FakeMongo()

        def execute(sim, tr, repo, remote_files, on_started=None, **kwargs):
            mongo.resource_repo.mark_busy.assert_not_called()
            on_started(4242)

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.mark_busy.assert_called_once_with(
            RES_ID, SIM_OID, ATTEMPT, 4242)
        mongo.simulation_repo.set_remote_pid.assert_called_once_with(
            SIM_OID, ATTEMPT, 4242)

    def test_a_run_that_never_starts_leaves_it_reserved(self, transport):
        # The release guard matches on simulation + attempt, which are set at
        # reservation time, so a container never promoted to busy is still
        # given back.
        mongo = FakeMongo()
        build_runner(mongo, execute=lambda *a, **k: None).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        mongo.resource_repo.mark_busy.assert_not_called()
        mongo.resource_repo.release.assert_called_once_with(
            RES_ID, SIM_OID, ATTEMPT, container_state_counts=True)

    def test_a_lost_pid_write_does_not_fail_the_simulation(self, transport):
        # Bookkeeping only: the run is already going, and failing it over a
        # lost write would waste the whole simulation.
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.side_effect = RuntimeError("mongo down")

        build_runner(
            mongo,
            execute=lambda sim, tr, repo, rf, on_started=None, **kw: on_started(99),
        ).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_error.assert_not_called()
        mongo.resource_repo.release.assert_called_once()

    def test_releases_the_container_afterwards(self, transport):
        mongo = FakeMongo()
        build_runner(mongo).run(SIM, HOST, RESOURCE, ATTEMPT)
        mongo.resource_repo.release.assert_called_once_with(
            RES_ID, SIM_OID, ATTEMPT, container_state_counts=True)

    def test_signals_capacity_after_releasing(self, transport):
        mongo = FakeMongo()
        notified: list[int] = []
        build_runner(mongo, on_release=lambda: notified.append(1)).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        # What wakes a dispatcher holding a job for want of capacity.
        assert notified == [1]

    def test_local_files_are_removed(self, transport):
        mongo = FakeMongo()
        removed: list[list[str]] = []
        build_runner(mongo, cleanup_local=removed.append).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        assert removed == [["tmp/a.c"]]


# ── Every failure still releases ─────────────────────────────────────────────

class TestReleaseOnEveryPath:
    def test_a_failed_preparation_releases_and_marks_the_error(self, transport):
        mongo = FakeMongo()
        build_runner(mongo, prepare=lambda sim, repo: (False, [], [])).run(
            SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_error.assert_called_once()
        assert "prepare" in mongo.simulation_repo.mark_error.call_args.args[1].lower()
        mongo.resource_repo.release.assert_called_once_with(
            RES_ID, SIM_OID, ATTEMPT, container_state_counts=True)
        transport.put_files.assert_not_called()

    def test_a_manual_worker_is_released_without_dockers_veto(self, transport):
        """The worker's own settings decide, exactly as in the health cycle.

        A stale `missing` from a host whose Docker daemon the SSH user cannot
        reach must not take the worker out of the pool on the way back.
        """
        mongo = FakeMongo()
        resource = {**RESOURCE, "connection": {"port": 2223}}

        build_runner(mongo).run(SIM, HOST, resource, ATTEMPT)

        assert mongo.resource_repo.release.call_args.kwargs[
            "container_state_counts"] is False

    def test_a_discovered_worker_keeps_dockers_veto(self, transport):
        mongo = FakeMongo()
        build_runner(mongo).run(SIM, HOST, RESOURCE, ATTEMPT)
        assert mongo.resource_repo.release.call_args.kwargs[
            "container_state_counts"] is True

    def test_an_unreachable_container_is_a_plain_failure(self, transport):
        # The simulation never started, so this is not the "lost contact with a
        # running simulation" case the reconciliation layer handles.
        mongo = FakeMongo()
        transport.__enter__.side_effect = TransportUnreachable("no route")
        build_runner(mongo).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_error.assert_called_once()
        mongo.resource_repo.release.assert_called_once_with(
            RES_ID, SIM_OID, ATTEMPT, container_state_counts=True)

    def test_a_transport_error_releases(self, transport):
        mongo = FakeMongo()
        transport.put_files.side_effect = TransportError("scp failed")
        build_runner(mongo).run(SIM, HOST, RESOURCE, ATTEMPT)
        mongo.resource_repo.release.assert_called_once()

    def test_an_unexpected_error_releases(self, transport):
        mongo = FakeMongo()
        build_runner(mongo, execute=MagicMock(side_effect=ZeroDivisionError())).run(
            SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_error.assert_called_once()
        mongo.resource_repo.release.assert_called_once()

    def test_a_failing_release_still_signals_capacity(self, transport):
        # Otherwise one failed release would stall every held job.
        mongo = FakeMongo()
        mongo.resource_repo.release.side_effect = RuntimeError("mongo down")
        notified: list[int] = []
        build_runner(mongo, on_release=lambda: notified.append(1)).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        assert notified == [1]

    def test_local_files_are_removed_even_on_failure(self, transport):
        mongo = FakeMongo()
        removed: list[list[str]] = []
        build_runner(
            mongo,
            execute=MagicMock(side_effect=RuntimeError("boom")),
            cleanup_local=removed.append,
        ).run(SIM, HOST, RESOURCE, ATTEMPT)
        assert removed == [["tmp/a.c"]]

    def test_a_crashing_prepare_is_contained(self, transport):
        mongo = FakeMongo()

        def prepare(sim, repo):
            raise OSError("disk full")

        build_runner(mongo, prepare=prepare).run(SIM, HOST, RESOURCE, ATTEMPT)
        mongo.simulation_repo.mark_error.assert_called_once()
        mongo.resource_repo.release.assert_called_once()


# ── The release guard ────────────────────────────────────────────────────────

class TestReleaseGuard:
    def test_a_container_taken_over_elsewhere_is_not_freed_twice(self, transport):
        # release() returns None when this attempt no longer holds the
        # resource: a recovery path decided the attempt was gone and the
        # container may already be running something else.
        mongo = FakeMongo()
        mongo.resource_repo.release.return_value = None
        notified: list[int] = []

        build_runner(mongo, on_release=lambda: notified.append(1)).run(
            SIM, HOST, RESOURCE, ATTEMPT)

        assert mongo.resource_repo.release.call_count == 1
        assert notified == [1]

    def test_the_attempt_identifier_is_carried_through(self, transport):
        mongo = FakeMongo()
        build_runner(
            mongo,
            execute=lambda sim, tr, repo, rf, on_started=None, **kw: on_started(7),
        ).run(SIM, HOST, RESOURCE, "other-attempt")

        # Both ends of the lifecycle are guarded by the same attempt, so a
        # thread from an earlier one cannot report a start or free a container.
        assert mongo.resource_repo.mark_busy.call_args.args[2] == "other-attempt"
        assert mongo.simulation_repo.set_remote_pid.call_args.args[1] == "other-attempt"
        assert mongo.resource_repo.release.call_args.args[2] == "other-attempt"


class TestUnconfirmedStop:
    """A worker that may still be running the last simulation is not idle.

    `terminate()` used to report success whenever the command executed, so a
    run that could not be stopped released its container anyway. The next
    simulation then shared the worker with an orphaned JVM and its timing was
    silently wrong — and the measurements are the point of the platform.
    """

    def test_the_worker_leaves_the_pool_administratively(self, transport):
        mongo = FakeMongo()

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            raise TerminationUnconfirmed("could not be stopped")

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        # `enabled = False` rather than a status, because a successful probe
        # cannot undo it: a health check cannot tell a leftover JVM from an
        # idle container and would hand the worker straight back.
        mongo.resource_repo.set_enabled.assert_called_once_with(RES_ID, False)

    def test_the_simulation_is_still_marked_failed(self, transport):
        mongo = FakeMongo()

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            raise TerminationUnconfirmed("could not be stopped")

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.mark_error.assert_called_once()
        assert "stopped" in mongo.simulation_repo.mark_error.call_args.args[1]

    def test_a_plain_failure_leaves_the_worker_enabled(self, transport):
        mongo = FakeMongo()

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            raise RuntimeError("the simulation did not complete")

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.set_enabled.assert_not_called()

    def test_a_failure_to_disable_does_not_crash_the_thread(self, transport):
        mongo = FakeMongo()
        mongo.resource_repo.set_enabled.side_effect = RuntimeError("mongo gone")

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            raise TerminationUnconfirmed("could not be stopped")

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.release.assert_called_once()


class TestLostOwnership:
    """A thread that lost its container must not act as though it still has it.

    A recovery pass or the stale-reservation sweep can hand a container on
    while a runner thread is working. Until the writes were guarded, that
    thread went on to promote the container, publish a result over the newer
    attempt and release a reservation that was no longer its own — three ways
    for two attempts to collide on one document and one worker.

    The decision taken: stop the remote run and return the simulation to the
    queue. The pool already believes the container is free, so a running JVM
    on it would invite a second simulation onto the same worker.
    """

    def test_a_refused_promotion_stops_the_run(self, transport):
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.return_value = False
        reached_the_end = []

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            on_started(4242)
            reached_the_end.append(True)

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        assert reached_the_end == []

    def test_it_goes_back_to_the_queue_not_to_an_error(self, transport):
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.return_value = False

        build_runner(
            mongo,
            execute=lambda sim, tr, repo, rf, on_started=None, **kw: on_started(1),
        ).run(SIM, HOST, RESOURCE, ATTEMPT)

        # The work was sound; it simply is not ours any more.
        mongo.simulation_repo.mark_error.assert_not_called()
        mongo.simulation_repo.requeue.assert_called_once()
        sim_oid, attempt, reason = mongo.simulation_repo.requeue.call_args.args
        assert sim_oid == SIM_OID and attempt == ATTEMPT
        assert "reservation" in reason

    def test_a_lost_write_is_still_only_bookkeeping(self, transport):
        """An exception is Mongo blinking, not a refusal.

        The simulation is already running, and failing it over a write that
        did not land would waste the whole run.
        """
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.side_effect = RuntimeError("mongo blinked")
        reached_the_end = []

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            on_started(4242)
            reached_the_end.append(True)

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        assert reached_the_end == [True]
        mongo.simulation_repo.requeue.assert_not_called()

    def test_a_refused_publication_goes_back_to_the_queue(self, transport):
        # What `mark_done` returning false means, raised by the execution.
        mongo = FakeMongo()

        def execute(sim, tr, repo, rf, on_started=None, **kw):
            raise LostOwnership("the simulation was taken over")

        build_runner(mongo, execute=execute).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.simulation_repo.requeue.assert_called_once()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_the_container_is_still_given_back(self, transport):
        # The release is guarded by the attempt, so this frees nothing if the
        # container is already someone else's — but it must still be tried,
        # for the case where the simulation moved on and the container did not.
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.return_value = False

        build_runner(
            mongo,
            execute=lambda sim, tr, repo, rf, on_started=None, **kw: on_started(1),
        ).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.release.assert_called_once()

    def test_a_newer_attempt_already_owning_it_is_left_alone(self, transport):
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.return_value = False
        mongo.simulation_repo.requeue.return_value = False      # guard refused

        build_runner(
            mongo,
            execute=lambda sim, tr, repo, rf, on_started=None, **kw: on_started(1),
        ).run(SIM, HOST, RESOURCE, ATTEMPT)

        # Nothing else to do: the newer attempt is doing the work.
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_failed_requeue_does_not_crash_the_thread(self, transport):
        mongo = FakeMongo()
        mongo.resource_repo.mark_busy.return_value = False
        mongo.simulation_repo.requeue.side_effect = RuntimeError("mongo gone")

        build_runner(
            mongo,
            execute=lambda sim, tr, repo, rf, on_started=None, **kw: on_started(1),
        ).run(SIM, HOST, RESOURCE, ATTEMPT)

        mongo.resource_repo.release.assert_called_once()


# ── Generation bookkeeping ───────────────────────────────────────────────────

class TestGenerationClosing:
    """The runner asks; the decision itself belongs to the repository.

    It used to carry its own copy of the five lines that decide, as did three
    other places — and a fifth, the stale-reservation sweep, carried none.
    ``close_if_finished`` is tested on the repository, where it now lives.
    """

    def test_a_failure_re_evaluates_the_generation(self, transport):
        mongo = FakeMongo()
        build_runner(mongo, prepare=lambda sim, repo: (False, [], [])).run(
            SIM, HOST, RESOURCE, ATTEMPT)
        mongo.generation_repo.close_if_finished.assert_called_once_with(GEN_OID)

    def test_a_failure_from_a_lost_attempt_is_not_recorded(self):
        # The newer attempt owns the simulation; this one's failure is its
        # own business and must not land on the document.
        mongo = FakeMongo()
        mongo.simulation_repo.mark_error.return_value = False

        build_runner(mongo, prepare=lambda sim, repo: (False, [], [])).run(
            SIM, HOST, RESOURCE, ATTEMPT)

        mongo.generation_repo.close_if_finished.assert_not_called()

    def test_a_simulation_without_a_generation_is_fine(self, transport):
        mongo = FakeMongo()
        sim = {"_id": SIM_OID, "status": "Waiting"}
        build_runner(mongo, prepare=lambda s, r: (False, [], [])).run(
            sim, HOST, RESOURCE, ATTEMPT)
        mongo.generation_repo.close_if_finished.assert_not_called()
        mongo.resource_repo.release.assert_called_once()


# ── Host-key learning ────────────────────────────────────────────────────────

class TestHostKeyLearning:
    def test_the_fingerprint_is_reported_with_its_host(self, transport):
        mongo = FakeMongo()
        learned: list[tuple] = []
        runner = SimulationRunner(
            mongo,
            execute=MagicMock(),
            prepare=lambda sim, repo: (True, [], []),
            on_learn_host_key=lambda host_id, fp: learned.append((host_id, fp)),
        )
        runner.run(SIM, HOST, RESOURCE, ATTEMPT)

        with patch("lib.scheduler.runner.build_transport") as factory:
            factory.return_value.__enter__.return_value = factory.return_value
            factory.return_value.__exit__.return_value = False
            runner.run(SIM, HOST, RESOURCE, ATTEMPT)
            sink = factory.call_args.kwargs["on_learn_host_key"]

        sink("SHA256:abc")
        assert learned == [(HOST["_id"], "SHA256:abc")]

    def test_no_sink_means_no_callback_is_passed(self, transport):
        mongo = FakeMongo()
        with patch("lib.scheduler.runner.build_transport") as factory:
            factory.return_value.__enter__.return_value = factory.return_value
            factory.return_value.__exit__.return_value = False
            build_runner(mongo).run(SIM, HOST, RESOURCE, ATTEMPT)
            assert factory.call_args.kwargs["on_learn_host_key"] is None
