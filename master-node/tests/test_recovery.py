"""Tests for reconciling the pool with reality after a restart.

Two regressions are guarded here, and they pull in opposite directions.

Before this feature a restart cost nothing: workers held no state, so the
containers were usable again as soon as the process came back. Reservations
changed that — observed in deployment, rebuilding the image left all eight
containers held forever and the dispatcher had nowhere to place work.

Then detached execution made the opposite failure possible: a run now
*outlives* the master-node, so releasing its container would throw away the
work and risk a second simulation landing on a busy container. Also observed
in deployment — killing the master-node left the JVMs running.

The probe is what reconciles the two, and these tests pin down each of its
answers. It reports a ``RunState`` rather than a boolean, because "not alive"
alone cannot tell a run that finished from one that was interrupted — and
taking the first for the second threw away results that had completed while
the master-node was down.
"""
from datetime import timedelta
from unittest.mock import MagicMock

from bson import ObjectId

from lib.detached import RunState
from lib.recovery import RECOVERY_MESSAGE, reconcile_occupied_resources
from pylib.timeutil import utc_now

#: The probe's answers, named.
ALIVE = RunState(alive=True, finished=False)
GONE = RunState(alive=False, finished=False)
FINISHED = RunState(alive=False, finished=True, exit_code=0, testlog_bytes=780_000)

RES_1 = ObjectId("507f1f77bcf86cd799430021")
RES_2 = ObjectId("507f1f77bcf86cd799430022")
SIM_1 = ObjectId("507f1f77bcf86cd799430003")
GEN_1 = ObjectId("507f1f77bcf86cd799430004")


class FakeMongo:
    def __init__(self, occupied=(), simulation=None):
        self.resource_repo = MagicMock()
        self.simulation_repo = MagicMock()
        self.generation_repo = MagicMock()
        self.host_repo = MagicMock()
        self.host_repo.get.return_value = {"_id": "HOST", "name": "local"}
        self.resource_repo.find_occupied.return_value = list(occupied)
        self.resource_repo.force_release.return_value = True
        self.simulation_repo.get.return_value = simulation
        self.generation_repo.any_simulation_active.return_value = False
        self.generation_repo.all_simulations_done.return_value = False


def resource(_id=RES_1, name="cooja1", simulation_id=SIM_1, **over) -> dict:
    doc = {"_id": _id, "name": name, "current_simulation_id": simulation_id}
    doc.update(over)
    return doc


def simulation(status="Running", generation_id=GEN_1, remote_pid=4242) -> dict:
    return {"_id": SIM_1, "status": status, "generation_id": generation_id,
            "remote_pid": remote_pid}


# ── The probe: what keeps reconciliation from destroying live runs ───────────

class TestAdoption:
    """Detached execution means a run can outlive the master-node.

    The first version of this module released everything and failed every
    simulation, on the (then correct) assumption that Cooja died with the SSH
    channel. Verified in deployment after the detached launch landed: killing
    the master-node left the JVMs running, so releasing their containers
    would have thrown away the work and risked double-booking a container.
    """

    def test_a_live_run_is_adopted_not_destroyed(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda h, r, pid: ALIVE)

        assert report.adopted == ["cooja1"]
        assert report.released == []
        mongo.resource_repo.force_release.assert_not_called()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_dead_run_is_failed_and_released(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda h, r, pid: GONE)

        assert report.released == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_called_once_with(
            SIM_1, RECOVERY_MESSAGE, attempt_id=None)

    def test_an_unreachable_container_is_left_alone(self):
        # The specification's rule: losing contact is not the run failing.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda h, r, pid: None)

        assert report.unreachable == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_probe_that_raises_is_treated_as_unknown(self):
        # An error in the probe must not be able to destroy a running run.
        def exploding(host, resource, pid):
            raise RuntimeError("ssh exploded")

        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        report = reconcile_occupied_resources(mongo, probe=exploding)

        assert report.unreachable == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_the_probe_receives_the_recorded_pid(self):
        seen: list = []
        mongo = FakeMongo(
            occupied=[resource()], simulation=simulation("Running", remote_pid=777))
        reconcile_occupied_resources(
            mongo, probe=lambda h, r, pid: seen.append((h["name"], r["name"], pid)) or GONE)
        assert seen == [("local", "cooja1", 777)]

    def test_no_recorded_pid_means_definitely_gone(self):
        # It never got far enough to record one, so there is nothing to adopt.
        probe_calls = []
        mongo = FakeMongo(
            occupied=[resource()], simulation=simulation("Running", remote_pid=None))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: probe_calls.append(a) or ALIVE)

        assert probe_calls == []
        assert report.released == ["cooja1"]

    def test_a_vanished_host_is_not_probed(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        mongo.host_repo.get.return_value = None
        report = reconcile_occupied_resources(mongo, probe=lambda *a: ALIVE)
        assert report.released == ["cooja1"]

    def test_without_a_probe_everything_is_treated_as_gone(self):
        # Only correct on the legacy path, where a run cannot outlive the
        # master-node. The docstring of the module says so explicitly.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        report = reconcile_occupied_resources(mongo, probe=None)
        assert report.released == ["cooja1"]


class TestResumingAdoptedRuns:
    """Adoption hands the run on to be followed, not just left alone.

    Without this the container is correctly preserved but the run is wasted:
    nothing collects its result, so it resolves through SIM_TIMEOUT_SEC.
    """

    def test_an_adopted_run_is_handed_over_for_following(self):
        resumed: list = []
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        mongo.resource_repo.find_occupied.return_value = [
            {**resource(), "attempt_id": "att-1"}]

        reconcile_occupied_resources(
            mongo, probe=lambda *a: ALIVE,
            on_adopt=lambda sim, host, res, attempt, pid: resumed.append(
                (sim["_id"], host["name"], res["name"], attempt, pid)))

        assert resumed == [(SIM_1, "local", "cooja1", "att-1", 4242)]

    def test_the_attempt_id_falls_back_to_the_simulation(self):
        resumed: list = []
        mongo = FakeMongo(
            occupied=[resource()],
            simulation={**simulation("Running"), "attempt_id": "from-sim"})

        reconcile_occupied_resources(
            mongo, probe=lambda *a: ALIVE,
            on_adopt=lambda sim, host, res, attempt, pid: resumed.append(attempt))

        assert resumed == ["from-sim"]

    def test_without_an_attempt_id_it_is_left_alone(self):
        # Resuming needs it to release the container afterwards, and leaving
        # the run untouched beats freeing a container something is using.
        resumed: list = []
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: ALIVE, on_adopt=lambda *a: resumed.append(a))
        assert resumed == []
        assert report.adopted == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()

    def test_a_failing_handover_does_not_release_the_container(self):
        def exploding(*args):
            raise RuntimeError("thread pool exhausted")

        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        mongo.resource_repo.find_occupied.return_value = [
            {**resource(), "attempt_id": "att-1"}]

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: ALIVE, on_adopt=exploding)

        assert report.adopted == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_nothing_is_handed_over_for_a_dead_run(self):
        resumed: list = []
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(
            mongo, probe=lambda *a: GONE, on_adopt=lambda *a: resumed.append(a))
        assert resumed == []


class TestReleasesOrphans:
    def test_nothing_held_is_a_no_op(self):
        mongo = FakeMongo()
        assert reconcile_occupied_resources(mongo).total == 0
        mongo.resource_repo.force_release.assert_not_called()

    def test_every_held_container_is_freed(self):
        mongo = FakeMongo(
            occupied=[resource(RES_1, "cooja1"), resource(RES_2, "cooja2")],
            simulation=simulation(),
        )
        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)
        assert report.released == ["cooja1", "cooja2"]
        freed = [c.args[0] for c in mongo.resource_repo.force_release.call_args_list]
        assert freed == [RES_1, RES_2]

    def test_the_guard_is_skipped_deliberately(self):
        # force_release, not release: the attempt that held it is known to be
        # gone, so there is no attempt id left to match on.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation())
        reconcile_occupied_resources(mongo, probe=lambda *a: GONE)
        mongo.resource_repo.release.assert_not_called()
        assert "restart" in mongo.resource_repo.force_release.call_args.args[1]

    def test_one_failure_does_not_stop_the_rest(self):
        mongo = FakeMongo(
            occupied=[resource(RES_1, "cooja1"), resource(RES_2, "cooja2")],
            simulation=simulation(),
        )
        mongo.resource_repo.force_release.side_effect = [RuntimeError("boom"), True]
        # A single bad document must not leave the remaining containers stuck.
        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)
        assert report.released == ["cooja2"]


class TestInterruptedSimulations:
    def test_a_run_that_is_really_over_is_marked_failed(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(mongo, probe=lambda *a: GONE)
        mongo.simulation_repo.mark_error.assert_called_once_with(
            SIM_1, RECOVERY_MESSAGE, attempt_id=None)

    def test_a_finished_simulations_result_is_never_overwritten(self):
        # The run completed and only the release was lost. Whatever happens to
        # the worker, the measurement is not touched.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Done"))

        reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert mongo.simulation_repo.mark_error.call_count == 0

    def test_a_finished_simulation_whose_process_is_gone_frees_the_worker(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Done"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert report.released == ["cooja1"]

    def test_a_terminal_simulation_with_a_live_process_holds_the_worker(self):
        """The assumption this used to make, stated and refused.

        "Nothing is running, so there is no question to ask" is wrong, and it
        is wrong in the one case that matters: a simulation marked in error
        *because* nobody could confirm its process had stopped. If the write
        that was meant to record that also failed, this is the only thing
        between the database coming back and a worker with a live JVM going
        into the pool.
        """
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Error"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: ALIVE)

        assert report.released == []
        assert report.held == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()
        # And written down, so the next pass does not have to ask again.
        mongo.resource_repo.hold.assert_called_once()

    def test_leftover_artifacts_hold_it_too(self):
        """A testlog is an uncollected result; the next run deletes it."""
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Error"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: FINISHED)

        assert report.held == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()

    def test_a_terminal_simulation_with_no_pid_is_not_probed(self):
        """Nothing was ever started under this attempt, so there is genuinely
        no question to ask."""
        probed = []
        mongo = FakeMongo(
            occupied=[resource()],
            simulation=simulation("Error", remote_pid=None))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: probed.append(a) or GONE)

        assert probed == []
        assert report.released == ["cooja1"]

    def test_an_unreachable_container_holding_a_terminal_simulation_waits(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Error"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: None)

        assert report.released == []
        assert report.unreachable == ["cooja1"]

    def test_without_a_probe_it_refuses_rather_than_assumes(self):
        """The conservative direction: a worker that waits for the next pass
        costs less than one that takes a second simulation."""
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Error"))

        report = reconcile_occupied_resources(mongo)

        assert report.released == []
        assert report.held == ["cooja1"]

    def test_a_container_holding_nothing_is_still_freed(self):
        mongo = FakeMongo(occupied=[resource(simulation_id=None)])
        report = reconcile_occupied_resources(mongo, probe=lambda *a: ALIVE)
        assert report.released == ["cooja1"]
        mongo.simulation_repo.get.assert_not_called()

    def test_a_deleted_simulation_does_not_block_recovery(self):
        mongo = FakeMongo(occupied=[resource()], simulation=None)
        report = reconcile_occupied_resources(mongo, probe=lambda *a: ALIVE)
        assert report.released == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_not_called()


class TestFinishedWhileTheMasterWasDown:
    """A run that completed during the outage is collected, not discarded.

    The probe used to answer a bare boolean. With the process gone, the
    simulation was marked failed and the container freed — and the next job's
    defensive cleanup removed the testlog that proved the run had succeeded.
    Now a finished run with its log still on disk is *adopted*: the follower
    sees the exit code on its first poll and the collection proceeds.
    """

    def test_a_finished_run_is_adopted_not_failed(self):
        mongo = FakeMongo(
            occupied=[{**resource(), "attempt_id": "attempt-A"}],
            simulation=simulation("Running"))
        resumed: list = []

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: FINISHED,
            on_adopt=lambda *a: resumed.append(a))

        assert report.adopted == ["cooja1"]
        assert report.failed_simulations == []
        assert len(resumed) == 1
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_its_container_is_not_released(self):
        # Releasing it would let the next simulation clear the artifacts the
        # collection still needs.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: FINISHED, on_adopt=lambda *a: None)

        assert report.released == []
        mongo.resource_repo.force_release.assert_not_called()

    def test_a_vanished_run_that_left_a_log_is_still_collected(self):
        """No exit code, but bytes on disk.

        The cautious reading: the alternative is deleting them. This is also
        the shape of the race that cost a simulation in a pilot — reaped
        before the exit file landed.
        """
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        left_behind = RunState(alive=False, finished=False, testlog_bytes=512)

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: left_behind, on_adopt=lambda *a: None)

        assert report.adopted == ["cooja1"]

    def test_a_run_with_nothing_to_show_is_still_a_failure(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert report.failed_simulations == [str(SIM_1)]
        assert report.released == ["cooja1"]

    def test_an_unreachable_container_is_still_left_alone(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: None)

        assert report.unreachable == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_not_called()
        mongo.resource_repo.force_release.assert_not_called()


class TestPeriodicReconciliation:
    """Running once at startup was not enough.

    A container whose host was unreachable at boot kept its reservation —
    correctly, since unreachable says nothing about the run — and then nothing
    ever came back to it: the health cycle skips occupied containers and the
    TTL sweep only covers reserved ones. It stayed `busy` until someone
    restarted the master-node again.

    So the same pass runs every cycle, over the containers no thread of this
    process is watching.
    """

    def test_an_attempt_being_followed_is_skipped(self):
        mongo = FakeMongo(
            occupied=[{**resource(), "attempt_id": "att-1"}],
            simulation=simulation("Running"))
        probed: list = []

        report = reconcile_occupied_resources(
            mongo,
            probe=lambda *a: probed.append(a) or ALIVE,
            is_followed=lambda attempt: attempt == "att-1",
        )

        # Its follower will collect and release; a second reconciliation would
        # mean two followers for one attempt.
        assert report.followed == ["cooja1"]
        assert probed == []
        mongo.resource_repo.force_release.assert_not_called()

    def test_an_attempt_nobody_follows_is_reconciled(self):
        mongo = FakeMongo(
            occupied=[{**resource(), "attempt_id": "att-1"}],
            simulation=simulation("Running"))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: GONE, is_followed=lambda attempt: False)

        assert report.followed == []
        assert report.released == ["cooja1"]

    def test_a_container_unreachable_earlier_is_retried_later(self):
        """The case that used to stay `busy` for ever."""
        mongo = FakeMongo(
            occupied=[{**resource(), "attempt_id": "att-1"}],
            simulation=simulation("Running"))

        # First pass: no answer, so nothing is touched.
        first = reconcile_occupied_resources(
            mongo, probe=lambda *a: None, is_followed=lambda a: False)
        assert first.unreachable == ["cooja1"]

        # Later pass, host back: the run is gone and the container comes back.
        second = reconcile_occupied_resources(
            mongo, probe=lambda *a: GONE, is_followed=lambda a: False)
        assert second.released == ["cooja1"]

    def test_without_the_hook_nothing_is_skipped(self):
        # Startup: no thread is following anything yet.
        mongo = FakeMongo(
            occupied=[{**resource(), "attempt_id": "att-1"}],
            simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert report.followed == []
        assert report.released == ["cooja1"]


class TestGenerationClosing:
    def test_the_generation_is_re_evaluated(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(mongo, probe=lambda *a: GONE)
        mongo.generation_repo.close_if_finished.assert_called_once_with(GEN_1)

    def test_an_adopted_run_does_not_close_its_generation(self):
        # It is still going; declaring the generation finished would be wrong.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(mongo, probe=lambda *a: ALIVE)
        mongo.generation_repo.mark_error.assert_not_called()
        mongo.generation_repo.mark_done.assert_not_called()

    def test_a_simulation_without_a_generation_is_fine(self):
        mongo = FakeMongo(
            occupied=[resource()], simulation=simulation("Running", generation_id=None))
        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)
        assert report.released == ["cooja1"]
        mongo.generation_repo.any_simulation_active.assert_not_called()


# ── R04: a placement in progress is not an orphan ───────────────────────────

class TestFreshReservationsAreLeftAlone:
    """A reservation seconds old belongs to a thread that is still working.

    Reproduces R04. The periodic pass read a held container, saw that nobody
    was following it — the registration was taken later, on the runner thread
    — and treated it as the leftover of a dead process. A reservation created
    moments earlier was freed and its simulation failed while the thread that
    owned it was still assembling its inputs.

    Two mechanisms now prevent it, and both are tested: the dispatcher claims
    the attempt before the reservation can be observed, and this pass refuses
    to conclude anything from a ``reserved`` container younger than the TTL
    that the sweep owns.
    """

    def _just_reserved(self):
        return resource(
            status="reserved", reserved_at=utc_now(), attempt_id="A")

    def test_a_recent_reservation_is_not_released(self):
        mongo = FakeMongo(
            occupied=[self._just_reserved()],
            simulation=simulation("Waiting", remote_pid=None))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: GONE, reserved_grace_sec=600)

        assert report.preparing == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_an_old_reservation_is_still_reconciled(self):
        """The grace is a grace, not an exemption."""
        old = resource(
            status="reserved",
            reserved_at=utc_now() - timedelta(seconds=1200),
            attempt_id="A")
        mongo = FakeMongo(
            occupied=[old], simulation=simulation("Waiting", remote_pid=None))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: GONE, reserved_grace_sec=600)

        assert report.preparing == []
        mongo.resource_repo.force_release.assert_called_once()

    def test_a_reservation_that_did_start_is_reconciled_regardless(self):
        """A pid means the run began; the status write is simply behind."""
        started = resource(
            status="reserved", reserved_at=utc_now(), attempt_id="A",
            remote_pid=4242)
        mongo = FakeMongo(
            occupied=[started], simulation=simulation("Running"))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: ALIVE, reserved_grace_sec=600)

        assert report.adopted == ["cooja1"]

    def test_startup_concludes_about_everything(self):
        """Nothing at startup can belong to this process, so no grace."""
        mongo = FakeMongo(
            occupied=[self._just_reserved()],
            simulation=simulation("Waiting", remote_pid=None))

        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert report.preparing == []
        mongo.resource_repo.force_release.assert_called_once()


class TestTheWritesAreGuardedByTheAttempt:
    def test_the_release_insists_on_the_attempt_it_read(self):
        mongo = FakeMongo(
            occupied=[resource(status="busy", attempt_id="A")],
            simulation=simulation("Running"))

        reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert mongo.resource_repo.force_release.call_args.kwargs[
            "expect_attempt_id"] == "A"

    def test_a_refused_release_leaves_the_simulation_alone(self):
        """Between the read and the write, a new placement may have landed.

        Failing its simulation then marks a failure against work that is
        running perfectly well.
        """
        mongo = FakeMongo(
            occupied=[resource(status="busy", attempt_id="A")],
            simulation=simulation("Running"))
        mongo.resource_repo.force_release.return_value = False

        report = reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        mongo.simulation_repo.mark_error.assert_not_called()
        assert report.released == []
        assert report.failed_simulations == []

    def test_the_failure_is_guarded_too(self):
        mongo = FakeMongo(
            occupied=[resource(status="busy", attempt_id="A")],
            simulation=simulation("Running"))

        reconcile_occupied_resources(mongo, probe=lambda *a: GONE)

        assert mongo.simulation_repo.mark_error.call_args.kwargs[
            "attempt_id"] == "A"


# ── R05.3: the pid is written in two places ─────────────────────────────────

class TestThePidIsFoundWhereverItWasRecorded:
    """``_started`` writes the resource first, then the simulation.

    A failure between the two writes left the only copy of the pid on the
    resource — where this never looked. The probe was then asked about
    nothing, answered "gone", and a live run had its container released
    (R05.3).
    """

    def test_the_resource_pid_is_used_when_the_simulation_has_none(self):
        asked = []

        def probe(host, res, pid):
            asked.append(pid)
            return ALIVE

        mongo = FakeMongo(
            occupied=[resource(status="busy", attempt_id="A", remote_pid=9001)],
            simulation=simulation("Running", remote_pid=None))

        report = reconcile_occupied_resources(mongo, probe=probe)

        assert asked == [9001]
        assert report.adopted == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()

    def test_an_adopted_run_is_resumed_with_that_pid(self):
        resumed = []
        mongo = FakeMongo(
            occupied=[resource(status="busy", attempt_id="A", remote_pid=9001)],
            simulation=simulation("Running", remote_pid=None))

        reconcile_occupied_resources(
            mongo, probe=lambda *a: ALIVE,
            on_adopt=lambda sim, h, r, attempt, pid: resumed.append((attempt, pid)))

        assert resumed == [("A", 9001)]

    def test_no_pid_anywhere_is_still_a_definite_gone(self):
        mongo = FakeMongo(
            occupied=[resource(status="busy", attempt_id="A", remote_pid=None)],
            simulation=simulation("Running", remote_pid=None))

        reconcile_occupied_resources(mongo, probe=lambda *a: ALIVE)

        # The probe is never asked: there is nothing to ask about.
        mongo.resource_repo.force_release.assert_called_once()
