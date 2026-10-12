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
three answers.
"""
from unittest.mock import MagicMock

from bson import ObjectId

from lib.recovery import RECOVERY_MESSAGE, reconcile_occupied_resources

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


def resource(_id=RES_1, name="cooja1", simulation_id=SIM_1) -> dict:
    return {"_id": _id, "name": name, "current_simulation_id": simulation_id}


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

        report = reconcile_occupied_resources(mongo, probe=lambda h, r, pid: True)

        assert report.adopted == ["cooja1"]
        assert report.released == []
        mongo.resource_repo.force_release.assert_not_called()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_dead_run_is_failed_and_released(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))

        report = reconcile_occupied_resources(mongo, probe=lambda h, r, pid: False)

        assert report.released == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_called_once_with(SIM_1, RECOVERY_MESSAGE)

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
            mongo, probe=lambda h, r, pid: seen.append((h["name"], r["name"], pid)))
        assert seen == [("local", "cooja1", 777)]

    def test_no_recorded_pid_means_definitely_gone(self):
        # It never got far enough to record one, so there is nothing to adopt.
        probe_calls = []
        mongo = FakeMongo(
            occupied=[resource()], simulation=simulation("Running", remote_pid=None))

        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: probe_calls.append(a) or True)

        assert probe_calls == []
        assert report.released == ["cooja1"]

    def test_a_vanished_host_is_not_probed(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        mongo.host_repo.get.return_value = None
        report = reconcile_occupied_resources(mongo, probe=lambda *a: True)
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
            mongo, probe=lambda *a: True,
            on_adopt=lambda sim, host, res, attempt, pid: resumed.append(
                (sim["_id"], host["name"], res["name"], attempt, pid)))

        assert resumed == [(SIM_1, "local", "cooja1", "att-1", 4242)]

    def test_the_attempt_id_falls_back_to_the_simulation(self):
        resumed: list = []
        mongo = FakeMongo(
            occupied=[resource()],
            simulation={**simulation("Running"), "attempt_id": "from-sim"})

        reconcile_occupied_resources(
            mongo, probe=lambda *a: True,
            on_adopt=lambda sim, host, res, attempt, pid: resumed.append(attempt))

        assert resumed == ["from-sim"]

    def test_without_an_attempt_id_it_is_left_alone(self):
        # Resuming needs it to release the container afterwards, and leaving
        # the run untouched beats freeing a container something is using.
        resumed: list = []
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        report = reconcile_occupied_resources(
            mongo, probe=lambda *a: True, on_adopt=lambda *a: resumed.append(a))
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
            mongo, probe=lambda *a: True, on_adopt=exploding)

        assert report.adopted == ["cooja1"]
        mongo.resource_repo.force_release.assert_not_called()
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_nothing_is_handed_over_for_a_dead_run(self):
        resumed: list = []
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(
            mongo, probe=lambda *a: False, on_adopt=lambda *a: resumed.append(a))
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
        report = reconcile_occupied_resources(mongo, probe=lambda *a: False)
        assert report.released == ["cooja1", "cooja2"]
        freed = [c.args[0] for c in mongo.resource_repo.force_release.call_args_list]
        assert freed == [RES_1, RES_2]

    def test_the_guard_is_skipped_deliberately(self):
        # force_release, not release: the attempt that held it is known to be
        # gone, so there is no attempt id left to match on.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation())
        reconcile_occupied_resources(mongo, probe=lambda *a: False)
        mongo.resource_repo.release.assert_not_called()
        assert "restart" in mongo.resource_repo.force_release.call_args.args[1]

    def test_one_failure_does_not_stop_the_rest(self):
        mongo = FakeMongo(
            occupied=[resource(RES_1, "cooja1"), resource(RES_2, "cooja2")],
            simulation=simulation(),
        )
        mongo.resource_repo.force_release.side_effect = [RuntimeError("boom"), True]
        # A single bad document must not leave the remaining containers stuck.
        report = reconcile_occupied_resources(mongo, probe=lambda *a: False)
        assert report.released == ["cooja2"]


class TestInterruptedSimulations:
    def test_a_run_that_is_really_over_is_marked_failed(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(mongo, probe=lambda *a: False)
        mongo.simulation_repo.mark_error.assert_called_once_with(SIM_1, RECOVERY_MESSAGE)

    def test_a_finished_simulation_is_left_alone(self):
        # The run completed and only the release was lost; there is nothing to
        # correct, and overwriting its result would destroy the measurement.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Done"))
        report = reconcile_occupied_resources(mongo, probe=lambda *a: True)
        assert report.released == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_finished_simulation_is_not_even_probed(self):
        # Nothing is running, so there is no question to ask.
        probed = []
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Error"))
        reconcile_occupied_resources(mongo, probe=lambda *a: probed.append(a))
        assert probed == []

    def test_a_container_holding_nothing_is_still_freed(self):
        mongo = FakeMongo(occupied=[resource(simulation_id=None)])
        report = reconcile_occupied_resources(mongo, probe=lambda *a: True)
        assert report.released == ["cooja1"]
        mongo.simulation_repo.get.assert_not_called()

    def test_a_deleted_simulation_does_not_block_recovery(self):
        mongo = FakeMongo(occupied=[resource()], simulation=None)
        report = reconcile_occupied_resources(mongo, probe=lambda *a: True)
        assert report.released == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_not_called()


class TestGenerationClosing:
    def test_the_generation_is_re_evaluated(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(mongo, probe=lambda *a: False)
        mongo.generation_repo.mark_error.assert_called_once_with(GEN_1)

    def test_a_generation_with_work_left_is_untouched(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        mongo.generation_repo.any_simulation_active.return_value = True
        reconcile_occupied_resources(mongo, probe=lambda *a: False)
        mongo.generation_repo.mark_error.assert_not_called()
        mongo.generation_repo.mark_done.assert_not_called()

    def test_an_adopted_run_does_not_close_its_generation(self):
        # It is still going; declaring the generation finished would be wrong.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        reconcile_occupied_resources(mongo, probe=lambda *a: True)
        mongo.generation_repo.mark_error.assert_not_called()
        mongo.generation_repo.mark_done.assert_not_called()

    def test_a_simulation_without_a_generation_is_fine(self):
        mongo = FakeMongo(
            occupied=[resource()], simulation=simulation("Running", generation_id=None))
        report = reconcile_occupied_resources(mongo, probe=lambda *a: False)
        assert report.released == ["cooja1"]
        mongo.generation_repo.any_simulation_active.assert_not_called()
