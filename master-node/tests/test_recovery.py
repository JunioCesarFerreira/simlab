"""Tests for recovering the pool after a restart.

Regression guard. Before this feature a restart cost nothing — workers held no
state, so the containers were usable again as soon as the process came back.
Reservations changed that: a container claimed by a process that is gone stays
claimed. Observed in deployment, rebuilding the image left all eight
containers busy forever and the dispatcher had nowhere to place work.
"""
from unittest.mock import MagicMock

from bson import ObjectId

from lib.recovery import RECOVERY_MESSAGE, release_orphaned_reservations

RES_1 = ObjectId("507f1f77bcf86cd799430021")
RES_2 = ObjectId("507f1f77bcf86cd799430022")
SIM_1 = ObjectId("507f1f77bcf86cd799430003")
GEN_1 = ObjectId("507f1f77bcf86cd799430004")


class FakeMongo:
    def __init__(self, occupied=(), simulation=None):
        self.resource_repo = MagicMock()
        self.simulation_repo = MagicMock()
        self.generation_repo = MagicMock()
        self.resource_repo.find_occupied.return_value = list(occupied)
        self.resource_repo.force_release.return_value = True
        self.simulation_repo.get.return_value = simulation
        self.generation_repo.any_simulation_active.return_value = False
        self.generation_repo.all_simulations_done.return_value = False


def resource(_id=RES_1, name="cooja1", simulation_id=SIM_1) -> dict:
    return {"_id": _id, "name": name, "current_simulation_id": simulation_id}


def simulation(status="Running", generation_id=GEN_1) -> dict:
    return {"_id": SIM_1, "status": status, "generation_id": generation_id}


class TestReleasesOrphans:
    def test_nothing_held_is_a_no_op(self):
        mongo = FakeMongo()
        assert release_orphaned_reservations(mongo) == 0
        mongo.resource_repo.force_release.assert_not_called()

    def test_every_held_container_is_freed(self):
        mongo = FakeMongo(
            occupied=[resource(RES_1, "cooja1"), resource(RES_2, "cooja2")],
            simulation=simulation(),
        )
        assert release_orphaned_reservations(mongo) == 2
        freed = [c.args[0] for c in mongo.resource_repo.force_release.call_args_list]
        assert freed == [RES_1, RES_2]

    def test_the_guard_is_skipped_deliberately(self):
        # force_release, not release: the attempt that held it is known to be
        # gone, so there is no attempt id left to match on.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation())
        release_orphaned_reservations(mongo)
        mongo.resource_repo.release.assert_not_called()
        assert "restart" in mongo.resource_repo.force_release.call_args.args[1]

    def test_one_failure_does_not_stop_the_rest(self):
        mongo = FakeMongo(
            occupied=[resource(RES_1, "cooja1"), resource(RES_2, "cooja2")],
            simulation=simulation(),
        )
        mongo.resource_repo.force_release.side_effect = [RuntimeError("boom"), True]
        # A single bad document must not leave the remaining containers stuck.
        assert release_orphaned_reservations(mongo) == 1


class TestInterruptedSimulations:
    def test_a_running_simulation_is_marked_failed(self):
        # Cooja is launched attached to its SSH channel, so when the
        # master-node died the JVM got SIGHUP. The run really is over.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        release_orphaned_reservations(mongo)
        mongo.simulation_repo.mark_error.assert_called_once_with(SIM_1, RECOVERY_MESSAGE)

    def test_a_finished_simulation_is_left_alone(self):
        # The run completed and only the release was lost; there is nothing to
        # correct, and overwriting its result would destroy the measurement.
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Done"))
        release_orphaned_reservations(mongo)
        mongo.simulation_repo.mark_error.assert_not_called()
        mongo.resource_repo.force_release.assert_called_once()

    def test_a_failed_simulation_is_left_alone(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Error"))
        release_orphaned_reservations(mongo)
        mongo.simulation_repo.mark_error.assert_not_called()

    def test_a_container_holding_nothing_is_still_freed(self):
        mongo = FakeMongo(occupied=[resource(simulation_id=None)])
        assert release_orphaned_reservations(mongo) == 1
        mongo.simulation_repo.get.assert_not_called()

    def test_a_deleted_simulation_does_not_block_recovery(self):
        mongo = FakeMongo(occupied=[resource()], simulation=None)
        assert release_orphaned_reservations(mongo) == 1
        mongo.simulation_repo.mark_error.assert_not_called()


class TestGenerationClosing:
    def test_the_generation_is_re_evaluated(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        release_orphaned_reservations(mongo)
        mongo.generation_repo.mark_error.assert_called_once_with(GEN_1)

    def test_a_generation_with_work_left_is_untouched(self):
        mongo = FakeMongo(occupied=[resource()], simulation=simulation("Running"))
        mongo.generation_repo.any_simulation_active.return_value = True
        release_orphaned_reservations(mongo)
        mongo.generation_repo.mark_error.assert_not_called()
        mongo.generation_repo.mark_done.assert_not_called()

    def test_a_simulation_without_a_generation_is_fine(self):
        mongo = FakeMongo(
            occupied=[resource()], simulation=simulation("Running", generation_id=None))
        assert release_orphaned_reservations(mongo) == 1
        mongo.generation_repo.any_simulation_active.assert_not_called()
