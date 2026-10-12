"""Tests for the placement loop.

``step()`` is exercised one decision at a time, with no threads and no timing,
so what is asserted is the decision itself: which host was chosen, that the
container was claimed atomically, and — the part that matters most — that a
simulation with nowhere to go is *held* rather than failed or dropped.

Properties locked in:
  * a job survives a lack of capacity and is retried, not lost;
  * a redelivered change-stream event cannot place a second attempt;
  * the assignment is persisted before anything runs;
  * a reservation that cannot be recorded is given back, never stranded;
  * the decision loop cannot die.
"""
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.scheduler.dispatcher import Dispatcher
from lib.scheduler.policy import HostLoad, LeastLoadedHostPolicy

SIM_ID = "507f1f77bcf86cd799430003"
SIM_OID = ObjectId(SIM_ID)
HOST_A = ObjectId("507f1f77bcf86cd799430011")
HOST_B = ObjectId("507f1f77bcf86cd799430012")
RES_1 = ObjectId("507f1f77bcf86cd799430021")
RES_2 = ObjectId("507f1f77bcf86cd799430022")


def waiting_sim(status="Waiting") -> dict:
    return {"_id": SIM_OID, "status": status, "generation_id": ObjectId()}


def snapshot(*rows) -> list[dict]:
    return [
        {"_id": host_id, "name": name, "free": free, "active": active,
         "schedulable": True}
        for host_id, name, free, active in rows
    ]


class FakeMongo:
    """Just enough of MongoRepository for the decision loop."""

    def __init__(self, *, sim=None, inventory=None, free=None, reserve=None):
        self.simulation_repo = MagicMock()
        self.host_repo = MagicMock()
        self.resource_repo = MagicMock()

        self.simulation_repo.get.return_value = sim if sim is not None else waiting_sim()
        self.simulation_repo.set_assignment.return_value = True
        self.host_repo.inventory_snapshot.return_value = (
            inventory if inventory is not None else snapshot((HOST_A, "host-a", 4, 0)))
        self.host_repo.get.side_effect = lambda host_id: {
            "_id": host_id, "name": "host-a", "type": "local"}
        self.resource_repo.find_by_simulation.return_value = None
        self.resource_repo.find_free_by_host.return_value = (
            free if free is not None else [{"_id": RES_1, "name": "cooja1"}])
        self.resource_repo.reserve.side_effect = (
            reserve if reserve is not None
            else lambda rid, sim_oid, attempt: {
                "_id": rid, "name": "cooja1", "container": {"id": "abc"}})


def make_dispatcher(mongo, **kwargs) -> tuple[Dispatcher, list]:
    """A dispatcher whose runner threads are recorded instead of started."""
    started: list[dict] = []

    def thread_factory(target, args, name):
        started.append({"target": target, "args": args, "name": name})
        return MagicMock()

    dispatcher = Dispatcher(
        mongo,
        runner=MagicMock(),
        policy=LeastLoadedHostPolicy(),
        idle_tick_sec=0.01,
        thread_factory=thread_factory,
        **kwargs,
    )
    return dispatcher, started


# ── A successful placement ───────────────────────────────────────────────────

class TestPlacement:
    def test_claims_a_container_and_starts_it(self):
        mongo = FakeMongo()
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is True

        mongo.resource_repo.reserve.assert_called_once()
        resource_id, sim_oid, attempt = mongo.resource_repo.reserve.call_args.args
        assert resource_id == RES_1 and sim_oid == SIM_OID
        assert len(attempt) == 32            # uuid4().hex, one per attempt
        assert len(started) == 1

    def test_assignment_is_persisted_before_anything_runs(self):
        mongo = FakeMongo()
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)
        dispatcher.step()

        # Written at reservation time so a master-node that restarts
        # mid-simulation can find its way back to the container.
        kwargs = mongo.simulation_repo.set_assignment.call_args.kwargs
        assert kwargs["host_id"] == HOST_A
        assert kwargs["resource_name"] == "cooja1"
        assert kwargs["container_id"] == "abc"
        assert kwargs["attempt_id"] == mongo.resource_repo.reserve.call_args.args[2]

    def test_the_runner_receives_the_whole_placement(self):
        mongo = FakeMongo()
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)
        dispatcher.step()

        sim, host, resource, attempt = started[0]["args"]
        assert sim["_id"] == SIM_OID
        assert host["_id"] == HOST_A
        assert resource["_id"] == RES_1
        assert attempt == mongo.resource_repo.reserve.call_args.args[2]

    def test_the_busier_host_is_not_chosen(self):
        mongo = FakeMongo(inventory=snapshot(
            (HOST_A, "host-a", 8, 3), (HOST_B, "host-b", 1, 0)))
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)
        dispatcher.step()

        # Absolute count, not utilization: host-b has one container free and
        # host-a has eight, yet host-b is the right answer.
        assert mongo.resource_repo.find_free_by_host.call_args.args[0] == HOST_B


# ── Losing a race for a container ────────────────────────────────────────────

class TestClaimRaces:
    def test_moves_on_to_the_next_free_container(self):
        taken = {RES_1}

        def reserve(resource_id, sim_oid, attempt):
            if resource_id in taken:
                return None     # someone claimed it between snapshot and now
            return {"_id": resource_id, "name": "cooja2", "container": {}}

        mongo = FakeMongo(
            free=[{"_id": RES_1, "name": "cooja1"}, {"_id": RES_2, "name": "cooja2"}],
            reserve=reserve,
        )
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is True
        assert mongo.resource_repo.reserve.call_count == 2
        assert started[0]["args"][2]["_id"] == RES_2

    def test_host_with_nothing_left_holds_the_job(self):
        mongo = FakeMongo(free=[])
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        assert started == []
        # Held: the next wake-up tries the same job, possibly elsewhere.
        assert dispatcher.pending == 1

    def test_host_vanished_between_snapshot_and_claim(self):
        mongo = FakeMongo()
        mongo.host_repo.get.side_effect = lambda host_id: None
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        assert started == []
        assert dispatcher.pending == 1


# ── No capacity anywhere ─────────────────────────────────────────────────────

class TestNoCapacity:
    def test_the_job_is_held_not_dropped(self):
        mongo = FakeMongo(inventory=snapshot((HOST_A, "host-a", 0, 4)))
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        assert dispatcher.pending == 1
        assert started == []

    def test_the_held_job_is_retried_before_anything_new(self):
        mongo = FakeMongo(inventory=snapshot((HOST_A, "host-a", 0, 4)))
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)
        dispatcher.submit("507f1f77bcf86cd799439999")

        dispatcher.step()                    # holds the first
        mongo.host_repo.inventory_snapshot.return_value = snapshot(
            (HOST_A, "host-a", 1, 4))        # a container comes back

        assert dispatcher.step() is True
        # The same simulation, not the one queued behind it.
        assert started[0]["args"][0]["_id"] == SIM_OID
        assert dispatcher.pending == 1       # the second is still waiting

    def test_an_empty_queue_is_not_an_error(self):
        dispatcher, started = make_dispatcher(FakeMongo())
        assert dispatcher.step() is False
        assert started == []

    def test_no_registered_hosts_holds_rather_than_fails(self):
        mongo = FakeMongo(inventory=[])
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        # Nothing is marked as an error: the hosts may be registered a second
        # from now, and the simulation is still perfectly runnable.
        mongo.simulation_repo.mark_error.assert_not_called()
        assert dispatcher.pending == 1


# ── Simulations that must not be placed ──────────────────────────────────────

class TestSkipped:
    def test_a_missing_simulation_is_dropped(self):
        mongo = FakeMongo()
        mongo.simulation_repo.get.return_value = None
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        assert dispatcher.pending == 0       # dropped, not held
        assert started == []

    @pytest.mark.parametrize("status", ["Running", "Done", "Error", "Cancelled"])
    def test_a_simulation_past_waiting_is_skipped(self, status):
        mongo = FakeMongo(sim=waiting_sim(status))
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        mongo.resource_repo.reserve.assert_not_called()
        assert started == []

    def test_a_simulation_already_holding_a_container_is_not_placed_again(self):
        # The change stream can redeliver a document. Without this check the
        # same simulation would be started twice, on two containers.
        mongo = FakeMongo()
        mongo.resource_repo.find_by_simulation.return_value = {
            "_id": RES_1, "name": "cooja1"}
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        mongo.resource_repo.reserve.assert_not_called()
        assert started == []

    def test_an_intercepted_simulation_takes_no_container(self):
        # A synthetic run is evaluated analytically; holding a container for it
        # would take capacity from work that needs Cooja.
        mongo = FakeMongo()
        dispatcher, started = make_dispatcher(mongo, intercept=lambda sim: True)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        mongo.resource_repo.reserve.assert_not_called()
        assert dispatcher.pending == 0
        assert started == []

    def test_a_declining_intercept_lets_the_placement_proceed(self):
        mongo = FakeMongo()
        dispatcher, started = make_dispatcher(mongo, intercept=lambda sim: False)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is True
        assert len(started) == 1


# ── A reservation must never be stranded ─────────────────────────────────────

class TestReservationIsNeverStranded:
    def test_a_failed_assignment_gives_the_container_back(self):
        mongo = FakeMongo()
        mongo.simulation_repo.set_assignment.side_effect = RuntimeError("mongo down")
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False

        # The reservation stands but the simulation does not know about it,
        # which would hold the container forever. It is released instead.
        mongo.resource_repo.release.assert_called_once()
        resource_id, sim_oid, attempt = mongo.resource_repo.release.call_args.args
        assert resource_id == RES_1 and sim_oid == SIM_OID
        assert attempt == mongo.resource_repo.reserve.call_args.args[2]
        assert started == []

    def test_each_attempt_gets_its_own_identifier(self):
        mongo = FakeMongo(
            free=[{"_id": RES_1, "name": "cooja1"}, {"_id": RES_2, "name": "cooja2"}])
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)
        dispatcher.step()
        dispatcher.submit(SIM_ID)
        dispatcher.step()

        attempts = [c.args[2] for c in mongo.resource_repo.reserve.call_args_list]
        assert len(set(attempts)) == len(attempts)


# ── The loop itself ──────────────────────────────────────────────────────────

class TestLoopRobustness:
    def test_a_failing_step_does_not_kill_the_loop(self):
        mongo = FakeMongo()
        mongo.simulation_repo.get.side_effect = RuntimeError("transient")
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        # run() swallows the failure; if it did not, nothing would ever be
        # placed again after one bad document.
        dispatcher._stop.set()
        dispatcher.run()

        with pytest.raises(RuntimeError):
            dispatcher.step()

    def test_notify_capacity_is_safe_with_nobody_waiting(self):
        dispatcher, _ = make_dispatcher(FakeMongo())
        dispatcher.notify_capacity()

    def test_stop_wakes_the_loop(self):
        dispatcher, _ = make_dispatcher(FakeMongo())
        dispatcher.stop()
        dispatcher.run()                     # returns immediately

    def test_the_cursor_advances_only_on_a_placement(self):
        mongo = FakeMongo(inventory=snapshot(
            (HOST_A, "host-a", 1, 0), (HOST_B, "host-b", 1, 0)))
        dispatcher, _ = make_dispatcher(mongo)

        dispatcher.submit(SIM_ID)
        dispatcher.step()
        first = mongo.resource_repo.find_free_by_host.call_args.args[0]

        mongo.resource_repo.find_by_simulation.return_value = {"name": "cooja1"}
        dispatcher.submit(SIM_ID)
        dispatcher.step()                    # skipped, so no rotation

        mongo.resource_repo.find_by_simulation.return_value = None
        dispatcher.submit(SIM_ID)
        dispatcher.step()
        second = mongo.resource_repo.find_free_by_host.call_args.args[0]

        # One placement happened between the two, so the tie rotated exactly
        # once: a skip must not consume a turn.
        assert first != second
