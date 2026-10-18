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
from lib.scheduler.policy import (
    FastestThenLeastLoadedPolicy,
    HostLoad,
    LeastLoadedHostPolicy,
)

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
            else lambda rid, sim_oid, attempt, **_: {
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
        policy=kwargs.pop("policy", None) or LeastLoadedHostPolicy(),
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

        def reserve(resource_id, sim_oid, attempt, **_):
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

    def test_an_assignment_that_matched_nothing_is_not_a_success(self):
        """The write can succeed and still have changed nothing.

        ``set_assignment`` returns false when no simulation matched — it was
        deleted, or the experiment was removed. The return value was ignored,
        so the run started anyway, with its result having nowhere to land and
        the container held by an attempt no document knows about.
        """
        mongo = FakeMongo()
        mongo.simulation_repo.set_assignment.return_value = False
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False

        mongo.resource_repo.release.assert_called_once()
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

class TestTheJobIsNeverDropped:
    """A transient failure must not cost a simulation.

    ``_next_job`` takes the id off the queue and calls ``task_done``. Until it
    is held, nothing owns it — so an exception while placing left the
    simulation Waiting in the database with an empty queue and nothing that
    would ever look at it again. The capacity tick only retries work it
    already knows about, so the experiment stalled until someone restarted the
    master-node. Reproduced in review: one failing ``simulation_repo.get()``
    and ``pending`` went to zero after a single ``submit``.
    """

    def test_a_transient_failure_keeps_the_job(self):
        mongo = FakeMongo()
        mongo.simulation_repo.get.side_effect = RuntimeError("mongo blinked")
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False

        assert dispatcher.pending == 1

    def test_the_same_job_is_tried_again(self):
        mongo = FakeMongo()
        mongo.simulation_repo.get.side_effect = [
            RuntimeError("mongo blinked"), waiting_sim()]
        dispatcher, started = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False      # failed, held
        assert dispatcher.step() is True       # same id, now placed

        assert [entry["args"][0]["_id"] for entry in started] == [SIM_OID]

    def test_a_job_that_keeps_failing_stops_blocking_the_others(self):
        """Held is not the same as held for ever."""
        mongo = FakeMongo()
        mongo.simulation_repo.get.side_effect = RuntimeError("always")
        dispatcher, _ = make_dispatcher(mongo, placement_attempts=2)
        dispatcher.submit(SIM_ID)

        dispatcher.step()
        assert dispatcher._held == SIM_ID
        dispatcher.step()

        # Still ours — at the back of the queue, not ahead of work that might
        # succeed, and not thrown away.
        assert dispatcher._held is None
        assert dispatcher.pending == 1

    def test_a_successful_placement_resets_the_count(self):
        mongo = FakeMongo()
        mongo.simulation_repo.get.side_effect = [
            RuntimeError("blink"), waiting_sim()]
        dispatcher, _ = make_dispatcher(mongo, placement_attempts=2)
        dispatcher.submit(SIM_ID)

        dispatcher.step()
        dispatcher.step()

        assert dispatcher._held_failures == 0

    def test_no_capacity_still_holds_the_job(self):
        # The behaviour that already existed, kept intact by the change.
        mongo = FakeMongo(inventory=snapshot((HOST_A, "host-a", 0, 0)))
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is False
        assert dispatcher._held == SIM_ID
        assert dispatcher.pending == 1


class TestRescanningWhatTheQueueLost:
    """The database is the truth; the change stream is an optimisation."""

    def test_an_idle_step_re_offers_waiting_simulations(self):
        mongo = FakeMongo()
        mongo.simulation_repo.find_pending.return_value = [
            {"_id": SIM_OID}, {"_id": ObjectId("507f1f77bcf86cd799430004")}]
        dispatcher, _ = make_dispatcher(mongo)

        assert dispatcher.step() is False      # nothing queued

        assert dispatcher.pending == 2

    def test_it_does_not_re_read_on_every_idle_step(self):
        mongo = FakeMongo()
        mongo.simulation_repo.find_pending.return_value = []
        dispatcher, _ = make_dispatcher(mongo, rescan_interval_sec=3600)

        for _ in range(5):
            dispatcher.step()

        assert mongo.simulation_repo.find_pending.call_count == 1

    def test_it_does_not_run_while_a_job_is_held(self):
        # Holding means there is no capacity; re-reading would only add
        # duplicates behind a job that cannot be placed yet.
        mongo = FakeMongo(inventory=snapshot((HOST_A, "host-a", 0, 0)))
        mongo.simulation_repo.find_pending.return_value = [{"_id": SIM_OID}]
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        dispatcher.step()
        dispatcher.step()

        mongo.simulation_repo.find_pending.assert_not_called()

    def test_a_failing_rescan_is_not_fatal(self):
        mongo = FakeMongo()
        mongo.simulation_repo.find_pending.side_effect = RuntimeError("mongo gone")
        dispatcher, _ = make_dispatcher(mongo)

        assert dispatcher.step() is False

    def test_a_re_offered_simulation_that_no_longer_wants_placement_is_dropped(self):
        # The idempotency guard already covers a redelivery, so re-offering is
        # safe: it costs one read and one log line.
        mongo = FakeMongo()
        mongo.simulation_repo.get.return_value = waiting_sim(status="Running")
        mongo.simulation_repo.find_pending.return_value = [{"_id": SIM_OID}]
        dispatcher, started = make_dispatcher(mongo)

        dispatcher.step()                      # re-offers it
        assert dispatcher.step() is False      # and declines to place it

        assert started == []


class TestLoopRobustness:
    def test_a_failing_step_does_not_kill_the_loop(self):
        mongo = FakeMongo()
        mongo.simulation_repo.get.side_effect = RuntimeError("transient")
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)

        # run() keeps going; if it did not, nothing would ever be placed again
        # after one bad document.
        dispatcher._stop.set()
        dispatcher.run()

        # The step absorbs the failure rather than propagating it, because
        # propagating is what used to lose the job: the id had already left
        # the queue and nothing held it.
        assert dispatcher.step() is False
        assert dispatcher.pending == 1

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


# ── R04: the attempt is this process's before anyone can observe it ─────────

class TestTheAttemptIsClaimedOnThisThread:
    """Reproduces R04, from the dispatcher's side.

    The runner registered the attempt it was following inside
    ``_on_container`` — on the *runner* thread, which starts later. Between
    the reservation being written and that line being reached, the periodic
    reconciliation could read a held container, see nobody following it, and
    free it as the leftover of a dead process.

    The window is closed by construction rather than narrowed: the attempt is
    claimed on the dispatcher's own thread, between the atomic reservation and
    everything that follows it.
    """

    def _runner_with_a_real_registry(self):
        from lib.scheduler.runner import SimulationRunner
        return SimulationRunner(
            MagicMock(), execute=MagicMock(), prepare=MagicMock())

    def _dispatcher(self, mongo, runner):
        started: list[dict] = []

        def thread_factory(target, args, name):
            started.append({"target": target, "args": args, "name": name})
            return MagicMock()

        return Dispatcher(
            mongo, runner=runner, policy=LeastLoadedHostPolicy(),
            idle_tick_sec=0.01, thread_factory=thread_factory,
        ), started

    def test_the_attempt_is_followed_as_soon_as_step_returns(self):
        mongo = FakeMongo()
        runner = self._runner_with_a_real_registry()
        dispatcher, started = self._dispatcher(mongo, runner)
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is True

        # No runner thread has run — `thread_factory` only recorded it — and
        # the attempt is already known. That is the property: there is no
        # instant at which the reservation exists and the attempt does not.
        attempt = mongo.resource_repo.reserve.call_args.args[2]
        assert runner.is_following(attempt)
        assert len(started) == 1

    def test_the_registration_precedes_the_assignment_write(self):
        mongo = FakeMongo()
        order: list[str] = []
        runner = MagicMock()
        runner.register_attempt.side_effect = lambda a: order.append("register")
        mongo.simulation_repo.set_assignment.side_effect = (
            lambda *a, **k: order.append("assignment") or True)
        mongo.resource_repo.reserve.side_effect = (
            lambda rid, sim_oid, attempt, **_: order.append("reserve") or {
                "_id": rid, "name": "cooja1", "container": {"id": "abc"}})
        dispatcher, _ = self._dispatcher(mongo, runner)
        dispatcher.submit(SIM_ID)

        dispatcher.step()

        assert order == ["reserve", "register", "assignment"]

    def test_a_reservation_given_back_is_also_unregistered(self):
        """Otherwise an attempt nobody follows is invisible to reconciliation
        for the lifetime of the process — the opposite leak."""
        mongo = FakeMongo()
        mongo.simulation_repo.set_assignment.return_value = False
        runner = self._runner_with_a_real_registry()
        dispatcher, started = self._dispatcher(mongo, runner)
        dispatcher.submit(SIM_ID)

        dispatcher.step()

        attempt = mongo.resource_repo.reserve.call_args.args[2]
        assert not runner.is_following(attempt)
        mongo.resource_repo.release.assert_called_once()
        assert started == []

    def test_a_thread_that_cannot_start_is_also_unregistered(self):
        mongo = FakeMongo()
        runner = self._runner_with_a_real_registry()

        def refuse(target, args, name):
            raise RuntimeError("cannot start a thread")

        dispatcher = Dispatcher(
            mongo, runner=runner, policy=LeastLoadedHostPolicy(),
            idle_tick_sec=0.01, thread_factory=refuse,
        )
        dispatcher.submit(SIM_ID)

        # The step absorbs it and holds the job, as it does any placement
        # failure; what matters here is the registry.
        dispatcher.step()

        attempt = mongo.resource_repo.reserve.call_args.args[2]
        assert not runner.is_following(attempt)


# ── The policy also decides the order inside the host ────────────────────────

class TestTheOrderingInsideTheHostFollowsThePolicy:
    """Two halves of one decision.

    A speed-aware policy that picked the host finishing soonest and was then
    handed that host's *slowest* free container would undo half of its own
    choice, so the ordering is asked for by the policy rather than fixed in
    the repository.
    """

    def test_the_default_policy_does_not_ask_for_the_fastest(self):
        mongo = FakeMongo()
        dispatcher, _ = make_dispatcher(mongo)
        dispatcher.submit(SIM_ID)
        dispatcher.step()

        assert mongo.resource_repo.find_free_by_host.call_args.kwargs == {
            "prefer_fastest": False}

    def test_a_speed_aware_policy_does(self):
        mongo = FakeMongo()
        dispatcher, _ = make_dispatcher(
            mongo, policy=FastestThenLeastLoadedPolicy())
        dispatcher.submit(SIM_ID)

        assert dispatcher.step() is True
        assert mongo.resource_repo.find_free_by_host.call_args.kwargs == {
            "prefer_fastest": True}

    def test_a_policy_that_does_not_declare_it_is_treated_as_not_wanting_it(
            self):
        """A third-party policy predating the attribute must still work."""
        class Old:
            name = "old"

            def select(self, loads, rr_cursor):
                return loads[0]

        mongo = FakeMongo()
        dispatcher, _ = make_dispatcher(mongo, policy=Old())
        dispatcher.submit(SIM_ID)
        dispatcher.step()

        assert mongo.resource_repo.find_free_by_host.call_args.kwargs == {
            "prefer_fastest": False}
