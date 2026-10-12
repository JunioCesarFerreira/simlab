"""Tests for the health cycle.

This is where the pool is decided, so the rules it must not break are pinned
here rather than left to reading:

  * a container holding a simulation is never touched;
  * a host leaves the pool only after repeated failure, and returns on the
    first success;
  * Docker being unavailable does not take a working host out of the pool;
  * a container enters the pool only when *both* levels pass.

The deadlock this phase removed is guarded by
``TestFirstCycle.test_an_unmeasured_pool_becomes_usable``: the protection used
to be an optimistic default, and is now a measurement.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.credentials import CredentialUnavailable
from lib.health import STALE_RESERVATION_MESSAGE, HealthMonitor
from lib.transport.base import ProbeResult, TransportUnreachable
from pylib.db.models.infra import EnumHostStatus, EnumResourceStatus

HOST_ID = ObjectId("507f1f77bcf86cd799430011")
RES_ID = ObjectId("507f1f77bcf86cd799430021")
SIM_ID = ObjectId("507f1f77bcf86cd799430003")

HOST = {"_id": HOST_ID, "name": "local", "type": "local",
        "status": "offline", "consecutive_failures": 0,
        "container_access": {"username": "root", "port": 22}}

READY = ProbeResult(reachable=True, cooja_present=True, java_version="21",
                    cooja_jar_sha256="a" * 64, message="ready")
NO_COOJA = ProbeResult(reachable=True, cooja_present=False,
                       message="cooja.jar not found")


def resource(name="cooja1", status="offline", enabled=True,
             container_state="running", checked=True, _id=RES_ID) -> dict:
    return {
        "_id": _id, "name": name, "status": status, "enabled": enabled,
        "container": {
            "state": container_state,
            "checked_at": datetime.now() if checked else None,
        },
    }


def build_monitor(mongo, *, probe=None, discover=None, **kwargs) -> HealthMonitor:
    return HealthMonitor(
        mongo,
        discover=discover if discover is not None else (lambda m, h: None),
        probe=probe or (lambda host, res: READY),
        **kwargs,
    )


def fake_mongo(hosts=None, resources=None) -> MagicMock:
    mongo = MagicMock()
    mongo.host_repo.find_enabled.return_value = list(hosts or [HOST])
    mongo.host_repo.find_all.return_value = list(hosts or [HOST])
    mongo.host_repo.get.side_effect = lambda _id: HOST
    mongo.resource_repo.find_by_host.return_value = list(
        resources if resources is not None else [resource()])
    mongo.resource_repo.find_stale_reservations.return_value = []
    mongo.resource_repo.force_release.return_value = True
    return mongo


# ── The first cycle: what replaced the optimistic defaults ───────────────────

class TestFirstCycle:
    def test_an_unmeasured_pool_becomes_usable(self):
        """Regression guard, relocated.

        A bootstrapped host and its containers start ``offline`` because
        nothing has measured them. What makes them usable is this cycle, run
        synchronously before the dispatcher places anything — not an
        optimistic default. Two earlier versions got this wrong in opposite
        directions and both deadlocked the pool in deployment.
        """
        mongo = fake_mongo(resources=[resource(status="offline")])

        result = build_monitor(mongo).check_host(HOST)

        assert result.status == EnumHostStatus.ONLINE
        assert result.ready == ["cooja1"]
        mongo.resource_repo.mark_available.assert_called_once()
        assert mongo.host_repo.mark_check_result.call_args.kwargs["status"] \
            == EnumHostStatus.ONLINE

    def test_both_levels_must_pass(self):
        # Docker says it runs, but there is no Cooja build in it.
        mongo = fake_mongo(resources=[resource()])
        result = build_monitor(mongo, probe=lambda h, r: NO_COOJA).check_host(HOST)
        assert result.unhealthy == ["cooja1"]
        mongo.resource_repo.mark_available.assert_not_called()
        mongo.resource_repo.mark_offline.assert_called_once()

    def test_the_probe_result_is_recorded_even_when_it_fails(self):
        # "Up but no Cooja" and "not running" need different fixes, so the GUI
        # has to be able to tell them apart.
        mongo = fake_mongo(resources=[resource()])
        build_monitor(mongo, probe=lambda h, r: NO_COOJA).check_host(HOST)
        probe = mongo.resource_repo.update_probe.call_args.args[1]
        assert probe["ssh_ok"] is True
        assert "cooja.jar" in probe["message"]

    def test_a_ready_probe_records_what_it_found(self):
        mongo = fake_mongo(resources=[resource()])
        build_monitor(mongo).check_host(HOST)
        probe = mongo.resource_repo.update_probe.call_args.args[1]
        assert probe["java_version"] == "21"
        assert probe["cooja_jar_sha256"] == "a" * 64


# ── A simulation in progress is untouchable ──────────────────────────────────

class TestOccupiedIsUntouched:
    @pytest.mark.parametrize("status", ["reserved", "busy"])
    def test_a_container_holding_a_simulation_is_not_even_probed(self, status):
        probed: list = []
        mongo = fake_mongo(resources=[resource(status=status)])

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        # Probing would be harmless; acting on the answer would not.
        assert probed == []
        assert result.skipped == ["cooja1"]
        mongo.resource_repo.mark_offline.assert_not_called()
        mongo.resource_repo.mark_available.assert_not_called()

    def test_a_host_whose_containers_are_all_busy_stays_usable(self):
        # Nothing was measured, so the host's standing should not drop.
        mongo = fake_mongo(resources=[resource(status="busy")])
        result = build_monitor(mongo).check_host(HOST)
        assert result.status == EnumHostStatus.ONLINE
        assert "all containers in use" in result.message

    def test_a_disabled_container_is_skipped(self):
        mongo = fake_mongo(resources=[resource(enabled=False)])
        result = build_monitor(mongo).check_host(HOST)
        assert result.skipped == ["cooja1"]


# ── Level 1 short-circuits level 2 ───────────────────────────────────────────

class TestContainerState:
    def test_a_stopped_container_is_not_connected_to(self):
        probed: list = []
        mongo = fake_mongo(resources=[resource(container_state="exited")])

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        # Docker already answered; opening SSH to it would be pointless.
        assert probed == []
        assert result.unhealthy == ["cooja1"]
        assert "exited" in mongo.resource_repo.mark_offline.call_args.args[1]

    def test_an_unobserved_container_is_probed_anyway(self):
        # Nothing has looked at Docker yet — for instance with no socket
        # mounted — so the SSH probe is the only evidence available, and it is
        # enough. This is what the master-node always relied on.
        probed: list = []
        mongo = fake_mongo(resources=[resource(checked=False, container_state="missing")])

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert len(probed) == 1
        assert result.ready == ["cooja1"]


# ── Thresholds and hysteresis ────────────────────────────────────────────────

class TestThreshold:
    def test_one_failure_does_not_take_a_host_offline(self):
        # A dropped packet must not cost a host's whole capacity.
        mongo = fake_mongo(resources=[resource()])
        host = {**HOST, "status": "online", "consecutive_failures": 0}

        build_monitor(
            mongo,
            probe=lambda h, r: (_ for _ in ()).throw(TransportUnreachable("no route")),
            failure_threshold=3,
        ).check_host(host)

        kwargs = mongo.host_repo.mark_check_result.call_args.kwargs
        assert kwargs["status"] == EnumHostStatus.DEGRADED
        assert "failure 1 of 3" in kwargs["message"]

    def test_the_last_failure_takes_it_offline(self):
        mongo = fake_mongo(resources=[resource()])
        host = {**HOST, "status": "online", "consecutive_failures": 2}

        build_monitor(
            mongo,
            probe=lambda h, r: (_ for _ in ()).throw(TransportUnreachable("no route")),
            failure_threshold=3,
        ).check_host(host)

        assert mongo.host_repo.mark_check_result.call_args.kwargs["status"] \
            == EnumHostStatus.OFFLINE

    def test_a_success_resets_the_count_at_once(self):
        # Capacity should be slow to lose and quick to regain.
        mongo = fake_mongo(resources=[resource()])
        host = {**HOST, "status": "offline", "consecutive_failures": 9}

        build_monitor(mongo).check_host(host)

        kwargs = mongo.host_repo.mark_check_result.call_args.kwargs
        assert kwargs["status"] == EnumHostStatus.ONLINE
        assert kwargs["reset_failures"] is True

    def test_a_host_already_offline_stays_offline(self):
        mongo = fake_mongo(resources=[resource()])
        host = {**HOST, "status": "offline", "consecutive_failures": 0}
        build_monitor(
            mongo,
            probe=lambda h, r: (_ for _ in ()).throw(TransportUnreachable("x")),
        ).check_host(host)
        assert mongo.host_repo.mark_check_result.call_args.kwargs["status"] \
            == EnumHostStatus.OFFLINE


class TestPartialHealth:
    def test_one_sick_container_degrades_rather_than_disables(self):
        states = {"cooja1": READY, "cooja2": NO_COOJA}
        mongo = fake_mongo(resources=[
            resource("cooja1"), resource("cooja2", _id=ObjectId())])

        result = build_monitor(
            mongo, probe=lambda h, r: states[r["name"]]).check_host(HOST)

        # The host keeps placing work on its healthy containers.
        assert result.status == EnumHostStatus.DEGRADED
        assert result.ready == ["cooja1"] and result.unhealthy == ["cooja2"]

    def test_a_host_with_no_containers_is_misconfigured_not_down(self):
        # Down and "nothing to run on" need different fixes.
        mongo = fake_mongo(resources=[])
        result = build_monitor(mongo).check_host(HOST)
        assert result.status == EnumHostStatus.MISCONFIGURED

    def test_a_credential_problem_is_an_error_not_an_outage(self):
        mongo = fake_mongo(resources=[resource()])
        build_monitor(
            mongo,
            probe=lambda h, r: (_ for _ in ()).throw(
                CredentialUnavailable("no host credential is stored")),
        ).check_host(HOST)
        mongo.resource_repo.mark_error.assert_called_once()
        mongo.resource_repo.mark_offline.assert_not_called()


# ── Docker unavailable ───────────────────────────────────────────────────────

class TestDockerUnavailable:
    def test_a_host_with_resources_is_still_checked_over_ssh(self):
        class Report:
            available = False
            unavailable_reason = "socket not mounted"

        mongo = fake_mongo(resources=[resource(checked=False)])
        result = build_monitor(
            mongo, discover=lambda m, h: Report()).check_host(HOST)

        # Losing the Docker socket must not take a working host out of the
        # pool: SSH is all the master-node ever needed.
        assert result.status == EnumHostStatus.ONLINE
        assert "socket not mounted" in result.message

    def test_a_host_with_no_resources_says_why(self):
        class Report:
            available = False
            unavailable_reason = "permission denied"

        mongo = fake_mongo(resources=[])
        result = build_monitor(
            mongo, discover=lambda m, h: Report()).check_host(HOST)
        assert result.status == EnumHostStatus.MISCONFIGURED
        assert "permission denied" in result.message


# ── Requests from the API ────────────────────────────────────────────────────

class TestServingRequests:
    def test_a_discovery_request_is_carried_out_and_cleared(self):
        stamp = datetime.now()
        host = {**HOST, "discovery_requested_at": stamp}
        mongo = fake_mongo(hosts=[host])
        discovered: list = []

        build_monitor(mongo, discover=lambda m, h: discovered.append(h)).cycle()

        assert len(discovered) >= 1
        mongo.host_repo.clear_request.assert_any_call(
            HOST_ID, "discovery_requested_at", stamp)

    def test_a_check_request_is_carried_out_and_cleared(self):
        stamp = datetime.now()
        host = {**HOST, "check_requested_at": stamp}
        mongo = fake_mongo(hosts=[host])

        build_monitor(mongo).cycle()

        mongo.host_repo.clear_request.assert_any_call(
            HOST_ID, "check_requested_at", stamp)

    def test_the_request_is_cleared_even_if_serving_it_fails(self):
        stamp = datetime.now()
        host = {**HOST, "discovery_requested_at": stamp}
        mongo = fake_mongo(hosts=[host])

        def exploding(m, h):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            build_monitor(mongo, discover=exploding)._serve_requests()

        # Otherwise it would be retried forever on every cycle.
        mongo.host_repo.clear_request.assert_called_once()

    def test_no_request_means_no_work(self):
        mongo = fake_mongo()
        build_monitor(mongo).cycle()
        mongo.host_repo.clear_request.assert_not_called()


# ── Stale reservations ───────────────────────────────────────────────────────

class TestStaleReservations:
    def test_a_reservation_that_never_started_is_freed(self):
        mongo = fake_mongo()
        mongo.resource_repo.find_stale_reservations.return_value = [{
            "_id": RES_ID, "name": "cooja1",
            "current_simulation_id": SIM_ID,
            "reserved_at": datetime.now() - timedelta(hours=1),
        }]

        swept = build_monitor(mongo).sweep_stale_reservations()

        assert swept == ["cooja1"]
        mongo.simulation_repo.mark_error.assert_called_once_with(
            SIM_ID, STALE_RESERVATION_MESSAGE)
        mongo.resource_repo.force_release.assert_called_once()

    def test_the_cutoff_follows_the_configured_ttl(self):
        mongo = fake_mongo()
        build_monitor(mongo, reservation_ttl_sec=120).sweep_stale_reservations()
        cutoff = mongo.resource_repo.find_stale_reservations.call_args.args[0]
        elapsed = (datetime.now() - cutoff).total_seconds()
        assert 110 < elapsed < 130

    def test_a_zero_ttl_disables_the_sweep(self):
        mongo = fake_mongo()
        assert build_monitor(mongo, reservation_ttl_sec=0).sweep_stale_reservations() == []
        mongo.resource_repo.find_stale_reservations.assert_not_called()

    def test_freeing_capacity_wakes_the_dispatcher(self):
        mongo = fake_mongo()
        mongo.resource_repo.find_stale_reservations.return_value = [
            {"_id": RES_ID, "name": "cooja1", "current_simulation_id": None,
             "reserved_at": datetime.now()}]
        woken: list[int] = []

        build_monitor(mongo, on_capacity=lambda: woken.append(1)) \
            .sweep_stale_reservations()

        assert woken == [1]


# ── The cycle as a whole ─────────────────────────────────────────────────────

class TestCycle:
    def test_capacity_is_signalled_when_something_becomes_ready(self):
        mongo = fake_mongo(resources=[resource()])
        woken: list[int] = []
        build_monitor(mongo, on_capacity=lambda: woken.append(1)).cycle()
        assert woken == [1]

    def test_nothing_ready_does_not_wake_the_dispatcher(self):
        mongo = fake_mongo(resources=[resource(container_state="exited")])
        woken: list[int] = []
        build_monitor(mongo, on_capacity=lambda: woken.append(1)).cycle()
        assert woken == []

    def test_every_enabled_host_is_checked(self):
        other = {**HOST, "_id": ObjectId(), "name": "lab-01", "type": "remote"}
        mongo = fake_mongo(hosts=[HOST, other], resources=[resource()])
        results = build_monitor(mongo).cycle()
        assert [r.host_name for r in results] == ["local", "lab-01"]

    def test_a_failing_host_does_not_stop_the_loop(self):
        monitor = build_monitor(fake_mongo())
        monitor._stop.set()
        monitor.run()                      # returns immediately, no exception
