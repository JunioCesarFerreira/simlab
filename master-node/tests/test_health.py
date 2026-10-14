"""Tests for the health cycle.

This is where the pool is decided, so the rules it must not break are pinned
here rather than left to reading:

  * a container holding a simulation is never touched;
  * a host leaves the pool only after repeated failure, and returns on the
    first success;
  * Docker being unavailable does not take a working host out of the pool;
  * a container enters the pool only when *both* levels pass;
  * a container already in the pool is not re-probed every cycle, and nothing
    but good news earns that silence.

The deadlock this phase removed is guarded by
``TestFirstCycle.test_an_unmeasured_pool_becomes_usable``: the protection used
to be an optimistic default, and is now a measurement.
"""
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from lib.credentials import CredentialUnavailable
from lib.health import (
    DEFAULT_PROBE_TTL_SEC,
    STALE_RESERVATION_MESSAGE,
    HealthMonitor,
    HostHealth,
    probe_is_fresh,
)
from lib.transport.base import ProbeResult, TransportError, TransportUnreachable
from lib.transport.hostkey import HostKeyMismatch
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
             container_state="running", checked=True, _id=RES_ID,
             connection=None, origin="discovered", probe=None) -> dict:
    return {
        "_id": _id, "name": name, "status": status, "enabled": enabled,
        "origin": origin,
        "connection": connection or {},
        "probe": probe or {},
        "container": {
            "state": container_state,
            "checked_at": datetime.now() if checked else None,
        },
    }


def ready_probe(age_sec: float = 0.0) -> dict:
    """A stored probe that found the worker ready, ``age_sec`` ago."""
    return {
        "ssh_ok": True,
        "cooja_jar_sha256": "a" * 64,
        "java_version": "21",
        "message": "ready",
        "checked_at": datetime.now() - timedelta(seconds=age_sec),
    }


def in_pool(probe=None, **kwargs) -> dict:
    """A worker the last cycle left in the pool."""
    return resource(status=EnumResourceStatus.AVAILABLE.value,
                    probe=probe if probe is not None else ready_probe(), **kwargs)


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


class TestOperatorConfiguredWorkers:
    """Regression: Docker's silence overruled an address a person typed.

    Observed in deployment. Two workers were configured by hand with their
    own addresses (`localhost:2222`, `localhost:2223`) and their own sealed
    credentials, on a host left in `docker` mode whose SSH user cannot reach
    the daemon. Discovery found nothing and marked both `missing`, and the
    level-1 short-circuit then skipped the SSH probe entirely — so two
    perfectly reachable workers sat `offline` with "container is missing" and
    never got a single connection attempt.

    Where the operator stated the address, the probe is the whole of the
    evidence.
    """

    def test_a_worker_with_its_own_connection_is_probed_anyway(self):
        probed: list = []
        mongo = fake_mongo(resources=[resource(
            container_state="missing",
            connection={"transport": "ssh", "port": 2222, "username": "root"},
        )])

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert len(probed) == 1
        assert result.ready == ["cooja1"]
        mongo.resource_repo.mark_available.assert_called_once()

    def test_a_manually_registered_worker_is_probed_anyway(self):
        probed: list = []
        mongo = fake_mongo(resources=[resource(
            container_state="missing", origin="manual")])

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY) \
            .check_host(HOST)

        assert len(probed) == 1

    def test_a_host_listing_workers_manually_probes_all_of_them(self):
        probed: list = []
        host = {**HOST, "discovery": {"mode": "manual"}}
        mongo = fake_mongo(resources=[resource(container_state="missing")])

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY) \
            .check_host(host)

        assert len(probed) == 1

    def test_an_explicit_address_mode_probes_all_of_them(self):
        probed: list = []
        host = {**HOST, "container_access": {"username": "root", "port": 22,
                                             "address_mode": "explicit"}}
        mongo = fake_mongo(resources=[resource(container_state="missing")])

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY) \
            .check_host(host)

        assert len(probed) == 1

    def test_a_purely_discovered_worker_still_short_circuits(self):
        # The optimization is kept where it is sound: Docker saw the container
        # and it is down, and nobody claimed otherwise. Probing every stopped
        # container would cost one connection timeout each, per cycle.
        probed: list = []
        mongo = fake_mongo(resources=[resource(
            container_state="exited", origin="discovered")])

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert probed == []
        assert result.unhealthy == ["cooja1"]

    def test_transport_alone_does_not_count_as_configured(self):
        # The mapper always writes `transport`, so it is not evidence that a
        # person set anything.
        probed: list = []
        mongo = fake_mongo(resources=[resource(
            container_state="exited", connection={"transport": "ssh"})])

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY) \
            .check_host(HOST)

        assert probed == []


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

    def test_a_manual_worker_is_freed_without_dockers_veto(self):
        """The sweep reads the host so the release applies the same rule.

        Without it a hand-configured worker would be freed straight back to
        `offline` on a stale `missing`, which is the defect the release path
        carried on the live installation.
        """
        host = {**HOST, "discovery": {"mode": "manual"}}
        mongo = fake_mongo(hosts=[host])
        mongo.host_repo.get.side_effect = lambda _id: host
        mongo.resource_repo.find_stale_reservations.return_value = [
            {"_id": RES_ID, "name": "cooja1", "host_id": HOST_ID,
             "current_simulation_id": None, "reserved_at": datetime.now()}]

        build_monitor(mongo).sweep_stale_reservations()

        kwargs = mongo.resource_repo.force_release.call_args.kwargs
        assert kwargs["container_state_counts"] is False

    def test_a_discovered_worker_keeps_dockers_veto(self):
        mongo = fake_mongo()
        mongo.resource_repo.find_stale_reservations.return_value = [
            {"_id": RES_ID, "name": "cooja1", "host_id": HOST_ID,
             "origin": "discovered", "current_simulation_id": None,
             "reserved_at": datetime.now()}]

        build_monitor(mongo).sweep_stale_reservations()

        assert mongo.resource_repo.force_release.call_args.kwargs[
            "container_state_counts"] is True

    def test_freeing_capacity_wakes_the_dispatcher(self):
        mongo = fake_mongo()
        mongo.resource_repo.find_stale_reservations.return_value = [
            {"_id": RES_ID, "name": "cooja1", "current_simulation_id": None,
             "reserved_at": datetime.now()}]
        woken: list[int] = []

        build_monitor(mongo, on_capacity=lambda: woken.append(1)) \
            .sweep_stale_reservations()

        assert woken == [1]


# ── Not asking a question already answered ───────────────────────────────────

class TestProbeFreshness:
    """A probe is a two-hop SSH connection; repeating it is not free.

    A pilot measured 3078 transport opens for 344 simulations and 41 opens in
    three minutes while idle, until the host's sshd began refusing them. What
    makes it safe to skip is the asymmetry: only a worker *in the pool* whose
    last probe found it *ready* is taken at its word.
    """

    def test_a_worker_that_just_answered_is_not_asked_again(self):
        mongo = fake_mongo(resources=[in_pool()])
        probed: list[dict] = []

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert probed == []
        assert result.ready == ["cooja1"] and result.fresh == ["cooja1"]
        assert result.status == EnumHostStatus.ONLINE

    def test_skipping_writes_nothing(self):
        """Rewriting the probe would make its timestamp fresh forever."""
        mongo = fake_mongo(resources=[in_pool()])

        build_monitor(mongo, probe=lambda h, r: READY).check_host(HOST)

        mongo.resource_repo.update_probe.assert_not_called()
        mongo.resource_repo.mark_available.assert_not_called()

    def test_an_old_answer_is_measured_again(self):
        mongo = fake_mongo(
            resources=[in_pool(probe=ready_probe(DEFAULT_PROBE_TTL_SEC + 1))])
        probed: list[dict] = []

        build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_a_failed_answer_is_never_inherited(self):
        # Recovery must keep happening at the cycle interval.
        failed = {"ssh_ok": False, "message": "no route", "checked_at": datetime.now()}
        mongo = fake_mongo(resources=[in_pool(probe=failed)])
        probed: list[dict] = []

        build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_a_worker_outside_the_pool_is_measured(self):
        # Offline with a recent good probe: something took it out, and only a
        # measurement may put it back.
        mongo = fake_mongo(resources=[resource(status="offline", probe=ready_probe())])
        probed: list[dict] = []

        build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_a_stopped_container_still_leaves_the_pool(self):
        """Level 1 is read before freshness, so a negative observation wins."""
        mongo = fake_mongo(resources=[in_pool(container_state="exited")])
        probed: list[dict] = []

        result = build_monitor(
            mongo, probe=lambda h, r: probed.append(r) or READY).check_host(HOST)

        assert probed == []
        assert result.unhealthy == ["cooja1"] and result.fresh == []
        mongo.resource_repo.mark_offline.assert_called_once()

    def test_an_operator_asking_for_a_check_gets_a_real_one(self):
        mongo = fake_mongo(resources=[in_pool()])
        probed: list[dict] = []

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY) \
            .check_host(HOST, force=True)

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_the_check_button_forces(self):
        host = {**HOST, "check_requested_at": datetime.now()}
        mongo = fake_mongo(hosts=[host], resources=[in_pool()])
        mongo.host_repo.get.side_effect = lambda _id: host
        probed: list[dict] = []

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY).cycle()

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_the_first_cycle_of_a_process_measures(self):
        """A predecessor's probe describes a pool this process has not seen."""
        mongo = fake_mongo(resources=[in_pool()])
        probed: list[dict] = []

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY) \
            .cycle(force=True)

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_later_cycles_do_not(self):
        mongo = fake_mongo(resources=[in_pool()])
        probed: list[dict] = []

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY).cycle()

        assert probed == []

    def test_a_zero_ttl_restores_probing_every_cycle(self):
        mongo = fake_mongo(resources=[in_pool()])
        probed: list[dict] = []

        build_monitor(mongo, probe=lambda h, r: probed.append(r) or READY,
                      probe_ttl_sec=0).check_host(HOST)

        assert [r["name"] for r in probed] == ["cooja1"]

    def test_the_log_line_says_what_was_measured(self):
        mongo = fake_mongo(resources=[in_pool(), resource(name="cooja2", _id=ObjectId())])

        result = build_monitor(mongo).check_host(HOST)

        assert result.ready == ["cooja1", "cooja2"]
        assert "1 unchanged" in result.describe()

    def test_an_all_fresh_host_still_signals_capacity(self):
        # Nothing was measured, but the capacity is real and the dispatcher
        # must not be left waiting for a probe that will not happen.
        mongo = fake_mongo(resources=[in_pool()])
        woken: list[int] = []

        build_monitor(mongo, on_capacity=lambda: woken.append(1)).cycle()

        assert woken == [1]


class TestProbeIsFresh:
    """The predicate on its own, where the asymmetry is easiest to read."""

    def test_ready_and_recent(self):
        assert probe_is_fresh(in_pool(probe=ready_probe(10)), 300) is True

    def test_too_old(self):
        assert probe_is_fresh(in_pool(probe=ready_probe(301)), 300) is False

    def test_reachable_without_a_cooja_build_is_not_ready(self):
        probe = {**ready_probe(), "cooja_jar_sha256": None}
        assert probe_is_fresh(in_pool(probe=probe), 300) is False

    def test_never_probed(self):
        assert probe_is_fresh(in_pool(probe={}), 300) is False

    def test_a_cleared_probe_is_not_fresh(self):
        """What reconfiguring a worker leaves behind, so it gets measured."""
        assert probe_is_fresh(in_pool(probe={"checked_at": None}), 300) is False

    def test_a_timestamp_in_the_future_is_not_trusted(self):
        # A clock that moved would otherwise grant indefinite silence.
        assert probe_is_fresh(in_pool(probe=ready_probe(-600)), 300) is False

    def test_busy_is_not_the_same_as_available(self):
        busy = in_pool()
        busy["status"] = EnumResourceStatus.BUSY.value
        assert probe_is_fresh(busy, 300) is False


# ── One handshake per host per turn ──────────────────────────────────────────

class FakePool:
    """Stands in for ``JumpPool``, recording who asked for what."""

    def __init__(self, error=None):
        self.asked: list[str] = []
        self.closed: list[str] = []
        self.closed_all = 0
        self._error = error
        self.session_obj = MagicMock()

    def session(self, host):
        self.asked.append(str(host.get("name")))
        if self._error is not None:
            raise self._error
        return self.session_obj

    def close_host(self, host):
        self.closed.append(str(host.get("name")))

    def close(self):
        self.closed_all += 1


REMOTE = {**HOST, "_id": ObjectId("507f1f77bcf86cd799430012"), "name": "lab-01",
          "type": "remote",
          "access": {"address": "192.168.0.42", "port": 2222, "username": "simlab"}}


class TestSharedHostSession:
    def test_every_probe_on_a_host_rides_one_session(self):
        workers = [resource(name=f"cooja{i}", _id=ObjectId()) for i in range(1, 5)]
        mongo = fake_mongo(hosts=[REMOTE], resources=workers)
        pool = FakePool()
        opened: list[Any] = []

        monitor = HealthMonitor(
            mongo, jumps=pool, discover=lambda m, h: None,
            probe=lambda h, r: READY)
        with patch("lib.health.build_transport") as build:
            build.return_value.__enter__.return_value.probe.return_value = READY
            monitor._probe = monitor._default_probe
            monitor.check_host(REMOTE)
            opened = [c.kwargs["jump_client"] for c in build.call_args_list]

        assert len(opened) == 4
        assert all(client is pool.session_obj.client for client in opened)

    def test_the_session_is_closed_when_the_hosts_turn_ends(self):
        mongo = fake_mongo(hosts=[REMOTE], resources=[resource()])
        pool = FakePool()

        HealthMonitor(mongo, jumps=pool, discover=lambda m, h: None,
                      probe=lambda h, r: READY).check_host(REMOTE)

        assert pool.closed == ["lab-01"]

    def test_the_session_is_closed_even_when_the_check_blows_up(self):
        mongo = fake_mongo(hosts=[REMOTE], resources=[resource()])
        mongo.resource_repo.find_by_host.side_effect = RuntimeError("db gone")
        pool = FakePool()

        with pytest.raises(RuntimeError):
            HealthMonitor(mongo, jumps=pool, discover=lambda m, h: None,
                          probe=lambda h, r: READY).check_host(REMOTE)

        assert pool.closed == ["lab-01"]

    def test_discovery_uses_the_same_session(self):
        mongo = fake_mongo(hosts=[REMOTE], resources=[resource()])
        pool = FakePool()
        monitor = HealthMonitor(mongo, jumps=pool, probe=lambda h, r: READY)

        with patch("lib.health.discover_host") as discover:
            discover.return_value = None
            monitor.check_host(REMOTE)

        assert discover.call_args.kwargs["session"] is pool.session_obj

    def test_an_unreachable_host_is_not_attempted_once_per_worker(self):
        """The pool caches the failure; the cycle must not work around it."""
        workers = [resource(name=f"cooja{i}", _id=ObjectId()) for i in range(1, 4)]
        mongo = fake_mongo(hosts=[REMOTE], resources=workers)
        pool = FakePool(error=TransportUnreachable("no route to host"))
        monitor = HealthMonitor(mongo, jumps=pool)
        monitor._probe = monitor._default_probe

        result = monitor.check_host(REMOTE)

        # Docker's record says why it could not look, and each worker reports
        # the same error — all from one cached failure.
        assert result.unhealthy == ["cooja1", "cooja2", "cooja3"]
        assert mongo.host_repo.record_discovery.call_args.kwargs["count"] == 0
        assert "no route to host" in \
            mongo.host_repo.record_discovery.call_args.kwargs["message"]

    def test_a_local_host_needs_no_session(self):
        mongo = fake_mongo(resources=[resource()])
        pool = FakePool()
        pool.session_obj = None
        monitor = HealthMonitor(mongo, jumps=pool, discover=lambda m, h: None)
        monitor._probe = monitor._default_probe

        with patch("lib.health.build_transport") as build:
            build.return_value.__enter__.return_value.probe.return_value = READY
            monitor.check_host(HOST)

        assert build.call_args.kwargs["jump_client"] is None

    def test_an_on_demand_discovery_closes_what_it_opened(self):
        """The one path that opens a session outside ``check_host``."""
        host = {**REMOTE, "discovery_requested_at": datetime.now()}
        mongo = fake_mongo(hosts=[host], resources=[resource()])
        pool = FakePool()

        HealthMonitor(mongo, jumps=pool, probe=lambda h, r: READY,
                      discover=lambda m, h: None)._serve_requests()

        assert pool.closed == ["lab-01"]

    def test_stopping_releases_everything(self):
        pool = FakePool()
        HealthMonitor(fake_mongo(), jumps=pool).stop()
        assert pool.closed_all == 1


# ── One bad host must not stop the others ────────────────────────────────────

class TestFailureIsolation:
    """The pool is decided here, so a single host cannot be allowed to end it.

    `HostKeyMismatch` used to derive from `Exception` alone and nothing on
    this path expected it. Raised while checking the first host, it escaped
    the comprehension that checks them all: the remaining hosts were never
    measured, the reservation sweep never ran, and the failing host's own
    diagnosis was never written down. On the synchronous first pass it could
    stop the master-node before the dispatcher started — one rebuilt remote
    host would have taken the local pool with it.
    """

    def test_a_mismatch_is_a_configuration_fault_not_unreachability(self):
        # Waiting does not fix it, and the container is not offline: an
        # operator has to decide the host was legitimately rebuilt.
        mongo = fake_mongo(resources=[resource()])
        mismatch = HostKeyMismatch("192.168.0.42", "SHA256:old", "SHA256:new")

        result = build_monitor(
            mongo, probe=lambda h, r: (_ for _ in ()).throw(mismatch),
        ).check_host(HOST)

        assert result.unhealthy == ["cooja1"]
        mongo.resource_repo.mark_error.assert_called_once()
        mongo.resource_repo.mark_offline.assert_not_called()
        assert "pinned" in mongo.resource_repo.mark_error.call_args.args[1]

    def test_it_is_catchable_as_a_transport_error(self):
        """So no caller outside this module can leak it by accident."""
        assert issubclass(HostKeyMismatch, TransportError)

    def test_the_hosts_after_a_failing_one_are_still_checked(self):
        other = {**HOST, "_id": ObjectId(), "name": "lab-02"}
        mongo = fake_mongo(hosts=[HOST, other], resources=[resource()])
        monitor = build_monitor(mongo)
        calls: list[str] = []

        def check(host, *, discover=True, force=False):
            calls.append(str(host.get("name")))
            if host["name"] == "local":
                raise RuntimeError("something nobody expected")
            return HostHealth(host_name="lab-02", status=EnumHostStatus.ONLINE,
                              message="", ready=["cooja1"])

        monitor.check_host = check
        results = monitor.cycle()

        assert calls == ["local", "lab-02"]
        assert [r.host_name for r in results] == ["local", "lab-02"]

    def test_the_failure_is_recorded_against_its_host(self):
        mongo = fake_mongo(resources=[resource()])
        monitor = build_monitor(mongo)
        monitor.check_host = lambda host, **k: (_ for _ in ()).throw(
            RuntimeError("boom"))

        results = monitor.cycle()

        assert results[0].status == EnumHostStatus.OFFLINE
        assert "RuntimeError: boom" in results[0].message
        # An unexplained host is worse than an offline one.
        assert mongo.host_repo.mark_check_result.called

    def test_housekeeping_still_runs_after_a_host_fails(self):
        mongo = fake_mongo(resources=[resource()])
        monitor = build_monitor(mongo)
        monitor.check_host = lambda host, **k: (_ for _ in ()).throw(
            RuntimeError("boom"))

        monitor.cycle()

        # The sweep is what frees containers claimed by a dead attempt; it
        # must not be skipped because an unrelated host misbehaved.
        mongo.resource_repo.find_stale_reservations.assert_called_once()

    def test_a_failing_request_does_not_stop_the_pass(self):
        mongo = fake_mongo(resources=[resource()])
        mongo.host_repo.find_all.side_effect = RuntimeError("mongo blinked")

        results = build_monitor(mongo).cycle()

        assert [r.host_name for r in results] == ["local"]

    def test_a_failing_sweep_does_not_hide_the_results(self):
        mongo = fake_mongo(resources=[resource()])
        mongo.resource_repo.find_stale_reservations.side_effect = \
            RuntimeError("mongo blinked")

        results = build_monitor(mongo).cycle()

        assert results[0].ready == ["cooja1"]

    def test_even_a_failed_recording_returns_a_result(self):
        """The first cycle runs before the dispatcher; it must return."""
        mongo = fake_mongo(resources=[resource()])
        monitor = build_monitor(mongo)
        monitor.check_host = lambda host, **k: (_ for _ in ()).throw(
            RuntimeError("boom"))
        mongo.host_repo.mark_check_result.side_effect = RuntimeError("mongo gone")

        results = monitor.cycle(force=True)

        assert results[0].status == EnumHostStatus.OFFLINE
        assert results[0].ready == []


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
