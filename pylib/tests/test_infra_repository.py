"""Tests for the infrastructure repositories.

These lock in the invariants the scheduler depends on but cannot enforce,
because they live in the query filters:

  * a reservation is won by exactly one caller;
  * a release only works for the attempt that still holds the resource;
  * a health check or a discovery pass never evicts a running simulation;
  * a vanished container leaves the pool but keeps its history;
  * the inventory snapshot reports honest counts.

The connection is mocked in the style of ``test_chart_repository.py``: these
assert the *filters and updates sent to MongoDB*, which is where the
correctness of an atomic operation actually lives.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from pylib.db.models.infra import (
    EnumContainerState,
    EnumHostStatus,
    EnumResourceStatus,
    RESOURCES_COLLECTION,
    HOSTS_COLLECTION,
)
from pylib.db.repositories.infra import (
    CoojaResourceRepository,
    HostRepository,
    InfraCryptoRepository,
    released_status_expression,
)

HOST_ID = ObjectId("507f1f77bcf86cd799430001")
RES_ID = ObjectId("507f1f77bcf86cd799430002")
SIM_ID = ObjectId("507f1f77bcf86cd799430003")
OTHER_SIM_ID = ObjectId("507f1f77bcf86cd799430004")
ATTEMPT = "8f14e45fceea167a5a36dedd4bea2543"
OTHER_ATTEMPT = "c9f0f895fb98ab9159f51fd0297e236d"


class FakeDb:
    """Minimal ``db[name]`` mapping over per-collection MagicMocks."""

    def __init__(self):
        self.collections: dict[str, MagicMock] = {}

    def __getitem__(self, name: str) -> MagicMock:
        return self.collections.setdefault(name, MagicMock())


def make_connection() -> tuple[MagicMock, FakeDb]:
    db = FakeDb()
    connection = MagicMock()

    @contextmanager
    def connect():
        yield db

    connection.connect = connect
    return connection, db


@pytest.fixture
def resources() -> tuple[CoojaResourceRepository, MagicMock]:
    connection, db = make_connection()
    repo = CoojaResourceRepository(connection)
    coll = db[RESOURCES_COLLECTION]
    coll.reset_mock()                       # drop the index calls from __init__
    # Nothing known yet. A MagicMock would be truthy, and discovery reads
    # `find_one` to recognise a container by its id.
    coll.find_one.return_value = None
    return repo, coll


@pytest.fixture
def hosts() -> tuple[HostRepository, MagicMock]:
    connection, db = make_connection()
    repo = HostRepository(connection)
    coll = db[HOSTS_COLLECTION]
    coll.reset_mock()
    return repo, coll


# ── Indexes ──────────────────────────────────────────────────────────────────

class TestIndexes:
    def test_resource_name_is_unique_per_host(self):
        connection, db = make_connection()
        CoojaResourceRepository(connection)
        calls = db[RESOURCES_COLLECTION].create_index.call_args_list
        unique = [c for c in calls if c.kwargs.get("unique")]
        # Without this, a repeated discovery pass could enroll the same
        # container twice and hand it to two simulations at once.
        assert [("host_id", 1), ("name", 1)] in [c.args[0] for c in unique]

    def test_host_name_is_unique(self):
        connection, db = make_connection()
        HostRepository(connection)
        calls = db[HOSTS_COLLECTION].create_index.call_args_list
        unique = [c for c in calls if c.kwargs.get("unique")]
        assert [("name", 1)] in [c.args[0] for c in unique]


# ── Reservation: the core invariant ──────────────────────────────────────────

class TestReserve:
    def test_filter_admits_only_a_free_enabled_resource(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = {"_id": RES_ID}

        repo.reserve(RES_ID, SIM_ID, ATTEMPT)

        flt, update = coll.find_one_and_update.call_args.args[:2]
        assert flt == {
            "_id": RES_ID,
            "enabled": True,
            "status": EnumResourceStatus.AVAILABLE.value,
        }
        assert update["$set"]["status"] == EnumResourceStatus.RESERVED.value
        assert update["$set"]["current_simulation_id"] == SIM_ID
        assert update["$set"]["attempt_id"] == ATTEMPT

    def test_loser_of_a_race_gets_none(self, resources):
        repo, coll = resources
        # Mongo returns nothing when the filter no longer matches, which is
        # exactly what the second dispatcher sees.
        coll.find_one_and_update.return_value = None
        assert repo.reserve(RES_ID, SIM_ID, ATTEMPT) is None

    def test_reservation_clears_any_previous_attempt_state(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = {"_id": RES_ID}
        repo.reserve(RES_ID, SIM_ID, ATTEMPT)
        sets = coll.find_one_and_update.call_args.args[1]["$set"]
        assert sets["started_at"] is None and sets["remote_pid"] is None

    def test_invalid_id_does_not_reach_mongo(self, resources):
        repo, coll = resources
        assert repo.reserve("not-an-oid", SIM_ID, ATTEMPT) is None
        coll.find_one_and_update.assert_not_called()


class TestMarkBusy:
    def test_requires_the_same_simulation_and_attempt(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.mark_busy(RES_ID, SIM_ID, ATTEMPT, remote_pid=4242) is True

        flt, update = coll.update_one.call_args.args[:2]
        assert flt == {
            "_id": RES_ID,
            "status": EnumResourceStatus.RESERVED.value,
            "current_simulation_id": SIM_ID,
            "attempt_id": ATTEMPT,
        }
        assert update["$set"]["status"] == EnumResourceStatus.BUSY.value
        assert update["$set"]["remote_pid"] == 4242

    def test_stale_attempt_cannot_report_a_start(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=0)
        assert repo.mark_busy(RES_ID, SIM_ID, OTHER_ATTEMPT) is False


# ── Release: guarded, and status derived from what was observed ──────────────

class TestRelease:
    def test_guard_is_simulation_plus_attempt(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = {"_id": RES_ID}

        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=True)

        flt = coll.find_one_and_update.call_args.args[0]
        assert flt == {
            "_id": RES_ID,
            "current_simulation_id": SIM_ID,
            "attempt_id": ATTEMPT,
        }

    def test_late_thread_from_an_old_attempt_releases_nothing(self, resources):
        repo, coll = resources
        # The simulation was recovered and re-placed; this thread must not free
        # a container that is now running something else.
        coll.find_one_and_update.return_value = None
        assert repo.release(RES_ID, OTHER_SIM_ID, OTHER_ATTEMPT,
                           container_state_counts=True) is None

    def test_clears_the_whole_placement(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = {"_id": RES_ID}
        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=True)
        sets = coll.find_one_and_update.call_args.args[1][0]["$set"]
        for field in ("current_simulation_id", "attempt_id", "remote_pid",
                      "reserved_at", "started_at"):
            assert sets[field] is None

    def _release_switch(self, repo, coll) -> dict:
        coll.find_one_and_update.return_value = {"_id": RES_ID}
        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=True)
        pipeline = coll.find_one_and_update.call_args.args[1]
        # An aggregation pipeline, so the outcome is decided inside the same
        # atomic update rather than by a follow-up write that could race.
        assert isinstance(pipeline, list)
        return pipeline[0]["$set"]["status"]["$switch"]

    def test_default_is_back_in_the_pool(self, resources):
        """Regression: requiring positive evidence drained the pool.

        A seeded container has no container.checked_at and no probe.checked_at,
        because discovery has not described it yet. When the release demanded
        `state == running AND ssh_ok`, every completed simulation sent its
        container to `offline` — one container lost per run, until nothing was
        left. Observed in deployment: four of eight offline within a minute.
        """
        repo, coll = resources
        switch = self._release_switch(repo, coll)
        assert switch["default"] == EnumResourceStatus.AVAILABLE.value

    def test_only_a_negative_observation_removes_it_from_the_pool(self, resources):
        repo, coll = resources
        branches = self._release_switch(repo, coll)["branches"]

        outcomes = [b["then"] for b in branches]
        assert outcomes == [
            EnumResourceStatus.DISABLED.value,   # administrative
            EnumResourceStatus.OFFLINE.value,    # Docker looked, not running
            EnumResourceStatus.OFFLINE.value,    # the probe ran, and failed
        ]
        # Each offline branch is guarded by "something actually looked",
        # never by the mere absence of an observation.
        for branch in branches[1:]:
            guard = repr(branch["case"])
            assert "checked_at" in guard
            assert "$ne" in guard

    def test_a_disabled_container_drains_instead_of_returning(self, resources):
        repo, coll = resources
        first = self._release_switch(repo, coll)["branches"][0]
        assert first["case"] == {"$ne": ["$enabled", True]}
        assert first["then"] == EnumResourceStatus.DISABLED.value

    def test_a_stopped_container_does_not_return_to_the_pool(self, resources):
        repo, coll = resources
        branch = self._release_switch(repo, coll)["branches"][1]
        # The dispatcher must not be handed a container Docker says is dead.
        assert EnumContainerState.RUNNING.value in repr(branch["case"])


class TestForceRelease:
    def test_skips_the_attempt_guard_for_recovery(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.force_release(RES_ID, "recovered on startup",
                              container_state_counts=True) is True

        flt = coll.update_one.call_args.args[0]
        assert flt == {"_id": RES_ID}           # no attempt_id: it is known gone
        sets = coll.update_one.call_args.args[1][0]["$set"]
        assert sets["current_simulation_id"] is None
        assert sets["attempt_id"] is None

    def test_the_sweep_can_insist_on_the_reservation_it_read(self, resources):
        """The race the sweep had: read, then free by `_id` alone.

        Between the query and the write, the runner may have promoted the
        reservation to ``busy`` and started the JVM. Freeing it then hands a
        container that is running something to the next placement — two
        simulations on one worker, which is the one thing the reservation
        exists to prevent.
        """
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)
        cutoff = datetime.now() - timedelta(minutes=10)

        repo.force_release(
            RES_ID, "stale", container_state_counts=True,
            expect_attempt_id=ATTEMPT, reserved_before=cutoff)

        flt = coll.update_one.call_args.args[0]
        assert flt["_id"] == RES_ID
        assert flt["attempt_id"] == ATTEMPT
        # Still waiting to start: one that reached `busy` is governed by the
        # simulation timeout, not by this.
        assert flt["status"] == EnumResourceStatus.RESERVED.value
        assert flt["reserved_at"] == {"$lt": cutoff}

    def test_startup_recovery_still_frees_unconditionally(self, resources):
        # There the attempt is known gone: the process that held it died.
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.force_release(RES_ID, "recovered", container_state_counts=True)

        assert coll.update_one.call_args.args[0] == {"_id": RES_ID}

    def test_a_reservation_that_moved_on_is_not_freed(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.force_release(
            RES_ID, "stale", container_state_counts=True,
            expect_attempt_id=ATTEMPT) is False

    def test_recovery_returns_the_container_to_the_pool(self, resources):
        """Regression: recovery handed back a pool of zero.

        force_release used to force `offline`. On a restart the container is
        fine — it is the process that died — so every recovered container left
        the pool and the dispatcher had nowhere to place work. Observed in
        deployment as `local[0a/0f]` right after a successful recovery of all
        eight containers.
        """
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.force_release(RES_ID, "recovered on master-node restart",
                           container_state_counts=True)

        status = coll.update_one.call_args.args[1][0]["$set"]["status"]
        assert status == released_status_expression(container_state_counts=True)

    def test_both_release_paths_land_a_container_the_same_way(self, resources):
        # One expression, so there is a single place to revisit when the health
        # cycle starts supplying real observations.
        repo, coll = resources
        coll.find_one_and_update.return_value = {"_id": RES_ID}
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=True)
        guarded = coll.find_one_and_update.call_args.args[1][0]["$set"]["status"]
        repo.force_release(RES_ID, container_state_counts=True)
        forced = coll.update_one.call_args.args[1][0]["$set"]["status"]

        assert guarded == forced == released_status_expression(
            container_state_counts=True)


class TestDockerEvidenceIsOptionalOnRelease:
    """Where the operator stated the address, Docker's view does not apply.

    The health cycle already made that exception; the release path had not,
    and the two documents it reads are on different collections, so the
    aggregation cannot work it out for itself. The cost of that disagreement
    was measured on a live installation: `vm7-cooja1` and `vm7-cooja2` carried
    `container.state = "missing"` from a host whose SSH user cannot reach the
    Docker daemon, so **every completed simulation sent them out of the pool**,
    and they only returned at the next health cycle. Across one pilot's 54
    remote simulations that is up to 27 minutes of remote capacity lost.

    The parameter is required precisely because the silent default is what
    produced the defect.
    """

    def _switch(self, repo, coll, *, counts: bool) -> dict:
        coll.find_one_and_update.return_value = {"_id": RES_ID}
        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=counts)
        pipeline = coll.find_one_and_update.call_args.args[1]
        return pipeline[0]["$set"]["status"]["$switch"]

    def test_the_docker_branch_is_dropped_when_it_does_not_apply(self, resources):
        repo, coll = resources
        branches = self._switch(repo, coll, counts=False)["branches"]

        guards = repr(branches)
        assert "container.state" not in guards
        assert "probe.ssh_ok" in guards          # the probe still decides

    def test_the_probe_still_removes_it_from_the_pool(self, resources):
        repo, coll = resources
        branches = self._switch(repo, coll, counts=False)["branches"]

        assert [b["then"] for b in branches] == [
            EnumResourceStatus.DISABLED.value,   # administrative
            EnumResourceStatus.OFFLINE.value,    # the probe ran, and failed
        ]

    def test_a_manual_worker_with_a_stale_missing_stays_in_the_pool(self, resources):
        repo, coll = resources
        assert self._switch(repo, coll, counts=False)["default"] \
            == EnumResourceStatus.AVAILABLE.value

    def test_docker_still_counts_for_a_discovered_worker(self, resources):
        repo, coll = resources
        branches = self._switch(repo, coll, counts=True)["branches"]

        assert EnumContainerState.RUNNING.value in repr(branches[1]["case"])

    def test_both_paths_agree_under_the_same_answer(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = {"_id": RES_ID}
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=False)
        guarded = coll.find_one_and_update.call_args.args[1][0]["$set"]["status"]
        repo.force_release(RES_ID, container_state_counts=False)
        forced = coll.update_one.call_args.args[1][0]["$set"]["status"]

        assert guarded == forced


# ── Health checks must never evict a simulation ──────────────────────────────

class TestOccupiedIsProtected:
    @pytest.mark.parametrize("method", ["mark_available", "mark_offline", "mark_error"])
    def test_status_changes_exclude_reserved_and_busy(self, resources, method):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        getattr(repo, method)(RES_ID, "check result")

        flt = coll.update_one.call_args.args[0]
        allowed = flt["status"]["$in"]
        # The whole point of §9.10: losing contact with a host says nothing
        # about whether the simulation on it is still running.
        assert EnumResourceStatus.RESERVED.value not in allowed
        assert EnumResourceStatus.BUSY.value not in allowed

    def test_disabling_an_occupied_resource_lets_it_finish(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1, modified_count=1)

        repo.set_enabled(RES_ID, False)

        status_call, enabled_call = coll.update_one.call_args_list[:2]
        # The status is only forced for a resource that is not occupied; the
        # enabled flag is set either way, so a busy one drains on release.
        assert status_call.args[0]["status"]["$in"]
        assert EnumResourceStatus.BUSY.value not in status_call.args[0]["status"]["$in"]
        assert enabled_call.args[1]["$set"]["enabled"] is False


# ── Discovery ────────────────────────────────────────────────────────────────

class TestSyncDiscovered:
    def _containers(self):
        return [
            {"id": "aaa", "name": "cooja1", "image": "cooja:v1.1",
             "state": EnumContainerState.RUNNING.value, "labels": {}},
            {"id": "bbb", "name": "cooja2", "image": "cooja:v1.1",
             "state": EnumContainerState.EXITED.value, "labels": {}},
        ]

    def test_new_container_starts_outside_the_pool(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=ObjectId())
        coll.find.return_value = []

        result = repo.sync_discovered(
            HOST_ID, self._containers(), remote_dir="/opt/cooja")

        assert result["created"] == ["cooja1", "cooja2"]
        on_insert = coll.update_one.call_args.args[1]["$setOnInsert"]
        # Docker saying "running" is not evidence that Cooja can run: the SSH
        # probe decides, so nothing enters the pool on discovery alone.
        assert on_insert["status"] == EnumResourceStatus.OFFLINE.value
        assert on_insert["enabled"] is True
        assert on_insert["origin"] == "discovered"

    def test_known_container_keeps_enabled_and_status(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=None)
        coll.find.return_value = []

        result = repo.sync_discovered(
            HOST_ID, self._containers(), remote_dir="/opt/cooja")

        assert result["updated"] == ["cooja1", "cooja2"]
        set_fields = coll.update_one.call_args.args[1]["$set"]
        # Overwriting either would undo an administrative decision or evict a
        # running simulation on every discovery pass.
        assert "enabled" not in set_fields
        assert "status" not in set_fields
        assert set_fields["container"]["name"] == "cooja2"

    def _sync_with_endpoint(self, repo, coll, *, upserted):
        coll.update_one.return_value = MagicMock(upserted_id=upserted)
        coll.find.return_value = []
        repo.sync_discovered(
            HOST_ID, self._containers()[:1], remote_dir="/opt/cooja",
            endpoints={"cooja1": {"address": "172.18.0.5", "port": 22}},
        )
        return coll.update_one.call_args_list

    def test_endpoint_is_applied_when_resolved(self, resources):
        repo, coll = resources
        calls = self._sync_with_endpoint(repo, coll, upserted=None)

        last = calls[-1]
        assert last.args[1]["$set"]["endpoint"] == {
            "address": "172.18.0.5", "port": 22,
        }

    def test_the_address_of_a_hand_configured_worker_is_left_alone(self, resources):
        """Who owns the address is who typed it.

        Discovery excludes manual workers from the vanished list, but used to
        refresh the endpoint of *any* record matching `(host_id, name)`. A
        port or address a person set was replaced on the next inspection,
        silently undoing the configuration — and on a host whose daemon the
        SSH user cannot reach, replaced with whatever Docker guessed.
        """
        repo, coll = resources
        calls = self._sync_with_endpoint(repo, coll, upserted=None)

        guarded = calls[-1]
        assert guarded.args[0] == {
            "host_id": HOST_ID, "name": "cooja1", "origin": {"$ne": "manual"},
        }
        assert set(guarded.args[1]["$set"]) == {"endpoint"}

    def test_a_new_worker_gets_its_address_at_creation(self, resources):
        """One write, not two: nothing owns it yet."""
        repo, coll = resources
        calls = self._sync_with_endpoint(repo, coll, upserted=ObjectId())

        assert len(calls) == 1
        assert calls[0].args[1]["$setOnInsert"]["endpoint"] == {
            "address": "172.18.0.5", "port": 22,
        }

    def test_container_facts_are_refreshed_for_everyone(self, resources):
        # Only the *address* is the operator's; what Docker saw is still an
        # observation worth recording against a manual worker.
        repo, coll = resources
        calls = self._sync_with_endpoint(repo, coll, upserted=None)

        upsert = calls[0]
        assert upsert.args[0] == {"host_id": HOST_ID, "name": "cooja1"}
        assert "container" in upsert.args[1]["$set"]

    def test_a_container_is_recognised_by_its_id(self, resources):
        """The identity is the container, not the name it happens to carry.

        Matching on the name alone meant a record whose name had diverged —
        renamed here, or renamed in Docker — went unrecognised and the
        container was inserted a second time: two registrations and two
        reservations for one machine.
        """
        repo, coll = resources
        coll.find_one.return_value = {"_id": RES_ID, "name": "old-name"}
        coll.update_one.return_value = MagicMock(upserted_id=None)
        coll.find.return_value = []

        repo.sync_discovered(
            HOST_ID, self._containers()[:1], remote_dir="/opt/cooja")

        # Looked up by the container's id...
        assert coll.find_one.call_args.args[0] == {
            "host_id": HOST_ID, "container.id": "aaa"}
        # ...and updated by _id, so the name is not the key.
        first = coll.update_one.call_args_list[0]
        assert first.args[0] == {"_id": RES_ID}
        # The divergence is followed rather than duplicated.
        assert first.args[1]["$set"]["name"] == "cooja1"

    def test_a_container_nobody_has_seen_is_inserted_by_name(self, resources):
        repo, coll = resources
        coll.find_one.return_value = None
        coll.update_one.return_value = MagicMock(upserted_id=ObjectId())
        coll.find.return_value = []

        result = repo.sync_discovered(
            HOST_ID, self._containers()[:1], remote_dir="/opt/cooja")

        assert result["created"] == ["cooja1"]
        assert coll.update_one.call_args_list[0].args[0] == {
            "host_id": HOST_ID, "name": "cooja1"}

    def test_a_container_without_an_id_falls_back_to_the_name(self, resources):
        # Nothing to match on; the name is all there is.
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=None)
        coll.find.return_value = []

        repo.sync_discovered(
            HOST_ID, [{"name": "cooja9", "state": "running"}],
            remote_dir="/opt/cooja")

        coll.find_one.assert_not_called()

    def test_vanished_container_is_marked_not_deleted(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=None)
        coll.find.return_value = [
            {"name": "cooja9", "status": EnumResourceStatus.AVAILABLE.value},
        ]

        result = repo.sync_discovered(
            HOST_ID, self._containers(), remote_dir="/opt/cooja")

        assert result["missing"] == ["cooja9"]
        assert result["missing_occupied"] == []
        # The resource is what ties finished simulations to where they ran.
        coll.delete_many.assert_not_called()
        coll.delete_one.assert_not_called()
        marked = coll.update_many.call_args_list[0].args[1]["$set"]
        assert marked["container.state"] == EnumContainerState.MISSING.value

    def test_vanished_while_busy_is_reported_not_decided(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=None)
        coll.find.return_value = [
            {"name": "cooja9", "status": EnumResourceStatus.BUSY.value},
        ]

        result = repo.sync_discovered(
            HOST_ID, self._containers(), remote_dir="/opt/cooja")

        assert result["missing_occupied"] == ["cooja9"]
        # Whether the simulation died is a policy question; the second
        # update_many only moves resources that are not occupied.
        pool_filter = coll.update_many.call_args_list[1].args[0]
        assert EnumResourceStatus.BUSY.value not in pool_filter["status"]["$in"]

    def test_seeded_insert_never_overwrites(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=ObjectId())

        inserted = repo.insert_seeded({
            "host_id": HOST_ID, "name": "cooja1",
            "status": EnumResourceStatus.AVAILABLE.value, "enabled": True,
        })

        assert inserted is True
        flt, update = coll.update_one.call_args.args[:2]
        assert flt == {"host_id": HOST_ID, "name": "cooja1"}
        # $setOnInsert only, and no $set at all: the startup seed must not
        # overwrite a resource discovery has described, one an operator
        # disabled, or one holding a simulation right now.
        assert set(update) == {"$setOnInsert"}
        assert "host_id" not in update["$setOnInsert"]
        assert "name" not in update["$setOnInsert"]
        assert coll.update_one.call_args.kwargs["upsert"] is True

    def test_seeded_insert_reports_when_it_did_nothing(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(upserted_id=None)
        assert repo.insert_seeded({"host_id": HOST_ID, "name": "cooja1"}) is False

    def test_unnamed_container_is_skipped(self, resources):
        repo, coll = resources
        coll.find.return_value = []
        result = repo.sync_discovered(HOST_ID, [{"id": "ccc"}], remote_dir="/opt/cooja")
        assert result["created"] == [] and result["updated"] == []
        coll.update_one.assert_not_called()


# ── Queries the scheduler and recovery paths rely on ─────────────────────────

class TestQueries:
    def test_free_by_host_requires_enabled_and_available(self, resources):
        repo, coll = resources
        coll.find.return_value = []
        repo.find_free_by_host(HOST_ID)
        assert coll.find.call_args.args[0] == {
            "host_id": HOST_ID,
            "enabled": True,
            "status": EnumResourceStatus.AVAILABLE.value,
        }

    def test_free_by_host_spreads_across_containers(self, resources):
        repo, coll = resources
        coll.find.return_value = []
        repo.find_free_by_host(HOST_ID)
        assert coll.find.call_args.kwargs["sort"] == [("last_check", 1), ("name", 1)]

    def test_occupied_lists_both_holding_states(self, resources):
        repo, coll = resources
        coll.find.return_value = []
        repo.find_occupied()
        statuses = coll.find.call_args.args[0]["status"]["$in"]
        assert set(statuses) == {
            EnumResourceStatus.RESERVED.value, EnumResourceStatus.BUSY.value,
        }

    def test_stale_reservations_look_only_at_reserved(self, resources):
        repo, coll = resources
        coll.find.return_value = []
        cutoff = datetime.now() - timedelta(minutes=10)
        repo.find_stale_reservations(cutoff)
        flt = coll.find.call_args.args[0]
        # Only the reserve-to-start window: a busy resource is governed by the
        # simulation timeout, which may legitimately be hours.
        assert flt["status"] == EnumResourceStatus.RESERVED.value
        assert flt["reserved_at"] == {"$lt": cutoff}

    def test_active_counts_are_grouped_by_host(self, resources):
        repo, coll = resources
        coll.aggregate.return_value = [{"_id": HOST_ID, "n": 3}]
        assert repo.count_active_by_host() == {HOST_ID: 3}
        match = coll.aggregate.call_args.args[0][0]["$match"]
        assert set(match["status"]["$in"]) == {
            EnumResourceStatus.RESERVED.value, EnumResourceStatus.BUSY.value,
        }


class TestConditionalDeletion:
    """Deletion carries its condition, instead of trusting a read before it.

    The API asked whether anything was running and deleted afterwards. The
    scheduler can reserve a container in that window, and the delete then
    removed the record of a live run — with no way to find it again, because
    the reservation went with it.
    """

    def test_a_worker_is_deleted_only_while_free(self, resources):
        repo, coll = resources
        coll.delete_one.return_value = MagicMock(deleted_count=1)

        assert repo.delete_if_free(RES_ID) is True

        flt = coll.delete_one.call_args.args[0]
        assert flt["_id"] == RES_ID
        assert set(flt["status"]["$nin"]) == {
            EnumResourceStatus.RESERVED.value, EnumResourceStatus.BUSY.value}
        assert flt["current_simulation_id"] is None

    def test_an_occupied_worker_survives(self, resources):
        repo, coll = resources
        coll.delete_one.return_value = MagicMock(deleted_count=0)

        assert repo.delete_if_free(RES_ID) is False

    def test_a_drained_host_and_its_workers_go(self):
        connection, db = make_connection()
        repo = HostRepository(connection)
        resources_coll = db[RESOURCES_COLLECTION]
        resources_coll.find.return_value = [
            {"_id": RES_ID, "name": "cooja1"},
            {"_id": OTHER_SIM_ID, "name": "cooja2"},
        ]
        resources_coll.delete_one.return_value = MagicMock(deleted_count=1)
        db[HOSTS_COLLECTION].delete_one.return_value = MagicMock(deleted_count=1)

        deleted, blocking = repo.delete_if_drained(HOST_ID)

        assert deleted is True and blocking == []
        assert resources_coll.delete_one.call_count == 2

    def test_one_occupied_worker_blocks_the_whole_host(self):
        # And names itself, so the operator knows what to drain.
        connection, db = make_connection()
        repo = HostRepository(connection)
        resources_coll = db[RESOURCES_COLLECTION]
        resources_coll.find.return_value = [
            {"_id": RES_ID, "name": "cooja1"},
            {"_id": OTHER_SIM_ID, "name": "cooja2"},
        ]
        resources_coll.delete_one.side_effect = [
            MagicMock(deleted_count=1), MagicMock(deleted_count=0)]

        deleted, blocking = repo.delete_if_drained(HOST_ID)

        assert deleted is False
        assert blocking == ["cooja2"]
        # The host stays, so the surviving worker is not orphaned.
        db[HOSTS_COLLECTION].delete_one.assert_not_called()

    def test_a_host_with_no_workers_is_deleted(self):
        connection, db = make_connection()
        repo = HostRepository(connection)
        db[RESOURCES_COLLECTION].find.return_value = []
        db[HOSTS_COLLECTION].delete_one.return_value = MagicMock(deleted_count=1)

        assert repo.delete_if_drained(HOST_ID) == (True, [])


# ── Workers configured by hand ───────────────────────────────────────────────

class TestUpdateManual:
    def test_configuring_a_connection_takes_ownership(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.update_manual(RES_ID, {"connection": {"port": 2223}})

        payload = coll.update_one.call_args.args[1]["$set"]
        # Without this the next discovery pass marks a hand-configured worker
        # as missing and it is never probed.
        assert payload["origin"] == "manual"

    def test_removing_the_last_override_gives_it_back_to_discovery(self, resources):
        """"Inherit again" has to actually inherit.

        Ownership used to be a one-way door: once edited, a worker stayed
        flagged as operator-configured for ever, so Docker's view of it went
        on being ignored even after the last override was gone.
        """
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)
        coll.find_one.return_value = {
            "_id": RES_ID, "container": {"id": "abc"}, "connection": {}}

        repo.update_manual(RES_ID, {"connection": {"transport": "ssh"}})

        assert coll.update_one.call_args.args[1]["$set"]["origin"] == "discovered"

    def test_a_standing_override_keeps_ownership(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)
        coll.find_one.return_value = {
            "_id": RES_ID, "container": {"id": "abc"}, "connection": {}}

        repo.update_manual(RES_ID, {"connection": {"port": 2223}})

        assert coll.update_one.call_args.args[1]["$set"]["origin"] == "manual"

    def test_a_hand_registered_worker_stays_manual(self, resources):
        # Docker never described it, so there is nothing to inherit back to.
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)
        coll.find_one.return_value = {"_id": RES_ID, "connection": {}}

        repo.update_manual(RES_ID, {"connection": {"transport": "ssh"}})

        assert coll.update_one.call_args.args[1]["$set"]["origin"] == "manual"

    def test_stating_an_endpoint_takes_ownership(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)
        coll.find_one.return_value = {"_id": RES_ID, "container": {"id": "abc"}}

        repo.update_manual(RES_ID, {"endpoint": {"address": "10.0.0.9"}})

        assert coll.update_one.call_args.args[1]["$set"]["origin"] == "manual"

    def test_the_endpoint_discovery_wrote_is_not_a_statement(self, resources):
        """A discovered worker always has an endpoint: discovery writes it.

        Reading the *stored* endpoint as ownership kept every edited worker
        manual for ever — the one-way door this is meant to remove.
        """
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)
        coll.find_one.return_value = {
            "_id": RES_ID,
            "container": {"id": "abc"},
            "endpoint": {"address": "172.18.0.5", "port": 22},
            "connection": {},
        }

        repo.update_manual(RES_ID, {"connection": {"transport": "ssh"}})

        assert coll.update_one.call_args.args[1]["$set"]["origin"] == "discovered"

    def test_changing_the_address_clears_the_probe(self, resources):
        """The stored probe measured the old address.

        Keeping it would show a result for a path that no longer exists, and
        would let the health cycle skip the one worker that most needs
        measuring — it trusts a recent *ready* probe.
        """
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.update_manual(RES_ID, {"endpoint": {"address": "10.0.0.9"}})

        assert coll.update_one.call_args.args[1]["$set"]["probe"] == {}

    def test_an_unrelated_change_leaves_the_probe_alone(self, resources):
        # Renaming or re-enabling a worker says nothing about reaching it.
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.update_manual(RES_ID, {"enabled": True})

        payload = coll.update_one.call_args.args[1]["$set"]
        assert "probe" not in payload and "origin" not in payload

    def test_measured_fields_are_refused(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.update_manual(RES_ID, {
            "status": "available", "container": {"state": "running"},
            "probe": {"ssh_ok": True}, "connection": {"port": 2223},
        })

        payload = coll.update_one.call_args.args[1]["$set"]
        assert "status" not in payload and "container" not in payload
        # The probe *is* in the payload — cleared by this repository because
        # the address changed, not written from the client's value.
        assert payload["probe"] == {}

    def test_nothing_to_change_is_not_a_write(self, resources):
        repo, coll = resources
        assert repo.update_manual(RES_ID, {"status": "available"}) is False
        coll.update_one.assert_not_called()


# ── Hosts ────────────────────────────────────────────────────────────────────

class TestHosts:
    def test_bootstrap_never_overwrites_an_edit(self, hosts):
        repo, coll = hosts
        coll.find_one_and_update.return_value = {"_id": HOST_ID}

        repo.upsert_bootstrap({
            "name": "local", "type": "local", "enabled": True,
            "access": None, "origin": "bootstrap",
        })

        update = coll.find_one_and_update.call_args.args[1]
        # Everything but the lookup key goes under $setOnInsert, so a restart
        # cannot re-enable a host an operator disabled or undo its access
        # config. Only updated_time is written on an existing document.
        assert set(update["$set"]) == {"updated_time"}
        assert update["$setOnInsert"]["origin"] == "bootstrap"
        assert coll.find_one_and_update.call_args.kwargs["upsert"] is True

    def test_update_refuses_observed_fields(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.update(HOST_ID, {
            "name": "lab-01",
            "status": EnumHostStatus.ONLINE.value,     # an observation
            "consecutive_failures": 0,                  # bookkeeping
        })

        payload = coll.update_one.call_args.args[1]["$set"]
        assert payload["name"] == "lab-01"
        assert "status" not in payload and "consecutive_failures" not in payload

    def test_update_with_only_protected_fields_is_a_no_op(self, hosts):
        repo, coll = hosts
        assert repo.update(HOST_ID, {"status": "online"}) is False
        coll.update_one.assert_not_called()

    def test_disabling_marks_the_host_out_of_the_pool(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)
        repo.set_enabled(HOST_ID, False)
        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["enabled"] is False
        assert sets["status"] == EnumHostStatus.DISABLED.value

    def test_enabling_leaves_the_status_to_the_next_check(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)
        repo.set_enabled(HOST_ID, True)
        # Claiming "online" without probing would show a state nobody measured.
        assert "status" not in coll.update_one.call_args.args[1]["$set"]

    def test_check_result_skips_a_disabled_host(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_check_result(
            HOST_ID, status=EnumHostStatus.ONLINE, message="ok", reset_failures=True)

        flt, update = coll.update_one.call_args.args[:2]
        # An administrative decision outranks an observation.
        assert flt["enabled"] is True
        assert update["$set"]["consecutive_failures"] == 0
        assert "$inc" not in update

    def test_failed_check_counts_towards_the_threshold(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)
        repo.mark_check_result(HOST_ID, status=EnumHostStatus.OFFLINE, message="timeout")
        update = coll.update_one.call_args.args[1]
        # Counted rather than reset, so one dropped packet does not remove a
        # host's capacity from the pool.
        assert update["$inc"] == {"consecutive_failures": 1}

    def test_request_is_recorded_for_the_master_node(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)
        repo.request_check(HOST_ID)
        sets = coll.update_one.call_args.args[1]["$set"]
        assert isinstance(sets["check_requested_at"], datetime)

    def test_clearing_a_request_respects_a_newer_one(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)
        stamp = datetime.now()

        repo.clear_request(HOST_ID, "check_requested_at", stamp)

        # Matching on the timestamp: a request that arrived while the check was
        # running stays pending instead of being silently swallowed.
        assert coll.update_one.call_args.args[0]["check_requested_at"] == stamp

    def test_delete_removes_the_hosts_resources_too(self):
        connection, db = make_connection()
        repo = HostRepository(connection)
        db[RESOURCES_COLLECTION].find.return_value = [{"_id": RES_ID, "name": "cooja1"}]
        db[RESOURCES_COLLECTION].delete_one.return_value = MagicMock(deleted_count=1)
        db[HOSTS_COLLECTION].delete_one.return_value = MagicMock(deleted_count=1)

        assert repo.delete_if_drained(HOST_ID) == (True, [])

        # One conditional delete per worker, rather than a blanket
        # `delete_many`: the condition is what makes it safe.
        db[RESOURCES_COLLECTION].delete_many.assert_not_called()
        assert db[RESOURCES_COLLECTION].delete_one.call_count == 1


class TestInventorySnapshot:
    def test_counts_are_honest_and_eligibility_is_separate(self, hosts):
        repo, coll = hosts
        coll.aggregate.return_value = []

        repo.inventory_snapshot()

        pipeline = coll.aggregate.call_args.args[0]
        assert pipeline[0]["$lookup"]["from"] == RESOURCES_COLLECTION
        fields = pipeline[1]["$set"]
        # free/active describe the containers; schedulable describes the host.
        # Folding the two together would make the GUI under-report capacity
        # that is merely unreachable.
        assert set(fields) >= {
            "resource_total", "free", "active", "unavailable", "disabled", "schedulable",
        }

    def test_free_requires_enabled_and_available(self, hosts):
        repo, coll = hosts
        coll.aggregate.return_value = []
        repo.inventory_snapshot()
        cond = repr(coll.aggregate.call_args.args[0][1]["$set"]["free"])
        assert "enabled" in cond and EnumResourceStatus.AVAILABLE.value in cond

    def test_snapshot_is_one_round_trip(self, hosts):
        repo, coll = hosts
        coll.aggregate.return_value = []
        repo.inventory_snapshot()
        # The placement decision reads this per job: a per-host query would
        # turn one decision into N round trips.
        assert coll.aggregate.call_count == 1
        coll.find.assert_not_called()


# ── Published crypto ─────────────────────────────────────────────────────────

class TestInfraCrypto:
    def test_publish_is_idempotent_on_one_document(self):
        connection, db = make_connection()
        repo = InfraCryptoRepository(connection)

        repo.publish(
            alg="crypto_box_seal", key_id="b114dd6d8410a62a",
            public_key="cHVi", fingerprint="SHA256:x", retired_key_ids=["old"],
        )

        coll = db["infra_crypto"]
        flt, update = coll.update_one.call_args.args[:2]
        assert flt == {"_id": "active"}
        assert coll.update_one.call_args.kwargs["upsert"] is True
        assert update["$set"]["public_key"] == "cHVi"
        assert "created_at" in update["$setOnInsert"]

    def test_nothing_private_is_published(self):
        connection, db = make_connection()
        repo = InfraCryptoRepository(connection)
        repo.publish(
            alg="crypto_box_seal", key_id="k", public_key="cHVi", fingerprint="SHA256:x")
        written = db["infra_crypto"].update_one.call_args.args[1]
        assert "private_key" not in repr(written)
