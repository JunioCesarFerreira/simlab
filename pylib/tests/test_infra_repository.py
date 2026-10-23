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

from pylib.timeutil import utc_now
from pylib.db.models.infra import (
    EnumContainerState,
    EnumHostStatus,
    EnumResourceStatus,
    MACHINES_COLLECTION,
    MASTER_STATUS_COLLECTION,
    machine_of,
    RESOURCES_COLLECTION,
    HOSTS_COLLECTION,
)
from pylib.db.repositories.infra import (
    _REASSIGNABLE as _REASSIGNABLE_FOR_TEST,
    CoojaResourceRepository,
    HostWithdrawing,
    MachineRepository,
    MasterStatusRepository,
    RemovalConflicted,
    HostRepository,
    InfraCryptoRepository,
    released_status_expression,
)

#: Stands in for a client session. Identity is what the tests compare, so a
#: sentinel is clearer than a fresh mock per call.
FAKE_SESSION = MagicMock(name="session")

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
        collection = self.collections.get(name)
        if collection is None:
            collection = MagicMock()
            # A count answers zero unless a test says otherwise. A MagicMock's
            # own answer is a truthy object, which reads as "yes, there is
            # one" for every `count_documents` guard in the repository — so
            # the default has to be the quiet one.
            collection.count_documents.return_value = 0
            self.collections[name] = collection
        return collection


def make_connection() -> tuple[MagicMock, FakeDb]:
    db = FakeDb()
    connection = MagicMock()

    @contextmanager
    def connect():
        yield db

    @contextmanager
    def transaction():
        # The session is a stand-in: what the tests assert is that every
        # operation *carries* one, which is what keeps a write inside the
        # transaction. A real session's isolation is MongoDB's business and is
        # exercised against the deployment, not here.
        yield db, FAKE_SESSION

    connection.connect = connect
    connection.transaction = transaction
    # A host in service, by default. Creation now *reads* the host to refuse a
    # withdrawal and *writes* it to conflict with one, so a MagicMock's own
    # answer — a truthy object whose `withdrawn_at` is also truthy — would
    # refuse every registration.
    db[HOSTS_COLLECTION].find_one.return_value = {
        "_id": HOST_ID, "withdrawn_at": None}
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
            "held_for": None,
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
            # And nothing unresolved on the container: see TestHeldWorkers.
            "held_for": None,
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
        # No attempt_id: it is known gone. But still not a worker holding
        # something unresolved.
        assert flt == {"_id": RES_ID, "held_for": None}
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
            expect_attempt_id=ATTEMPT, reserved_before=cutoff,
            expect_reserved=True)

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

        assert coll.update_one.call_args.args[0] == {
            "_id": RES_ID, "held_for": None}

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

    def test_a_retained_worker_cannot_be_deleted(self, resources):
        repo, coll = resources
        coll.delete_one.return_value = MagicMock(deleted_count=0)

        assert repo.delete_if_free(RES_ID) is False
        assert coll.delete_one.call_args.args[0]["held_for"] is None

    def test_an_occupied_worker_survives(self, resources):
        repo, coll = resources
        coll.delete_one.return_value = MagicMock(deleted_count=0)

        assert repo.delete_if_free(RES_ID) is False

    @staticmethod
    def _host_removal(listed, *, deleted_each=1, host_deleted=1, late=None):
        """A host repository whose collections answer a scripted removal.

        ``listed`` is what the loop sees, ``late`` what the re-count inside
        the same transaction finds — the worker a concurrent registration
        inserted, which is the subject of R09.
        """
        connection, db = make_connection()
        repo = HostRepository(connection)
        resources_coll = db[RESOURCES_COLLECTION]
        passes = [list(listed), list(late if late is not None else [])]
        resources_coll.find.side_effect = lambda *a, **k: passes.pop(0) if passes else []
        if isinstance(deleted_each, list):
            resources_coll.delete_one.side_effect = [
                MagicMock(deleted_count=n) for n in deleted_each]
        else:
            resources_coll.delete_one.return_value = MagicMock(
                deleted_count=deleted_each)
        db[HOSTS_COLLECTION].update_one.return_value = MagicMock(matched_count=1)
        db[HOSTS_COLLECTION].delete_one.return_value = MagicMock(
            deleted_count=host_deleted)
        return repo, db, resources_coll

    def test_a_drained_host_and_its_workers_go(self):
        repo, db, resources_coll = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"},
             {"_id": OTHER_SIM_ID, "name": "cooja2"}])

        deleted, blocking = repo.delete_if_drained(HOST_ID)

        assert deleted is True and blocking == []
        assert resources_coll.delete_one.call_count == 2

    def test_the_host_is_withdrawn_before_the_transaction(self):
        """Phase one, and it is outside the transaction on purpose.

        It is what makes a *later* registration refuse — a check the other
        side performs on its own. The transaction then covers the deletions,
        and the host write inside it is what makes a registration that
        started earlier conflict.
        """
        repo, db, _ = self._host_removal([{"_id": RES_ID, "name": "cooja1"}])

        repo.delete_if_drained(HOST_ID)

        marked = db[HOSTS_COLLECTION].update_one.call_args_list[0]
        assert marked.args[0] == {"_id": HOST_ID}
        assert "withdrawn_at" in marked.args[1]["$set"]

    def test_host_removal_preserves_retained_results_without_an_active_reservation(self):
        repo, _, resources_coll = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"}], deleted_each=0)

        assert repo.delete_if_drained(HOST_ID) == (False, ["cooja1"])
        assert resources_coll.delete_one.call_args.args[0]["held_for"] is None

    def test_the_transaction_writes_the_host_so_creation_conflicts(self):
        """A transaction's snapshot does not see a document someone else
        inserts, so reading the host would not exclude a concurrent
        registration. Writing it does: both sides write the same document and
        MongoDB aborts one of them."""
        repo, db, _ = self._host_removal([{"_id": RES_ID, "name": "cooja1"}])

        repo.delete_if_drained(HOST_ID)

        bumped = [
            c for c in db[HOSTS_COLLECTION].update_one.call_args_list
            if "$inc" in c.args[1] and "resource_epoch" in c.args[1]["$inc"]
        ]
        assert bumped, "the host was not written inside the transaction"
        assert bumped[0].kwargs["session"] is FAKE_SESSION

    def test_every_operation_carries_the_session(self):
        """An operation without one runs outside the transaction and silently
        loses the guarantee."""
        repo, db, resources_coll = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"}])

        repo.delete_if_drained(HOST_ID)

        for call in (list(resources_coll.find.call_args_list)
                     + list(resources_coll.delete_one.call_args_list)
                     + list(db[HOSTS_COLLECTION].delete_one.call_args_list)):
            assert call.kwargs.get("session") is FAKE_SESSION, call

    def test_one_occupied_worker_blocks_the_whole_host(self):
        # And names itself, so the operator knows what to drain.
        repo, db, _ = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"},
             {"_id": OTHER_SIM_ID, "name": "cooja2"}],
            deleted_each=[1, 0])

        deleted, blocking = repo.delete_if_drained(HOST_ID)

        assert deleted is False
        assert blocking == ["cooja2"]
        db[HOSTS_COLLECTION].delete_one.assert_not_called()

    def test_a_refusal_is_no_longer_partial(self):
        """The property the transaction buys.

        Before, the workers deleted before the blocking one stayed deleted,
        because undoing a delete is not possible outside a transaction. Now
        the whole thing rolls back, and the refusal costs nothing.
        """
        repo, db, _ = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"},
             {"_id": OTHER_SIM_ID, "name": "cooja2"}],
            deleted_each=[1, 0])

        deleted, blocking = repo.delete_if_drained(HOST_ID)

        assert (deleted, blocking) == (False, ["cooja2"])
        # Raised rather than returned, so the context manager rolls back. A
        # plain return would have committed the delete that did go through.
        FAKE_SESSION.reset_mock()

    def test_a_refused_removal_puts_the_host_back_in_service(self):
        """Otherwise the capacity is gone with nothing saying why."""
        repo, db, _ = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"}], deleted_each=0)

        repo.delete_if_drained(HOST_ID)

        unset = [
            c for c in db[HOSTS_COLLECTION].update_one.call_args_list
            if "$unset" in c.args[1]
        ]
        assert unset and "withdrawn_at" in unset[0].args[1]["$unset"]

    def test_a_worker_registered_during_the_removal_blocks_it(self):
        """R09, reproduced. The loop deletes what it listed; a worker
        inserted after that list was taken is invisible to it, and the host
        was deleted on top of it — leaving an orphan. The re-count inside the
        transaction sees it and the whole removal rolls back.
        """
        repo, db, _ = self._host_removal(
            [{"_id": RES_ID, "name": "cooja1"}],
            late=[{"_id": OTHER_SIM_ID, "name": "cooja9"}])

        deleted, blocking = repo.delete_if_drained(HOST_ID)

        assert deleted is False
        assert blocking == ["cooja9"]
        db[HOSTS_COLLECTION].delete_one.assert_not_called()

    def test_a_write_conflict_is_reported_as_a_refusal(self):
        """A registration committed against this host while the removal ran.
        Trying again is the correct outcome; removing the host and finding the
        worker afterwards is not."""
        from pymongo.errors import OperationFailure

        connection, db = make_connection()
        repo = HostRepository(connection)
        db[HOSTS_COLLECTION].update_one.side_effect = [
            MagicMock(matched_count=1),                 # the withdrawal
            OperationFailure("WriteConflict"),          # inside the transaction
            MagicMock(matched_count=1),                 # cancel_withdrawal
        ]

        with pytest.raises(RemovalConflicted):
            repo.delete_if_drained(HOST_ID)

        # And the host is back in service: a conflict means "try again", not
        # "this host is on its way out" — nor "this host does not exist",
        # which is what collapsing it into the empty-blocking case answered.
        assert "$unset" in db[HOSTS_COLLECTION].update_one.call_args.args[1]

    def test_a_host_with_no_workers_is_deleted(self):
        repo, _, _ = self._host_removal([])

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
        repo, db, resources_coll = TestConditionalDeletion._host_removal(
            [{"_id": RES_ID, "name": "cooja1"}])

        assert repo.delete_if_drained(HOST_ID) == (True, [])

        # One *conditional* delete per worker, rather than a blanket
        # `delete_many`: the condition — free, holding nothing — is what makes
        # it safe, and the transaction is what makes the whole set atomic.
        assert resources_coll.delete_one.call_count == 1
        resources_coll.delete_many.assert_not_called()


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


class TestOccupancyIsAPartition:
    """The four counters must add up to the inventory, not overlap it.

    A container that was busy and disabled for draining counted in both
    ``active`` and ``disabled``: one worker appeared twice, and the bar that
    sums the categories for its denominator reported a total larger than the
    inventory it was describing.

    The expressions are checked rather than evaluated — this is where the
    correctness of an aggregation lives — plus a truth table over every
    status and both administrative states.
    """

    def _counters(self, hosts) -> dict:
        repo, coll = hosts
        coll.aggregate.return_value = []
        repo.inventory_snapshot()
        for stage in coll.aggregate.call_args.args[0]:
            if "$set" in stage and "active" in stage["$set"]:
                return stage["$set"]
        raise AssertionError("no counter stage found")

    def _count(self, expression: dict, resource: dict) -> int:
        """Evaluate one `count_where` expression over a single resource."""
        cond = expression["$size"]["$filter"]["cond"]
        return 1 if _evaluate(cond, resource) else 0

    @pytest.mark.parametrize("status", [s.value for s in EnumResourceStatus])
    @pytest.mark.parametrize("enabled", [True, False])
    def test_every_worker_lands_in_exactly_one_category(
        self, hosts, status, enabled
    ):
        counters = self._counters(hosts)
        resource = {"status": status, "enabled": enabled}

        landed = sum(
            self._count(counters[name], resource)
            for name in ("active", "disabled", "unavailable", "free")
        )

        assert landed == 1, f"{status}/{enabled} landed in {landed} categories"

    def test_a_draining_worker_is_in_use_not_disabled(self, hosts):
        counters = self._counters(hosts)
        draining = {"status": EnumResourceStatus.BUSY.value, "enabled": False}

        assert self._count(counters["active"], draining) == 1
        assert self._count(counters["disabled"], draining) == 0
        # And it says so, as an attribute of a worker that is working.
        assert self._count(counters["draining"], draining) == 1

    def test_a_reservation_is_not_a_running_simulation(self, hosts):
        counters = self._counters(hosts)
        reserved = {"status": EnumResourceStatus.RESERVED.value, "enabled": True}

        assert self._count(counters["active"], reserved) == 1
        assert self._count(counters["running"], reserved) == 0
        assert self._count(counters["reserved"], reserved) == 1

    def test_running_and_reserved_split_active(self, hosts):
        counters = self._counters(hosts)
        busy = {"status": EnumResourceStatus.BUSY.value, "enabled": True}

        assert self._count(counters["running"], busy) == 1
        assert self._count(counters["reserved"], busy) == 0


def _evaluate(expression, resource):
    """A very small subset of the aggregation language, over one document."""
    if isinstance(expression, str):
        if expression.startswith("$$r."):
            return resource.get(expression[4:])
        if expression.startswith("$"):
            return resource.get(expression[1:])
        return expression
    if not isinstance(expression, dict):
        return expression
    (operator, operand), = expression.items()
    if operator == "$and":
        return all(_evaluate(item, resource) for item in operand)
    if operator == "$or":
        return any(_evaluate(item, resource) for item in operand)
    if operator == "$not":
        return not _evaluate(operand, resource)
    if operator == "$eq":
        return _evaluate(operand[0], resource) == _evaluate(operand[1], resource)
    if operator == "$ne":
        return _evaluate(operand[0], resource) != _evaluate(operand[1], resource)
    if operator == "$in":
        return _evaluate(operand[0], resource) in operand[1]
    if operator == "$ifNull":
        value = _evaluate(operand[0], resource)
        return _evaluate(operand[1], resource) if value is None else value
    raise AssertionError(f"unsupported operator in this test: {operator}")


# ── Exclusivity, in the writes themselves ───────────────────────────────────

class TestQuarantineIsOneWrite:
    """R05.1: the protective write and the release cannot be separate.

    It used to be ``set_enabled(False)`` followed by the ordinary release in
    the runner's ``finally``. When the first failed, the second still ran:
    the mechanism that keeps a worker with a possibly-live JVM out of the pool
    failed, and the mechanism that puts it back succeeded. One document, one
    update — there is no interleaving that produces an eligible worker.
    """

    def test_it_disables_and_lets_go_in_a_single_update(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.quarantine(RES_ID, "could not be stopped") is True

        assert coll.update_one.call_count == 1
        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["enabled"] is False
        assert sets["status"] == EnumResourceStatus.DISABLED.value
        assert sets["current_simulation_id"] is None
        assert sets["attempt_id"] is None

    def test_the_remote_pid_is_kept(self, resources):
        """It is the only record of what may still be running in there."""
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.quarantine(RES_ID, "could not be stopped")

        assert "remote_pid" not in coll.update_one.call_args.args[1]["$set"]

    def test_it_can_insist_on_the_attempt_that_held_it(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.quarantine(RES_ID, "x", expect_attempt_id=ATTEMPT)

        assert coll.update_one.call_args.args[0] == {
            "_id": RES_ID, "attempt_id": ATTEMPT}

    def test_a_refusal_is_reported(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.quarantine(RES_ID, "x", expect_attempt_id=ATTEMPT) is False


class TestTheReservationLease:
    """R06: a margin is not a guarantee.

    The stale-reservation sweep frees a container that was claimed and never
    started, and its TTL has to outlast the slowest legitimate preparation —
    firmware out of GridFS, then up a tunnel. An upload slower than the TTL
    had its container freed underneath it while the thread went on writing to
    the directory. The thread now says "still working", and the sweep's
    ``reserved_before`` condition refuses.
    """

    def test_renewing_pushes_the_reservation_forward(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.touch_reservation(RES_ID, SIM_ID, ATTEMPT) is True

        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["reserved_at"].tzinfo is not None

    def test_only_the_holder_may_renew(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.touch_reservation(RES_ID, SIM_ID, ATTEMPT)

        flt = coll.update_one.call_args.args[0]
        assert flt["attempt_id"] == ATTEMPT
        assert flt["current_simulation_id"] == SIM_ID
        # Refused once it is busy: from then on the simulation timeout
        # governs, not this.
        assert flt["status"] == EnumResourceStatus.RESERVED.value

    def test_a_refusal_means_the_reservation_is_gone(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.touch_reservation(RES_ID, SIM_ID, ATTEMPT) is False


class TestTwoNamesForOneWorker:
    """R08: discovery cannot produce an alias, a person can.

    Two registrations pointing at the same address, port and directory mean
    two reservations for one run directory — and then each run's defensive
    cleanup deletes the other's inputs, while the exclusivity the reservation
    provides is held on the wrong document.
    """

    def _two(self, first, second, hosts):
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[HOSTS_COLLECTION].find.return_value = hosts
        db[RESOURCES_COLLECTION].find.return_value = [first]
        return repo, second

    def test_the_same_target_under_another_name_is_found(self):
        host = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                "access": {"address": "10.0.0.7"},
                "container_access": {"port": 22}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "172.18.0.5", "port": 22},
                    "remote_dir": "/opt/contiki-ng/tools/cooja"}
        candidate = {"name": "cooja1-again", "host_id": HOST_ID,
                     "endpoint": {"address": "172.18.0.5", "port": 22},
                     "remote_dir": "/opt/contiki-ng/tools/cooja"}
        repo, candidate = self._two(existing, candidate, [host])

        clash = repo.conflicting_target(host, candidate)

        assert clash is not None and clash["name"] == "cooja1"

    def test_a_different_port_is_a_different_worker(self):
        host = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                "access": {"address": "10.0.0.7"}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "10.0.0.7", "port": 2231}}
        candidate = {"name": "cooja2", "host_id": HOST_ID,
                     "endpoint": {"address": "10.0.0.7", "port": 2232}}
        repo, candidate = self._two(existing, candidate, [host])

        assert repo.conflicting_target(host, candidate) is None

    def test_editing_a_worker_does_not_clash_with_itself(self):
        host = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                "access": {"address": "10.0.0.7"}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "172.18.0.5", "port": 22}}
        repo, candidate = self._two(existing, dict(existing), [host])

        assert repo.conflicting_target(
            host, candidate, exclude_id=RES_ID) is None

    def test_a_worker_with_no_address_points_nowhere(self):
        host = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                "access": {"address": "10.0.0.7"}}
        repo, candidate = self._two(
            {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1"},
            {"name": "cooja2", "host_id": HOST_ID}, [host])

        assert repo.conflicting_target(host, candidate) is None


class TestWhereTheUniquenessContractEnds:
    """F5.3: what "one registration per physical worker" does and does not mean.

    Two mechanisms, with different strengths, and the difference matters:

    * **Within one host** the unique partial index decides, so two concurrent
      registrations cannot both land — the loser gets a duplicate-key error.
    * **Across hosts** the host's own address is part of the identity and is
      not in this collection, so no index can express it. ``conflicting_target``
      reads and the caller then inserts, which makes it a *check* rather than
      a lock.

    And neither resolves an alias. The fingerprint compares address *strings*:
    one machine reached as ``10.0.0.7`` and as ``vm7.local`` is two
    fingerprints, and both registrations are accepted. That is the stated
    boundary of the contract — asserted here so that it is a documented limit
    rather than something a later reader assumes away.
    """

    def _repo(self, existing, hosts):
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[HOSTS_COLLECTION].find.return_value = hosts
        db[RESOURCES_COLLECTION].find.return_value = existing
        return repo

    OTHER_HOST_ID = ObjectId("507f1f77bcf86cd799430099")

    def test_the_same_machine_registered_twice_is_covered(self):
        """Registering one machine as two hosts is reachable through the API,
        so a worker pointing at the same container through both is caught."""
        first = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                 "access": {"address": "10.0.0.7"},
                 "container_access": {"port": 22}}
        second = {"_id": self.OTHER_HOST_ID, "name": "vm7-again",
                  "type": "remote", "access": {"address": "10.0.0.7"},
                  "container_access": {"port": 22}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "172.18.0.5", "port": 22},
                    "remote_dir": "/opt/contiki-ng/tools/cooja"}
        candidate = {"name": "cooja1-b", "host_id": self.OTHER_HOST_ID,
                     "endpoint": {"address": "172.18.0.5", "port": 22},
                     "remote_dir": "/opt/contiki-ng/tools/cooja"}
        repo = self._repo([existing], [first, second])

        clash = repo.conflicting_target(second, candidate)

        assert clash is not None and clash["name"] == "cooja1"

    def test_two_real_machines_with_the_same_container_address_are_distinct(self):
        """The common case, and the reason the host address is in the identity:
        ``172.18.0.5`` on two machines is two containers."""
        first = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                 "access": {"address": "10.0.0.7"},
                 "container_access": {"port": 22}}
        second = {"_id": self.OTHER_HOST_ID, "name": "vm8", "type": "remote",
                  "access": {"address": "10.0.0.8"},
                  "container_access": {"port": 22}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "172.18.0.5", "port": 22},
                    "remote_dir": "/opt/contiki-ng/tools/cooja"}
        candidate = {"name": "cooja1", "host_id": self.OTHER_HOST_ID,
                     "endpoint": {"address": "172.18.0.5", "port": 22},
                     "remote_dir": "/opt/contiki-ng/tools/cooja"}
        repo = self._repo([existing], [first, second])

        assert repo.conflicting_target(second, candidate) is None

    def test_an_aliased_host_address_is_not_detected(self):
        """**The boundary.** One machine under two names is two fingerprints.

        Nothing here resolves DNS, and resolving it would make the answer
        depend on a lookup at registration time that can differ from the
        lookup at connection time. The contract is therefore stated over the
        configured strings, and this is the case it does not cover — kept as
        a test so the limitation is visible next to the coverage.
        """
        first = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                 "access": {"address": "10.0.0.7"},
                 "container_access": {"port": 22}}
        aliased = {"_id": self.OTHER_HOST_ID, "name": "vm7-by-name",
                   "type": "remote", "access": {"address": "vm7.local"},
                   "container_access": {"port": 22}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "172.18.0.5", "port": 22},
                    "remote_dir": "/opt/contiki-ng/tools/cooja"}
        candidate = {"name": "cooja1", "host_id": self.OTHER_HOST_ID,
                     "endpoint": {"address": "172.18.0.5", "port": 22},
                     "remote_dir": "/opt/contiki-ng/tools/cooja"}
        repo = self._repo([existing], [first, aliased])

        assert repo.conflicting_target(aliased, candidate) is None, (
            "documented limitation, not an oversight: see the class docstring")

    def test_a_published_port_alias_is_not_detected_either(self):
        """The same container reached directly and through a published port.

        Two addresses, two ports, one container. Supported by design — it is
        how a host whose SSH user cannot reach Docker is configured — and
        therefore not distinguishable from two real workers.
        """
        host = {"_id": HOST_ID, "name": "vm7", "type": "remote",
                "access": {"address": "10.0.0.7"},
                "container_access": {"port": 22}}
        existing = {"_id": RES_ID, "host_id": HOST_ID, "name": "cooja1",
                    "endpoint": {"address": "172.18.0.5", "port": 22},
                    "remote_dir": "/opt/contiki-ng/tools/cooja"}
        candidate = {"name": "cooja1-published", "host_id": HOST_ID,
                     "endpoint": {"address": "127.0.0.1", "port": 2231},
                     "remote_dir": "/opt/contiki-ng/tools/cooja"}
        repo = self._repo([existing], [host])

        assert repo.conflicting_target(host, candidate) is None

    def test_the_index_cannot_serialise_the_cross_host_case(self):
        """Stated as a property of the index itself, so the asymmetry between
        the two mechanisms is not something a reader has to infer."""
        connection, db = make_connection()
        CoojaResourceRepository(connection)

        calls = [c for c in db[RESOURCES_COLLECTION].create_index.call_args_list
                 if c.kwargs.get("name") == "idx_resources_target"]
        assert len(calls) == 1
        keys = [field for field, _ in calls[0].args[0]]
        assert keys == ["host_id", "endpoint.address", "endpoint.port",
                        "remote_dir"]
        assert "host_id" in keys, (
            "scoped to one host; the host's address is in another collection")
        assert calls[0].kwargs["unique"] is True


class TestCreationRefusesOnAWithdrawingHost:
    """The other half of R09, checked in the collection being written.

    The orphan came from a caller that had read the host *before* the
    withdrawal began, so a check anywhere earlier than the write is a check
    at the wrong time.
    """

    def _withdrawing(self):
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[HOSTS_COLLECTION].find_one.return_value = {
            "_id": HOST_ID, "withdrawn_at": utc_now()}
        return repo, db

    def test_a_manual_worker_is_refused(self):
        repo, _ = self._withdrawing()

        with pytest.raises(HostWithdrawing):
            repo.insert_manual({"name": "cooja9", "host_id": HOST_ID})

    def test_discovery_is_refused(self):
        repo, _ = self._withdrawing()

        with pytest.raises(HostWithdrawing):
            repo.sync_discovered(
                HOST_ID, [{"id": "abc", "name": "cooja9"}],
                remote_dir="/opt/cooja")

    def test_the_bootstrap_seed_is_skipped_rather_than_raised(self):
        """The one creation path that must not raise.

        It runs at every start-up, before anything else. A withdrawal left
        set by an interrupted removal would stop the master-node from
        starting at all — a worse failure than the orphan the guard is for.
        """
        repo, db = self._withdrawing()

        assert repo.insert_seeded({"host_id": HOST_ID, "name": "cooja1"}) is False
        db[RESOURCES_COLLECTION].update_one.assert_not_called()

    def test_a_host_in_service_is_not_refused(self):
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[RESOURCES_COLLECTION].insert_one.return_value = MagicMock(
            inserted_id=RES_ID)

        assert repo.insert_manual(
            {"name": "cooja9", "host_id": HOST_ID}) == RES_ID

    def test_a_host_that_no_longer_exists_is_refused(self):
        """The check also covers a host deleted a moment ago, which the
        earlier version — a count of *withdrawing* hosts — did not."""
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[HOSTS_COLLECTION].find_one.return_value = None

        with pytest.raises(HostWithdrawing):
            repo.insert_manual({"name": "cooja9", "host_id": HOST_ID})

    def test_creation_writes_the_host_so_a_removal_conflicts(self):
        """Reading it refuses a withdrawal already visible. Writing it is what
        makes a removal that starts a moment later abort — a transaction's
        snapshot cannot see a document the other side inserts."""
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[RESOURCES_COLLECTION].insert_one.return_value = MagicMock(
            inserted_id=RES_ID)

        repo.insert_manual({"name": "cooja9", "host_id": HOST_ID})

        bumped = db[HOSTS_COLLECTION].update_one.call_args
        assert "resource_epoch" in bumped.args[1]["$inc"]
        assert bumped.kwargs["session"] is FAKE_SESSION


class TestThePortOverrideCanBeRemoved:
    """R17: an override is removed by the *absence* of the key.

    An older version of the form wrote the port in both ``endpoint`` and
    ``connection``, and the override wins in the connection plan — so editing
    such a worker changed the endpoint, reported the new port and went on
    connecting to the old one. ``$set`` cannot express absence, and a stored
    ``null`` is not the same thing: it reads as the operator having configured
    something, which is one of the signals that decides whether Docker's view
    of this worker applies at all.
    """

    def test_it_unsets_rather_than_nulling(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.clear_connection_port(RES_ID) is True

        update = coll.update_one.call_args.args[1]
        assert update["$unset"] == {"connection.port": ""}
        assert "connection.port" not in update.get("$set", {})


class TestHeldWorkers:
    """A worker holding something unresolved is not released by anyone.

    Being disabled kept the *dispatcher* off a quarantined worker, but not the
    reconciliation: that pass found the simulation in a terminal state,
    concluded nothing was running and called ``force_release`` with no probe.
    So a quarantine whose write had failed was undone by the next health
    cycle — exactly when the database came back. The decision has to outlive
    the process that made it, which means it has to be in the document.
    """

    def test_the_ordinary_release_refuses(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = None

        repo.release(RES_ID, SIM_ID, ATTEMPT, container_state_counts=True)

        assert coll.find_one_and_update.call_args.args[0]["held_for"] is None

    def test_the_forced_release_refuses_too(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.force_release(
            RES_ID, "recovered", container_state_counts=True) is False
        assert coll.update_one.call_args.args[0]["held_for"] is None

    def test_a_reservation_cannot_be_granted(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = None

        repo.reserve(RES_ID, SIM_ID, ATTEMPT)

        assert coll.find_one_and_update.call_args.args[0]["held_for"] is None

    def test_a_measurement_cannot_promote_it(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=0)

        repo.mark_available(RES_ID, "ready")

        assert coll.update_one.call_args.args[0]["held_for"] is None

    def test_the_quarantine_records_what_is_unresolved(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.quarantine(RES_ID, "could not be stopped", held_for={
            "kind": "termination_unknown", "remote_pid": 4242,
            "remote_dir": "/opt/cooja"})

        held = coll.update_one.call_args.args[1]["$set"]["held_for"]
        assert held["kind"] == "termination_unknown"
        assert held["remote_pid"] == 4242
        assert held["remote_dir"] == "/opt/cooja"
        assert held["at"].tzinfo is not None

    def test_holding_does_not_touch_the_reservation(self, resources):
        """The attempt is over but the container is not accounted for."""
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.hold(RES_ID, "result not collected", {"kind": "collection_pending"})

        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["enabled"] is False
        assert "current_simulation_id" not in sets
        assert "status" not in sets

    def test_clearing_requires_the_hold_to_exist(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.clear_held_for(RES_ID)

        flt = coll.update_one.call_args.args[0]
        assert flt["held_for"] == {"$ne": None}
        update = coll.update_one.call_args.args[1]
        assert update["$unset"] == {"held_for": ""}
        # Back in service, but **offline**: the next health cycle measures it.
        # Clearing a hold is not evidence that the worker works.
        assert update["$set"]["enabled"] is True
        assert update["$set"]["status"] == EnumResourceStatus.OFFLINE.value


class TestAStaleMeasurementCannotSchedule:
    """R22: the version has to be a *condition*, not a record.

    An access edit takes the idle workers out of the pool and clears what was
    measured about them. Two paths got past that: a probe that started before
    the edit and published after it, and a worker that was busy during the
    edit and came back on its old measurement when it was released.
    """

    def test_a_reservation_requires_the_current_configuration(self, resources):
        repo, coll = resources
        coll.find_one_and_update.return_value = None

        repo.reserve(RES_ID, SIM_ID, ATTEMPT, expect_config_version=3)

        assert coll.find_one_and_update.call_args.args[0]["$expr"] == {
            "$eq": [{"$ifNull": ["$probe.config_version", 0]}, 3]}

    def test_a_probe_written_before_this_existed_still_schedules(self, resources):
        """Through ``$ifNull``, not an equality on the field.

        An equality does not match a *missing* field, so on an existing
        deployment every worker would have become unschedulable until it was
        measured again — and `probe_is_fresh` reads a missing value as zero,
        so a worker that had just been probed would never be re-probed. An
        absent value means version zero here as it does everywhere else.
        """
        repo, coll = resources
        coll.find_one_and_update.return_value = None

        repo.reserve(RES_ID, SIM_ID, ATTEMPT, expect_config_version=0)

        expr = coll.find_one_and_update.call_args.args[0]["$expr"]
        assert expr["$eq"][0] == {"$ifNull": ["$probe.config_version", 0]}
        assert "probe.config_version" not in coll.find_one_and_update.call_args.args[0]

    def test_the_promotion_records_the_version_it_measured(self, resources):
        repo, coll = resources
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.mark_available(RES_ID, "ready", config_version=3)

        assert coll.update_one.call_args.args[1]["$set"][
            "probe.config_version"] == 3

    def test_a_release_after_an_edit_lands_offline(self):
        """The one path an access edit could not reach: a busy worker is left
        alone on purpose, and used to come back available on a measurement of
        how it was reached before."""
        branches = released_status_expression(
            container_state_counts=False, config_version=2)["$switch"]["branches"]
        stale = [b for b in branches
                 if "probe.config_version" in str(b["case"])]
        assert stale, branches
        assert stale[0]["then"] == EnumResourceStatus.OFFLINE.value

    def test_without_a_version_nothing_changes(self):
        """Callers that have no host record keep the previous behaviour."""
        branches = released_status_expression(
            container_state_counts=False)["$switch"]["branches"]
        assert not [b for b in branches
                    if "probe.config_version" in str(b["case"])]


class TestPinningAHostKey:
    """R11/P08: the *first* contact is the one that sticks.

    The pin is what every later session is checked against, so an
    unconditional write turns trust-on-first-use into trust-on-last-use: two
    paths reaching a host at the same time would each pin their own key, and
    whichever wrote second would make the other's accepted session the
    divergent one — silently.

    The condition is in the filter, which is the only place it can be atomic,
    so that is what is asserted here. The composition — a loser session
    closing — is in ``master-node/tests/test_transport.py``.
    """

    def test_the_write_only_matches_a_host_with_no_pin(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.pin_host_key(HOST_ID, "SHA256:abc") is True

        flt = coll.update_one.call_args.args[0]
        assert flt["_id"] == HOST_ID
        assert {"access.host_key_fingerprint": None} in flt["$or"]
        assert {"access.host_key_fingerprint": {"$exists": False}} in flt["$or"]
        update = coll.update_one.call_args.args[1]
        assert update["$set"]["access.host_key_fingerprint"] == "SHA256:abc"

    def test_a_host_already_pinned_is_not_overwritten(self, hosts):
        """``modified_count == 0`` is how the repository reports losing the
        race, and the caller has to be able to tell that from success."""
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.pin_host_key(HOST_ID, "SHA256:second") is False

    def test_an_empty_fingerprint_is_refused_without_a_write(self, hosts):
        repo, coll = hosts

        assert repo.pin_host_key(HOST_ID, "") is False
        coll.update_one.assert_not_called()

    def test_an_unusable_id_is_refused_without_a_write(self, hosts):
        repo, coll = hosts

        assert repo.pin_host_key("not-an-objectid", "SHA256:abc") is False
        coll.update_one.assert_not_called()


class TestTheOccupancyEstimate:
    """How long a worker holds a container, as a scheduling input.

    Only a *speed-aware* policy reads it, so it is advisory by construction:
    a stale or missing value costs a slightly worse placement and nothing
    else. What it must not do is lie in the direction that makes a bad worker
    look good.
    """

    def test_the_first_sample_becomes_the_mean(self, resources):
        repo, coll = resources
        coll.find_one.return_value = {"_id": RES_ID}

        assert repo.record_duration(RES_ID, 120.0) == 120.0

        update = coll.update_one.call_args.args[1]["$set"]
        assert update["mean_duration_sec"] == 120.0
        assert update["duration_samples"] == 1

    def test_later_samples_are_weighted(self, resources):
        """Exponentially, not a lifetime average: a host's speed changes with
        the load on the machine it shares, and an average over every run it
        ever did would keep reporting last month's."""
        repo, coll = resources
        coll.find_one.return_value = {
            "_id": RES_ID, "mean_duration_sec": 100.0, "duration_samples": 9}

        result = repo.record_duration(RES_ID, 200.0)

        assert result == pytest.approx(120.0), "0.8 * 100 + 0.2 * 200"
        assert coll.update_one.call_args.args[1]["$set"][
            "duration_samples"] == 10

    def test_one_slow_run_does_not_move_it_far(self, resources):
        """A simulation's duration varies more between individuals than
        between machines, so a single outlier must not decide a placement."""
        repo, coll = resources
        coll.find_one.return_value = {
            "_id": RES_ID, "mean_duration_sec": 111.0, "duration_samples": 40}

        result = repo.record_duration(RES_ID, 1100.0)

        assert result < 310.0, "one 10x outlier moves it by a fifth, not to it"

    def test_a_nonsensical_duration_is_refused(self, resources):
        repo, coll = resources

        assert repo.record_duration(RES_ID, 0) is None
        assert repo.record_duration(RES_ID, -5.0) is None
        coll.update_one.assert_not_called()

    def test_an_unknown_worker_writes_nothing(self, resources):
        repo, coll = resources
        coll.find_one.return_value = None

        assert repo.record_duration(RES_ID, 120.0) is None
        coll.update_one.assert_not_called()

    def test_an_unusable_id_writes_nothing(self, resources):
        repo, coll = resources

        assert repo.record_duration("not-an-objectid", 120.0) is None
        coll.update_one.assert_not_called()


class TestOrderingTheFreeContainersOfAHost:
    """Which container of the chosen host takes the next job.

    Two orderings for two policies, and the dispatcher picks between them
    from the policy — because choosing the host that finishes soonest and
    then handing it the slowest container would undo half the decision.
    """

    def test_by_default_it_spreads_over_the_containers(self, resources):
        repo, coll = resources

        repo.find_free_by_host(HOST_ID)

        assert coll.find.call_args.kwargs["sort"] == [
            ("last_check", 1), ("name", 1)]

    def test_a_speed_aware_policy_asks_for_the_fastest_first(self, resources):
        repo, coll = resources

        repo.find_free_by_host(HOST_ID, prefer_fastest=True)

        assert coll.find.call_args.kwargs["sort"] == [
            ("mean_duration_sec", 1), ("name", 1)]

    def test_either_way_only_free_enabled_containers_are_offered(
            self, resources):
        repo, coll = resources

        for prefer in (False, True):
            repo.find_free_by_host(HOST_ID, prefer_fastest=prefer)
            flt = coll.find.call_args.args[0]
            assert flt == {
                "host_id": HOST_ID, "enabled": True,
                "status": EnumResourceStatus.AVAILABLE.value}


class TestTheToggleAndTheBarAgree:
    """The reported bug: switching a worker back on left the bar saying
    *disabled*.

    Two facts were being conflated. ``enabled`` is the operator's **decision**
    and changes the instant the toggle is used; ``status`` is a
    **measurement** and only changes when the next health cycle reaches the
    container. The occupancy partition keyed its administrative bucket on
    both — `enabled == False` **or** `status == disabled` — so a worker
    switched back on stayed in the grey "disabled" segment for up to a cycle,
    next to a toggle that said it was on.
    """

    def _counters(self, hosts) -> dict:
        repo, coll = hosts
        coll.aggregate.return_value = []
        repo.inventory_snapshot()
        for stage in coll.aggregate.call_args.args[0]:
            if "$set" in stage and "active" in stage["$set"]:
                return stage["$set"]
        raise AssertionError("no counter stage found")

    def _bucket(self, counters: dict, resource: dict) -> str:
        landed = [name for name in ("active", "disabled", "unavailable", "free")
                  if _evaluate(counters[name]["$size"]["$filter"]["cond"],
                               resource)]
        assert len(landed) == 1, f"{resource} landed in {landed}"
        return landed[0]

    def test_a_worker_switched_back_on_is_no_longer_counted_disabled(
            self, hosts):
        """The defect, stated as the state it leaves behind."""
        counters = self._counters(hosts)

        assert self._bucket(counters, {"status": "disabled",
                                       "enabled": True}) == "unavailable"

    def test_a_worker_switched_off_is(self, hosts):
        counters = self._counters(hosts)

        assert self._bucket(counters, {"status": "disabled",
                                       "enabled": False}) == "disabled"

    def test_the_bucket_follows_the_decision_not_the_measurement(self, hosts):
        """Whatever the stale status says, the flag decides."""
        counters = self._counters(hosts)

        for status in ("available", "offline", "error", "disabled"):
            assert self._bucket(
                counters, {"status": status, "enabled": False}) == "disabled"

    def test_a_document_with_no_flag_is_treated_as_in_the_pool(self, hosts):
        """Written before the flag existed. A missing value compares equal to
        neither true nor false, so without this it would fall out of every
        bucket and the segments would stop adding up."""
        counters = self._counters(hosts)

        assert self._bucket(counters, {"status": "available"}) == "free"
        assert self._bucket(counters, {"status": "offline"}) == "unavailable"

    def test_an_unanticipated_status_still_lands_somewhere(self, hosts):
        """``unavailable`` is the complement, not a list. Written as a list,
        a status nobody enumerated falls into no bucket and the bar's
        denominator stops matching the inventory."""
        counters = self._counters(hosts)

        assert self._bucket(
            counters, {"status": "something-new", "enabled": True}
        ) == "unavailable"


class TestReEnablingLeavesTheAdministrativeStatus:
    """``status`` is not a second copy of ``enabled``.

    Re-enabling used to set the flag and leave the status at ``disabled``,
    with whatever message the last check had written — so the worker read as
    *enabled, disabled, ready* all at once until the next cycle.
    """

    @staticmethod
    def _counted(coll):
        coll.update_one.return_value = MagicMock(
            matched_count=1, modified_count=1)
        return coll

    def test_a_disabled_worker_moves_to_offline(self, resources):
        repo, coll = resources
        self._counted(coll)

        repo.set_enabled(RES_ID, True)

        flt, update = coll.update_one.call_args_list[0].args
        assert flt == {"_id": RES_ID,
                       "status": EnumResourceStatus.DISABLED.value}
        assert update["$set"]["status"] == EnumResourceStatus.OFFLINE.value
        assert "waiting to be measured" in update["$set"]["last_check_message"]

    def test_a_worker_measured_offline_or_broken_is_left_alone(self, resources):
        """Re-enabling says nothing about what was measured, so it must not
        overwrite a measurement."""
        repo, coll = resources
        self._counted(coll)

        repo.set_enabled(RES_ID, True)

        flt = coll.update_one.call_args_list[0].args[0]
        assert flt["status"] == EnumResourceStatus.DISABLED.value, (
            "scoped to the administrative status alone")

    def test_disabling_still_parks_it(self, resources):
        repo, coll = resources
        self._counted(coll)

        repo.set_enabled(RES_ID, False)

        flt, update = coll.update_one.call_args_list[0].args
        assert flt["status"] == {"$in": _REASSIGNABLE_FOR_TEST}
        assert update["$set"]["status"] == EnumResourceStatus.DISABLED.value

    def test_disabling_does_not_evict_a_running_simulation(self, resources):
        """``busy`` and ``reserved`` are absent from the reassignable set, so
        the status write cannot touch a container that is working."""
        repo, coll = resources
        self._counted(coll)

        repo.set_enabled(RES_ID, False)

        allowed = coll.update_one.call_args_list[0].args[0]["status"]["$in"]
        assert EnumResourceStatus.BUSY.value not in allowed
        assert EnumResourceStatus.RESERVED.value not in allowed

    def test_the_flag_itself_is_always_written(self, resources):
        repo, coll = resources

        for enabled in (True, False):
            coll.reset_mock()
            self._counted(coll)
            repo.set_enabled(RES_ID, enabled)
            last = coll.update_one.call_args_list[-1].args[1]["$set"]
            assert last["enabled"] is enabled


class TestReEnablingAHostLeavesItToo:
    def test_a_disabled_host_moves_to_offline(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(
            matched_count=1, modified_count=1)

        repo.set_enabled(HOST_ID, True)

        flt, update = coll.update_one.call_args_list[0].args
        assert flt == {"_id": HOST_ID, "status": EnumHostStatus.DISABLED.value}
        assert update["$set"]["status"] == EnumHostStatus.OFFLINE.value

    def test_disabling_a_host_still_parks_it(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(
            matched_count=1, modified_count=1)

        repo.set_enabled(HOST_ID, False)

        update = coll.update_one.call_args_list[-1].args[1]["$set"]
        assert update["status"] == EnumHostStatus.DISABLED.value
        assert update["enabled"] is False


class TestTwoMachinesBehindOneGateway:
    """Reported: changing a vm7 worker's port was refused with a 409 naming a
    vm8 worker.

    Both VMs are reached through the same gateway address on different
    forwarded SSH ports (``gateway:2197`` and ``gateway:2198``). The physical
    identity keyed the host on its address alone, so the two machines looked
    like one, and any pair of their containers with the same internal port
    looked like one container registered twice. The connection plan uses
    address **and** port for hop 1; the identity has to use the same rule, or
    it compares something nobody connects to.
    """

    OTHER_HOST_ID = ObjectId("507f1f77bcf86cd799430098")

    def _repo(self, existing, hosts):
        connection, db = make_connection()
        repo = CoojaResourceRepository(connection)
        db[HOSTS_COLLECTION].find.return_value = hosts
        db[RESOURCES_COLLECTION].find.return_value = existing
        return repo

    def _host(self, host_id, name, port):
        return {"_id": host_id, "name": name, "type": "remote",
                "access": {"address": "gateway.example.edu", "port": port},
                "container_access": {"port": 2222}}

    def _worker(self, host_id, name, port, _id=None):
        doc = {"name": name, "host_id": host_id,
               "endpoint": {"address": "localhost", "port": port},
               "remote_dir": "/opt/contiki-ng/tools/cooja"}
        if _id is not None:
            doc["_id"] = _id
        return doc

    def test_the_same_internal_port_on_two_forwarded_vms_is_two_workers(self):
        vm7 = self._host(HOST_ID, "vm7", 2197)
        vm8 = self._host(self.OTHER_HOST_ID, "vm8", 2198)
        existing = self._worker(self.OTHER_HOST_ID, "vm8-cooja1", 2222, RES_ID)
        repo = self._repo([existing], [vm7, vm8])

        clash = repo.conflicting_target(
            vm7, self._worker(HOST_ID, "vm7-cooja1", 2222))

        assert clash is None

    def test_the_same_gateway_and_the_same_port_is_still_one_machine(self):
        """The coverage the identity exists for is kept: one SSH endpoint
        registered as two hosts is one machine."""
        first = self._host(HOST_ID, "vm7", 2197)
        again = self._host(self.OTHER_HOST_ID, "vm7-again", 2197)
        existing = self._worker(HOST_ID, "vm7-cooja1", 2222, RES_ID)
        repo = self._repo([existing], [first, again])

        clash = repo.conflicting_target(
            again, self._worker(self.OTHER_HOST_ID, "dup", 2222))

        assert clash is not None and clash["name"] == "vm7-cooja1"

    def test_an_unset_port_means_22_as_it_does_for_the_connection(self):
        explicit = self._host(HOST_ID, "a", 22)
        implicit = self._host(self.OTHER_HOST_ID, "b", None)
        existing = self._worker(HOST_ID, "w1", 2222, RES_ID)
        repo = self._repo([existing], [explicit, implicit])

        clash = repo.conflicting_target(
            implicit, self._worker(self.OTHER_HOST_ID, "w2", 2222))

        assert clash is not None, "port 22 and no port are the same endpoint"

    def test_the_host_address_is_compared_without_case(self):
        """A DNS name is case-insensitive; ``Gateway`` and ``gateway`` are
        the same machine."""
        upper = self._host(HOST_ID, "a", 2197)
        upper["access"]["address"] = "GATEWAY.example.edu"
        lower = self._host(self.OTHER_HOST_ID, "b", 2197)
        existing = self._worker(HOST_ID, "w1", 2222, RES_ID)
        repo = self._repo([existing], [upper, lower])

        assert repo.conflicting_target(
            lower, self._worker(self.OTHER_HOST_ID, "w2", 2222)) is not None

    def test_editing_a_worker_that_already_matched_is_no_longer_blocked(self):
        """vm7-cooja2 and vm8-cooja2 were both ``localhost:2223``: under the
        old identity they collided, so any edit of either was refused."""
        vm7 = self._host(HOST_ID, "vm7", 2197)
        vm8 = self._host(self.OTHER_HOST_ID, "vm8", 2198)
        vm7_w = self._worker(HOST_ID, "vm7-cooja2", 2223, RES_ID)
        vm8_w = self._worker(self.OTHER_HOST_ID, "vm8-cooja2", 2223,
                             ObjectId("507f1f77bcf86cd799430077"))
        repo = self._repo([vm7_w, vm8_w], [vm7, vm8])

        assert repo.conflicting_target(vm7, dict(vm7_w), exclude_id=RES_ID) is None


# ── Backoff, inconclusive checks and the last known address ─────────────────

class TestHostCheckBookkeeping:
    """What the health cycle writes about a host besides its status."""

    def test_a_check_result_clears_the_backoff_unless_given_one(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_check_result(HOST_ID, status=EnumHostStatus.ONLINE, reset_failures=True)

        # Written as null rather than left alone: a host that recovers goes
        # back to the normal pace at once.
        assert coll.update_one.call_args.args[1]["$set"]["backoff_until"] is None

    def test_a_failed_check_carries_its_backoff(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)
        until = utc_now() + timedelta(minutes=2)

        repo.mark_check_result(HOST_ID, status=EnumHostStatus.OFFLINE, backoff_until=until)

        assert coll.update_one.call_args.args[1]["$set"]["backoff_until"] == until

    def test_an_inconclusive_check_leaves_the_verdict_alone(self, hosts):
        """A resolver on the master-node failing is not the host failing."""
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_check_inconclusive(HOST_ID, "not measured: DNS")

        flt, update = coll.update_one.call_args.args[:2]
        assert flt["enabled"] is True
        assert set(update) == {"$set"}
        assert set(update["$set"]) == {"last_check", "last_check_message"}

    def test_an_address_is_written_only_when_it_changed(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.record_resolved_address(HOST_ID, "lab.example", "10.0.0.1") is True

        flt, update = coll.update_one.call_args.args[:2]
        assert {"resolved_address.ip": {"$ne": "10.0.0.1"}} in flt["$or"]
        assert {"resolved_address.name": {"$ne": "lab.example"}} in flt["$or"]
        stored = update["$set"]["resolved_address"]
        assert stored["name"] == "lab.example" and stored["ip"] == "10.0.0.1"

    def test_an_unchanged_address_is_not_rewritten(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.record_resolved_address(HOST_ID, "lab.example", "10.0.0.1") is False

    def test_an_access_edit_clears_the_backoff(self, hosts):
        # It was earned by the configuration that just changed.
        repo, coll = hosts
        coll.find_one_and_update.return_value = {"config_version": 2}

        repo.bump_config_version(HOST_ID)

        update = coll.find_one_and_update.call_args.args[1]
        assert update["$set"]["backoff_until"] is None

    def test_re_enabling_clears_the_backoff(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.set_enabled(HOST_ID, True)

        assert coll.update_one.call_args.args[1]["$set"]["backoff_until"] is None

    def test_disabling_does_not_touch_it(self, hosts):
        repo, coll = hosts
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.set_enabled(HOST_ID, False)

        assert "backoff_until" not in coll.update_one.call_args.args[1]["$set"]

    def test_neither_can_be_written_by_a_client(self, hosts):
        repo, coll = hosts

        assert repo.update(HOST_ID, {
            "backoff_until": None,
            "resolved_address": {"name": "x", "ip": "10.6.6.6"},
        }) is False
        coll.update_one.assert_not_called()


class TestMasterStatus:
    @pytest.fixture
    def master(self):
        connection, db = make_connection()
        return MasterStatusRepository(connection), db[MASTER_STATUS_COLLECTION]

    def test_the_heartbeat_is_one_upserted_document(self, master):
        repo, coll = master

        repo.heartbeat(resolver={"ok": False}, cycle_sec=1.23456)

        flt, update = coll.update_one.call_args.args[:2]
        assert flt == {"_id": MasterStatusRepository.DOC_ID}
        assert coll.update_one.call_args.kwargs["upsert"] is True
        assert update["$set"]["resolver"] == {"ok": False}
        assert update["$set"]["cycle_sec"] == 1.235
        assert isinstance(update["$set"]["heartbeat_at"], datetime)

    def test_a_heartbeat_without_news_of_the_resolver_keeps_the_last(self, master):
        repo, coll = master

        repo.heartbeat()

        assert "resolver" not in coll.update_one.call_args.args[1]["$set"]

    def test_a_restart_is_recorded_and_the_history_bounded(self, master):
        repo, coll = master

        repo.record_restart("DNS")

        update = coll.update_one.call_args.args[1]
        assert update["$set"]["last_restart"]["reason"] == "DNS"
        assert update["$push"]["restarts"]["$slice"] == -MasterStatusRepository.RESTARTS_KEPT


# ── Physical machines ───────────────────────────────────────────────────────

class TestMachineOf:
    def test_port_forwarded_vms_are_one_machine(self):
        vm1 = {"type": "remote", "access": {"address": "Andromeda.example", "port": 2191}}
        vm2 = {"type": "remote", "access": {"address": "andromeda.example ", "port": 2192}}

        assert machine_of(vm1) == machine_of(vm2) == "andromeda.example"

    def test_an_explicit_label_wins(self):
        # Distinct machines behind one NAT gateway share its address.
        assert machine_of({"type": "remote", "machine": " rack-2 ",
                           "access": {"address": "gw.example"}}) == "rack-2"

    def test_the_master_nodes_own_host_is_local(self):
        assert machine_of({"type": "local", "name": "local"}) == "local"

    def test_a_host_with_no_address_is_its_own_machine(self):
        assert machine_of({"type": "remote", "name": "lab-01"}) == "lab-01"


class TestMachineLimits:
    @pytest.fixture
    def machines(self):
        connection, db = make_connection()
        return MachineRepository(connection), db[MACHINES_COLLECTION]

    def test_only_machines_with_a_limit_are_listed(self, machines):
        repo, coll = machines
        coll.find.return_value = [{"_id": "andromeda", "max_concurrent": 5}]

        assert repo.limits() == {"andromeda": 5}
        assert coll.find.call_args.args[0] == {"max_concurrent": {"$type": "number"}}

    def test_a_limit_is_upserted_by_label(self, machines):
        repo, coll = machines

        repo.set_limit("andromeda", 5)

        flt, update = coll.update_one.call_args.args[:2]
        assert flt == {"_id": "andromeda"}
        assert update["$set"]["max_concurrent"] == 5
        assert coll.update_one.call_args.kwargs["upsert"] is True

    def test_clearing_it_stores_null(self, machines):
        repo, coll = machines

        repo.set_limit("andromeda", None)

        assert coll.update_one.call_args.args[1]["$set"]["max_concurrent"] is None

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_limit_below_one_is_refused(self, machines, bad):
        repo, _ = machines

        with pytest.raises(ValueError):
            repo.set_limit("andromeda", bad)


class TestSnapshotCarriesWorkerDetail:
    def test_the_planner_gets_durations_and_start_times(self, hosts):
        repo, coll = hosts
        coll.aggregate.return_value = []

        repo.inventory_snapshot()

        pipeline = coll.aggregate.call_args.args[0]
        lookup = pipeline[0]["$lookup"]["pipeline"][0]["$project"]
        fields = pipeline[1]["$set"]
        assert lookup["started_at"] == 1
        assert "free_durations" in fields and "occupied_workers" in fields
