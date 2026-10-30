"""Tests for SimulationRepository query scoping.

``individual_id`` is a chromosome hash, not an identity. The same chromosome
is produced again by later experiments, and every run of it accumulates under
that one id, so a lookup by individual alone answers with simulations the
caller never asked about. One pilot's individual panel showed nine simulations
where two belonged to the experiment on screen.

The connection is mocked in the style of ``test_infra_repository.py``: what
these assert is the *filter sent to MongoDB*, which is where the scoping
actually lives.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.simulation import SimulationRepository

EXP_ID = ObjectId("507f1f77bcf86cd799440001")
SIM_OID = ObjectId("507f1f77bcf86cd799440002")
ATTEMPT = "8f14e45fceea167a5a36dedd4bea2543"
IND_HASH = "aa48834d584df6e66a035e491c5ad291136c25fc"


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
def simulations() -> tuple[SimulationRepository, MagicMock]:
    connection, db = make_connection()
    repo = SimulationRepository(connection)
    coll = db["simulations"]
    coll.reset_mock()                       # drop the index calls from __init__
    return repo, coll


class TestFindByIndividual:
    def test_scopes_by_experiment_when_given(self, simulations):
        repo, coll = simulations
        coll.find.return_value = []

        repo.find_by_individual(IND_HASH, EXP_ID)

        coll.find.assert_called_once_with(
            {"individual_id": IND_HASH, "experiment_id": EXP_ID}
        )

    def test_experiment_id_is_matched_as_objectid(self, simulations):
        """A string would silently match nothing: the field is an ObjectId."""
        repo, coll = simulations
        coll.find.return_value = []

        repo.find_by_individual(IND_HASH, EXP_ID)

        query = coll.find.call_args.args[0]
        assert isinstance(query["experiment_id"], ObjectId)

    def test_without_experiment_the_filter_is_the_hash_alone(self, simulations):
        """Kept working so an existing caller cannot break silently.

        It is the unscoped behaviour that produced the duplicated listing, so
        the omission has to be visible in the filter rather than guessed at.
        """
        repo, coll = simulations
        coll.find.return_value = []

        repo.find_by_individual(IND_HASH)

        coll.find.assert_called_once_with({"individual_id": IND_HASH})

    def test_returns_the_documents_found(self, simulations):
        repo, coll = simulations
        docs = [{"_id": ObjectId(), "individual_id": IND_HASH, "random_seed": 123456},
                {"_id": ObjectId(), "individual_id": IND_HASH, "random_seed": 336157}]
        coll.find.return_value = iter(docs)

        assert repo.find_by_individual(IND_HASH, EXP_ID) == docs

    def test_seeds_of_the_same_experiment_are_both_kept(self, simulations):
        """Two runs of one individual under different seeds are not duplicates.

        The pilot ran each individual twice by design. Scoping must cut the
        other experiments' runs and leave these.
        """
        repo, coll = simulations
        coll.find.return_value = [
            {"individual_id": IND_HASH, "experiment_id": EXP_ID, "random_seed": 123456},
            {"individual_id": IND_HASH, "experiment_id": EXP_ID, "random_seed": 336157},
        ]

        found = repo.find_by_individual(IND_HASH, EXP_ID)

        assert [d["random_seed"] for d in found] == [123456, 336157]


class TestAttemptGuards:
    """Every write about a simulation says which attempt it belongs to.

    An attempt can lose its container without knowing: a recovery pass or the
    stale-reservation sweep may have handed it on. Only ``set_remote_pid`` was
    guarded, so the thread that lost the race still marked the run started,
    published its result over the newer attempt, or failed a simulation that
    had already moved on. The keyword is required of every caller — ``None``
    is how a recovery path says "unconditional, and I mean it".
    """

    def test_running_is_guarded(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        assert repo.mark_running(SIM_OID, attempt_id=ATTEMPT) is True

        assert coll.update_one.call_args.args[0] == {
            "_id": SIM_OID, "attempt_id": ATTEMPT}

    def test_done_is_guarded(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_done(SIM_OID, None, None, {}, attempt_id=ATTEMPT)

        assert coll.update_one.call_args.args[0]["attempt_id"] == ATTEMPT

    def test_error_is_guarded(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(SIM_OID, "boom", attempt_id=ATTEMPT)

        assert coll.update_one.call_args.args[0]["attempt_id"] == ATTEMPT

    def test_a_refused_write_is_reported(self, simulations):
        """False means "not yours any more", and the caller has to act on it."""
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=0)

        assert repo.mark_done(SIM_OID, None, None, {}, attempt_id=ATTEMPT) is False
        assert repo.mark_error(SIM_OID, "boom", attempt_id=ATTEMPT) is False
        assert repo.mark_running(SIM_OID, attempt_id=ATTEMPT) is False

    def test_none_is_unconditional_for_recovery(self, simulations):
        # Recovery and the synthetic path pass None deliberately: there is no
        # attempt, or the one that existed is known to be gone.
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(SIM_OID, "recovered", attempt_id=None)

        # Unconditional on the *attempt*, but never over a published result:
        # see TestAPublishedResultIsFinal.
        assert coll.update_one.call_args.args[0] == {
            "_id": SIM_OID, "status": {"$ne": EnumStatus.DONE}}


class TestRequeue:
    """Losing the container is not the simulation's fault.

    The work was sound, so the simulation goes back to Waiting and the
    dispatcher places it again — rather than being marked in error for a race
    it had no part in.
    """

    def test_it_returns_to_waiting_and_loses_its_placement(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.requeue(SIM_OID, ATTEMPT, "taken over") is True

        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["status"] == EnumStatus.WAITING
        for field in ("attempt_id", "remote_pid", "assigned_resource_id",
                      "assigned_host_id", "start_time", "end_time"):
            assert sets[field] is None, field

    def test_it_is_guarded_by_the_attempt(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.requeue(SIM_OID, ATTEMPT, "taken over")

        assert coll.update_one.call_args.args[0] == {
            "_id": SIM_OID, "attempt_id": ATTEMPT}

    def test_a_simulation_a_newer_attempt_owns_is_not_dragged_back(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.requeue(SIM_OID, ATTEMPT, "taken over") is False

    def test_the_reason_is_kept_for_the_operator(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.requeue(SIM_OID, ATTEMPT, "the reservation is gone")

        assert "reservation" in \
            coll.update_one.call_args.args[1]["$set"]["system_message"]


class TestExpireAttempt:
    """Ending an attempt has to revoke it, not just mark an error.

    The guards on the other writes compare `_id` and `attempt_id` only. With
    the attempt left in place, a thread that still believed in it could follow
    an error with `mark_running` and then `mark_done` — all three matching, the
    document ending `Done` after having been expired. Reproduced in review
    against a collection that applied the real filters.
    """

    def test_it_clears_the_attempt(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        assert repo.expire_attempt(SIM_OID, ATTEMPT, "stale") is True

        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["attempt_id"] is None
        assert sets["status"] == EnumStatus.ERROR
        assert sets["system_message"] == "stale"

    def test_it_is_guarded_by_the_attempt_it_ends(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=1)

        repo.expire_attempt(SIM_OID, ATTEMPT, "stale")

        assert coll.update_one.call_args.args[0] == {
            "_id": SIM_OID, "attempt_id": ATTEMPT}

    def test_a_newer_attempt_is_not_expired(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(modified_count=0)

        assert repo.expire_attempt(SIM_OID, ATTEMPT, "stale") is False

    def test_a_revoked_attempt_can_no_longer_write(self, simulations):
        """The point of the revocation, end to end on one document.

        A collection that applies the filter the repository builds: once the
        attempt is cleared, the later writes match nothing.
        """
        repo, coll = simulations
        document = {"_id": SIM_OID, "attempt_id": ATTEMPT, "status": "Running"}

        def update_one(flt, update, **kwargs):
            matched = all(document.get(k) == v for k, v in flt.items())
            if matched:
                document.update(update["$set"])
            return MagicMock(matched_count=int(matched),
                             modified_count=int(matched))

        coll.update_one.side_effect = update_one

        assert repo.expire_attempt(SIM_OID, ATTEMPT, "stale") is True
        assert repo.mark_running(SIM_OID, attempt_id=ATTEMPT) is False
        assert repo.mark_done(
            SIM_OID, None, None, {}, attempt_id=ATTEMPT) is False
        assert document["status"] == EnumStatus.ERROR


class TestAPublishedResultIsFinal:
    """R24: a ``Done`` simulation is never reclassified as ``Error``.

    ``_publish`` wrote ``mark_done`` and then closed the generation. A failure
    of that *second*, administrative write propagated out of the publish and
    the generic handler marked the finished simulation in error — an
    experiment's outcome changed retroactively by a failure that had nothing
    to do with the run.

    The guard belongs in the repository and not only at the call site,
    because the call site is what got it wrong, and because the attempt filter
    alone matches a simulation this same attempt has just completed.
    """

    def test_the_filter_excludes_a_done_simulation(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(SIM_OID, "generation write failed", attempt_id=ATTEMPT)

        flt = coll.update_one.call_args.args[0]
        assert flt["attempt_id"] == ATTEMPT
        assert flt["status"] == {"$ne": EnumStatus.DONE}

    def test_a_refused_write_is_reported_as_refused(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=0)

        assert repo.mark_error(
            SIM_OID, "too late", attempt_id=ATTEMPT) is False


class TestUnconfirmedResultsKeepTheirArtifacts:
    """The stated policy for a run that produced bytes without completing.

    Preserved and referenced, but not published: ``network_metrics`` is what
    the objective aggregation reads, and an unconfirmed run must not enter a
    mean without the evaluation above it knowing that it did.
    """

    def test_the_artifacts_are_recorded_and_the_metrics_are_set_apart(
        self, simulations
    ):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(
            SIM_OID, "Unconfirmed result: no exit code", attempt_id=ATTEMPT,
            log_id="LOG", csv_id="CSV", partial_metrics={"latency": 3.0})

        sets = coll.update_one.call_args.args[1]["$set"]
        assert sets["status"] == EnumStatus.ERROR
        assert sets["log_cooja_id"] == "LOG"
        assert sets["csv_log_id"] == "CSV"
        assert sets["partial_metrics"] == {"latency": 3.0}
        assert sets["partial"] is True
        # Not under the name the aggregation reads.
        assert "network_metrics" not in sets

    def test_an_ordinary_failure_writes_none_of_that(self, simulations):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)

        repo.mark_error(SIM_OID, "did not complete", attempt_id=ATTEMPT)

        sets = coll.update_one.call_args.args[1]["$set"]
        assert "partial" not in sets and "log_cooja_id" not in sets


class TestTimestampsAreUtc:
    """R25, the write half.

    ``datetime.now()`` is naive *local* time, and PyMongo stores and compares
    it as though it were UTC. On a machine that is not on UTC every one of
    these was wrong by the offset — and the recovery cutoff, which compares
    against ``start_time``, was wrong in whichever direction hurts.
    """

    def _written(self, simulations, call):
        repo, coll = simulations
        coll.update_one.return_value = MagicMock(matched_count=1)
        call(repo)
        return coll.update_one.call_args.args[1]["$set"]

    def test_mark_running(self, simulations):
        sets = self._written(
            simulations, lambda r: r.mark_running(SIM_OID, attempt_id=ATTEMPT))
        assert sets["start_time"].tzinfo is not None

    def test_mark_done(self, simulations):
        sets = self._written(
            simulations,
            lambda r: r.mark_done(SIM_OID, None, None, {}, attempt_id=ATTEMPT))
        assert sets["end_time"].tzinfo is not None

    def test_mark_error(self, simulations):
        sets = self._written(
            simulations, lambda r: r.mark_error(SIM_OID, "x", attempt_id=ATTEMPT))
        assert sets["end_time"].tzinfo is not None

    def test_set_assignment(self, simulations):
        sets = self._written(simulations, lambda r: r.set_assignment(
            SIM_OID, host_id=SIM_OID, host_name="h", resource_id=SIM_OID,
            resource_name="w", attempt_id=ATTEMPT))
        assert sets["assigned_at"].tzinfo is not None


class TestRetryIdentity:
    def test_concurrent_retries_are_protected_by_a_unique_partial_index(self):
        connection, db = make_connection()
        SimulationRepository(connection)
        indices = db["simulations"].create_index.call_args_list
        retry_index = next(c for c in indices if c.kwargs.get("name") == "idx_simulations_retry_identity")
        assert retry_index.kwargs["unique"] is True
        assert retry_index.kwargs["partialFilterExpression"] == {"retry_round": {"$exists": True}}
        assert retry_index.args[0] == [
            ("generation_id", 1), ("individual_id", 1), ("random_seed", 1), ("retry_round", 1)]

    def test_replaying_the_round_keeps_the_existing_clone_and_its_result(self, simulations):
        repo, coll = simulations
        generation_id, clone_id = ObjectId(), ObjectId()
        coll.find_one.return_value = {
            "_id": SIM_OID, "generation_id": generation_id,
            "individual_id": IND_HASH, "random_seed": 42, "status": EnumStatus.ERROR,
            "attempt_id": "old", "remote_pid": 123, "network_metrics": {"latency": 2},
            "unfetched_result": {"remote_path": "old"},
            "recovered_result": {"log_cooja_id": "old"},
        }
        coll.find_one_and_update.return_value = {"_id": clone_id}

        assert repo.clone_for_retry(SIM_OID, retry_round=1) == clone_id

        identity, update = coll.find_one_and_update.call_args.args
        assert identity == {"generation_id": generation_id, "individual_id": IND_HASH,
                            "random_seed": 42, "retry_round": 1}
        assert set(update) == {"$setOnInsert"}, "an existing result must not be reset"
        inserted = update["$setOnInsert"]
        assert inserted["status"] == EnumStatus.WAITING
        assert inserted["retry_of"] == SIM_OID
        assert not {"attempt_id", "remote_pid", "network_metrics", "unfetched_result", "recovered_result"} & inserted.keys()
        assert coll.find_one_and_update.call_args.kwargs["upsert"] is True

    def test_successful_replicas_are_not_cloned(self, simulations):
        repo, coll = simulations
        coll.find_one.return_value = {"_id": SIM_OID, "status": EnumStatus.DONE}

        assert repo.clone_for_retry(SIM_OID, retry_round=1) is None
        coll.find_one_and_update.assert_not_called()


HOST_A = ObjectId("507f1f77bcf86cd799441001")
HOST_B = ObjectId("507f1f77bcf86cd799441002")
RES_1 = ObjectId("507f1f77bcf86cd799442001")
RES_2 = ObjectId("507f1f77bcf86cd799442002")
RES_3 = ObjectId("507f1f77bcf86cd799442003")


def group(host_id, host, resource_id, resource, total,
          done=0, failed=0, active=0, timed=0, duration_ms=0):
    """One row as the ``$group`` stage returns it."""
    return {
        "_id": {"host_id": host_id, "host": host,
                "resource_id": resource_id, "resource": resource},
        "total": total, "done": done, "failed": failed, "active": active,
        "timed": timed, "duration_ms": duration_ms,
    }


class TestPlacementDistribution:
    """Where an experiment's work actually ran.

    The placement is denormalized onto each simulation at reservation time,
    which is the only reason this is answerable after the fact: a worker can
    be renamed, re-registered or removed, and the finished simulation still
    says which container executed it.

    What is asserted here is the shaping — the aggregation itself is one
    ``$group`` and MongoDB's job. The shaping is where the two decisions
    live: what a percentage is a percentage *of*, and what happens to a
    simulation that never reached a container.
    """

    def _distribution(self, simulations, rows):
        repo, coll = simulations
        coll.aggregate.return_value = iter(rows)
        return repo.placement_distribution(EXP_ID)

    def test_it_is_scoped_to_the_experiment(self, simulations):
        repo, coll = simulations
        coll.aggregate.return_value = iter([])

        repo.placement_distribution(EXP_ID)

        pipeline = coll.aggregate.call_args.args[0]
        assert pipeline[0] == {"$match": {"experiment_id": EXP_ID}}

    def test_each_worker_gets_its_share_of_the_placed_simulations(
            self, simulations):
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 6, done=6),
            group(HOST_A, "local", RES_2, "cooja2", 2, done=2),
            group(HOST_B, "vm7", RES_3, "vm7-cooja1", 2, done=2),
        ])

        assert out["placed"] == 10 and out["total"] == 10
        by_host = {h["host_name"]: h for h in out["hosts"]}
        assert by_host["local"]["share"] == 80.0
        assert by_host["vm7"]["share"] == 20.0
        shares = {w["resource_name"]: w["share"]
                  for h in out["hosts"] for w in h["workers"]}
        assert shares == {"cooja1": 60.0, "cooja2": 20.0, "vm7-cooja1": 20.0}

    def test_the_shares_of_the_workers_add_up_to_the_host(self, simulations):
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 1, done=1),
            group(HOST_A, "local", RES_2, "cooja2", 2, done=2),
        ])

        host = out["hosts"][0]
        assert host["total"] == 3
        assert sum(w["total"] for w in host["workers"]) == host["total"]

    def test_a_simulation_that_never_reached_a_container_is_reported_apart(
            self, simulations):
        """Not dropped: a run still waiting, or one that failed before its
        reservation, belongs to no worker — and excluding it would make the
        percentages describe a smaller experiment than the real one."""
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 3, done=3),
            group(None, None, None, None, 2, failed=1, active=1),
        ])

        assert out["placed"] == 3
        assert out["unplaced"] == 2
        assert out["total"] == 5
        assert out["hosts"][0]["workers"][0]["share"] == 100.0, (
            "the share is of what was placed, so one worker that ran "
            "everything placed reads 100% even with work still unplaced")

    def test_outcomes_are_kept_per_worker(self, simulations):
        """A container that took work and produced nothing is the case worth
        seeing, and a single count cannot show it."""
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 5,
                  done=2, failed=2, active=1),
        ])

        worker = out["hosts"][0]["workers"][0]
        assert (worker["done"], worker["failed"], worker["active"]) == (2, 2, 1)
        assert out["hosts"][0]["done"] == 2

    def test_each_worker_reports_its_mean_execution_time(self, simulations):
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 4, done=4,
                  timed=4, duration_ms=4 * 90_000),
            group(HOST_A, "local", RES_2, "cooja2", 2, done=2,
                  timed=2, duration_ms=2 * 150_000),
        ])

        means = {w["resource_name"]: w["mean_duration_sec"]
                 for w in out["hosts"][0]["workers"]}
        assert means == {"cooja1": 90.0, "cooja2": 150.0}

    def test_the_host_mean_is_weighted_by_run(self, simulations):
        """Not the mean of its workers' means: a worker that ran one
        simulation must not weigh as much as one that ran forty."""
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 3, done=3,
                  timed=3, duration_ms=3 * 100_000),
            group(HOST_A, "local", RES_2, "cooja2", 1, done=1,
                  timed=1, duration_ms=1 * 200_000),
        ])

        assert out["hosts"][0]["mean_duration_sec"] == 125.0
        assert "timed" not in out["hosts"][0]
        assert "duration_ms" not in out["hosts"][0]

    def test_a_worker_with_no_finished_run_has_no_mean(self, simulations):
        """Unknown, not zero: zero would read as the fastest worker."""
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 2, failed=1, active=1),
        ])

        assert out["hosts"][0]["workers"][0]["mean_duration_sec"] is None
        assert out["hosts"][0]["mean_duration_sec"] is None

    def test_only_finished_runs_with_both_times_are_timed(self, simulations):
        repo, coll = simulations
        coll.aggregate.return_value = iter([])

        repo.placement_distribution(EXP_ID)

        stage = coll.aggregate.call_args.args[0][1]["$group"]
        condition = stage["timed"]["$sum"]["$cond"][0]["$and"]
        assert {"$eq": ["$status", "Done"]} in condition
        assert {"$eq": [{"$type": "$start_time"}, "date"]} in condition
        assert {"$eq": [{"$type": "$end_time"}, "date"]} in condition

    def test_two_hosts_may_carry_a_worker_of_the_same_name(self, simulations):
        """The name is the label; the id is the identity. Grouping by name
        would merge two different containers into one row."""
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 3, done=3),
            group(HOST_B, "vm7", RES_2, "cooja1", 1, done=1),
        ])

        assert len(out["hosts"]) == 2
        assert all(len(h["workers"]) == 1 for h in out["hosts"])

    def test_a_worker_whose_host_was_removed_still_appears(self, simulations):
        """The point of denormalizing the placement. Reporting nothing would
        lose the record of a machine that did the work."""
        out = self._distribution(simulations, [
            group(None, None, RES_1, "cooja1", 4, done=4),
        ])

        assert out["placed"] == 4
        assert out["hosts"][0]["host_name"] == "(unregistered host)"
        assert out["hosts"][0]["workers"][0]["resource_name"] == "cooja1"

    def test_hosts_and_workers_come_back_busiest_first(self, simulations):
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 1, done=1),
            group(HOST_A, "local", RES_2, "cooja2", 5, done=5),
            group(HOST_B, "vm7", RES_3, "vm7-cooja1", 9, done=9),
        ])

        assert [h["host_name"] for h in out["hosts"]] == ["vm7", "local"]
        assert [w["resource_name"] for w in out["hosts"][1]["workers"]] == [
            "cooja2", "cooja1"]

    def test_an_experiment_that_placed_nothing_is_an_empty_distribution(
            self, simulations):
        """Not an error and not a division by zero: "nothing ran yet" is the
        honest answer for a run that has not started."""
        out = self._distribution(simulations, [])

        assert out == {"placed": 0, "unplaced": 0, "total": 0, "hosts": []}

    def test_nothing_placed_but_work_waiting_does_not_divide_by_zero(
            self, simulations):
        out = self._distribution(simulations, [
            group(None, None, None, None, 8, active=8),
        ])

        assert out["placed"] == 0 and out["unplaced"] == 8
        assert out["hosts"] == []

    def test_a_repeated_replica_counts_on_both_containers_that_ran_it(
            self, simulations):
        """Per document, not per seed. Both occupied a container, and this
        question is about occupancy — the evaluation side deduplicates by
        seed, which is a different question."""
        out = self._distribution(simulations, [
            group(HOST_A, "local", RES_1, "cooja1", 1, failed=1),
            group(HOST_B, "vm7", RES_2, "vm7-cooja1", 1, done=1),
        ])

        assert out["placed"] == 2
        assert {h["host_name"] for h in out["hosts"]} == {"local", "vm7"}
