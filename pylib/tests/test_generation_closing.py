"""Tests for closing a generation, which now happens in exactly one place.

The five lines that decide used to be written out four times — the
master-node entrypoint, the runner, the recovery pass and the synthetic path —
and were **missing from a fifth**: the stale-reservation sweep failed the last
simulation still active in a generation without ever re-evaluating it, so the
generation stayed pending for ever and the experiment waited on a strategy
event that would never come.

The connection is mocked in the style of ``test_infra_repository.py``.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.generation import GenerationRepository

GEN_ID = ObjectId("507f1f77bcf86cd799450001")


class FakeDb:
    def __init__(self):
        self.collections: dict[str, MagicMock] = {}

    def __getitem__(self, name: str) -> MagicMock:
        return self.collections.setdefault(name, MagicMock())


@pytest.fixture
def repo_and_sims():
    db = FakeDb()
    connection = MagicMock()

    @contextmanager
    def connect():
        yield db

    connection.connect = connect
    repo = GenerationRepository(connection)
    sims = db["simulations"]
    sims.reset_mock()
    generations = db["generations"]
    generations.reset_mock()
    generations.update_one.return_value = MagicMock(modified_count=1)
    return repo, sims, generations


def _counts(sims, *, active: int, total: int, done: int) -> None:
    """Drive the three count queries the decision makes."""
    def count(flt, *a, **k):
        status = flt.get("status")
        if isinstance(status, dict) and "$in" in status:
            return active
        if status == EnumStatus.DONE:
            return done
        return total
    sims.count_documents.side_effect = count


class TestCloseIfFinished:
    def test_work_still_in_progress_decides_nothing(self, repo_and_sims):
        repo, sims, generations = repo_and_sims
        _counts(sims, active=2, total=10, done=8)

        assert repo.close_if_finished(GEN_ID) is None
        generations.update_one.assert_not_called()

    def test_everything_done_closes_it_done(self, repo_and_sims):
        repo, sims, generations = repo_and_sims
        _counts(sims, active=0, total=10, done=10)

        assert repo.close_if_finished(GEN_ID) == EnumStatus.DONE
        assert generations.update_one.called

    def test_one_failure_closes_it_in_error(self, repo_and_sims):
        repo, sims, generations = repo_and_sims
        _counts(sims, active=0, total=10, done=9)

        assert repo.close_if_finished(GEN_ID) == EnumStatus.ERROR
        assert generations.update_one.called

    def test_a_generation_with_no_simulations_is_done(self, repo_and_sims):
        # Nothing to wait for. Reporting it as in error would stall the
        # experiment on a generation that never had work.
        repo, sims, _ = repo_and_sims
        _counts(sims, active=0, total=0, done=0)

        assert repo.close_if_finished(GEN_ID) == EnumStatus.DONE

    def test_active_is_waiting_or_running(self, repo_and_sims):
        """What "still in progress" means, pinned.

        A simulation sitting in Waiting counts: the generation is not over
        until it has been placed and finished.
        """
        repo, sims, _ = repo_and_sims
        _counts(sims, active=0, total=1, done=1)

        repo.close_if_finished(GEN_ID)

        first = sims.count_documents.call_args_list[0].args[0]
        assert set(first["status"]["$in"]) == {EnumStatus.WAITING, EnumStatus.RUNNING}
        assert first["generation_id"] == GEN_ID

    def test_the_decision_is_reported_to_the_caller(self, repo_and_sims):
        # So the sweep, which had no closing at all, can log what it decided.
        repo, sims, _ = repo_and_sims
        _counts(sims, active=0, total=3, done=2)

        assert repo.close_if_finished(GEN_ID) in (
            EnumStatus.DONE, EnumStatus.ERROR)


class TestReplicaCompleteness:
    """How much evidence stands behind each individual's objectives.

    The aggregation averages whatever finished and said nothing about what did
    not, so an individual evaluated from one replicate was indistinguishable
    from one evaluated from two. That happened: a pilot lost one simulation of
    a two-seed individual, and the objective published for it rested on a
    single run with no record of the difference.
    """

    def _sims(self, repo_and_sims, docs, *, plan=None):
        repo, sims, gens = repo_and_sims
        sims.find.return_value = docs
        # Explicit: a MagicMock's answer here would be a truthy object, and
        # the fallback it lands in is exactly the branch under test.
        gens.find_one.return_value = {"expected_seeds": plan} if plan else {}
        return repo.replica_completeness(GEN_ID, ["latency", "energy"])

    def test_a_complete_individual_reports_both_replicas(self, repo_and_sims):
        result = self._sims(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0, "energy": 2.0}},
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 2, "network_metrics": {"latency": 1.5, "energy": 2.5}},
        ])

        assert result["ind-a"]["expected"] == 2
        assert result["ind-a"]["observed"] == 2
        assert result["ind-a"]["missing_seeds"] == []

    def test_a_failed_replica_is_named_by_its_seed(self, repo_and_sims):
        """So the one to repeat is identifiable, not just countable."""
        result = self._sims(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 123456, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 5343, "network_metrics": {}},
        ])

        entry = result["ind-a"]
        assert entry["expected"] == 2 and entry["observed"] == 1
        assert entry["missing_seeds"] == [5343]
        assert entry["observed_seeds"] == [123456]

    def test_metrics_are_counted_separately(self, repo_and_sims):
        """A run can finish and not carry every metric.

        Two objectives of the *same* individual then rest on different
        numbers of replicas, which a single count cannot express.
        """
        result = self._sims(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0, "energy": 2.0}},
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 2, "network_metrics": {"latency": 1.5}},
        ])

        assert result["ind-a"]["per_metric"] == {"latency": 2, "energy": 1}

    def test_a_simulation_still_running_counts_as_expected_only(
        self, repo_and_sims
    ):
        result = self._sims(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.RUNNING,
             "random_seed": 7, "network_metrics": {}},
        ])

        assert result["ind-a"]["expected"] == 1
        assert result["ind-a"]["observed"] == 0
        assert result["ind-a"]["missing_seeds"] == [7]

    def test_individuals_are_kept_apart(self, repo_and_sims):
        result = self._sims(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-b", "status": EnumStatus.ERROR,
             "random_seed": 1, "network_metrics": {}},
        ])

        assert result["ind-a"]["observed"] == 1
        assert result["ind-b"]["observed"] == 0

    def test_a_simulation_without_an_individual_is_ignored(self, repo_and_sims):
        result = self._sims(repo_and_sims, [
            {"status": EnumStatus.DONE, "random_seed": 1, "network_metrics": {}},
        ])

        assert result == {}


class TestCompletenessIsMeasuredAgainstThePlan:
    """R19: the number of documents is not the number of replicas intended.

    Deriving ``expected`` from the simulations that exist cannot see a replica
    whose *creation* failed — a generation configured for two seeds that
    enqueued one looked complete, with one replica. And after a repeat there
    are more documents than seeds, which would make the same individual look
    over-replicated and let one seed contribute twice.
    """

    def _result(self, repo_and_sims, docs, plan):
        repo, sims, gens = repo_and_sims
        sims.find.return_value = docs
        gens.find_one.return_value = {"expected_seeds": plan}
        return repo.replica_completeness(GEN_ID, ["latency"])

    def test_a_replica_that_was_never_created_is_missing(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
        ], plan=[1, 2])

        entry = result["ind-a"]
        assert entry["expected"] == 2
        assert entry["observed"] == 1
        assert entry["missing_seeds"] == [2]
        assert entry["state"] == "partial"

    def test_five_expected_and_four_done_is_incomplete(self, repo_and_sims):
        """The acceptance criterion, stated literally."""
        docs = [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": seed, "network_metrics": {"latency": 1.0}}
            for seed in (1, 2, 3, 4)
        ]
        result = self._result(repo_and_sims, docs, plan=[1, 2, 3, 4, 5])

        assert result["ind-a"]["state"] == "partial"
        assert result["ind-a"]["missing_seeds"] == [5]

    def test_a_repeated_replica_counts_once(self, repo_and_sims):
        """Two documents, one seed. Counting documents would report three of
        two replicas — and would let the same seed into the mean twice."""
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 2, "network_metrics": {}},
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 2, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
        ], plan=[1, 2])

        entry = result["ind-a"]
        assert entry["observed"] == 2
        assert entry["expected"] == 2
        assert entry["attempts"] == 3, "the repeat is visible"
        assert entry["state"] == "complete"
        assert entry["failed"] == 0, "that seed did produce a result"

    def test_a_metric_missing_from_one_replica_is_partial(self, repo_and_sims):
        """Both replicas finished; one objective rests on fewer than the other."""
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 2, "network_metrics": {}},
        ], plan=[1, 2])

        entry = result["ind-a"]
        assert entry["observed"] == 2 and entry["missing_seeds"] == []
        assert entry["per_metric"] == {"latency": 1}
        assert entry["state"] == "partial"

    def test_a_planned_seed_with_no_simulation_is_absent(self, repo_and_sims):
        """Not pending: nothing will arrive, and there is nothing to repeat.

        Collapsing the two made a generation whose second replica was never
        created look like one that was merely slow — the state that waits for
        ever.
        """
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
        ], plan=[1, 2])

        entry = result["ind-a"]
        assert entry["state"] == "partial"
        assert entry["absent"] == 1
        assert entry["pending"] == 0
        assert entry["failed_seeds"] == [], "nothing to run again"

    def test_a_failed_seed_is_named_as_repeatable(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 2, "network_metrics": {}},
        ], plan=[1, 2])

        assert result["ind-a"]["failed_seeds"] == [2]
        assert result["ind-a"]["absent"] == 0

    def test_a_replica_still_running_is_pending_not_failed(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-a", "status": EnumStatus.RUNNING,
             "random_seed": 2, "network_metrics": {}},
        ], plan=[1, 2])

        entry = result["ind-a"]
        assert entry["state"] == "pending"
        assert entry["pending"] == 1 and entry["failed"] == 0

    def test_nothing_finished_is_invalid(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 1, "network_metrics": {}},
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 2, "network_metrics": {}},
        ], plan=[1, 2])

        assert result["ind-a"]["state"] == "invalid"
        assert result["ind-a"]["failed"] == 2

    def test_a_generation_with_no_plan_reports_unknown(self, repo_and_sims):
        """Experiments created before the plan was recorded. Claiming they
        were complete would be a guess about data nobody wrote down."""
        repo, sims, gens = repo_and_sims
        sims.find.return_value = [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
        ]
        gens.find_one.return_value = {}

        result = repo.replica_completeness(GEN_ID, ["latency"])

        assert result["ind-a"]["state"] == "unknown"
        # Still describes what it saw; it just cannot call it complete.
        assert result["ind-a"]["observed"] == 1

    def test_an_explicit_plan_overrides_the_generation(self, repo_and_sims):
        repo, sims, gens = repo_and_sims
        sims.find.return_value = [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
        ]
        gens.find_one.return_value = {"expected_seeds": [1]}

        result = repo.replica_completeness(
            GEN_ID, ["latency"], expected_seeds=[1, 2, 3])

        assert result["ind-a"]["expected"] == 3


class TestClosingRefusesAnEmptyGeneration:
    """A generation is inserted before its simulations.

    "No simulations" therefore means "not enqueued yet" as often as it means
    "nothing to wait for", and only the caller knows which. The periodic
    pass that repairs a failed close cannot know, so it must not decide.
    """

    def test_a_generation_with_no_simulations_is_left_open(self, repo_and_sims):
        repo, sims, _ = repo_and_sims
        _counts(sims, active=0, total=0, done=0)

        assert repo.close_if_finished(GEN_ID, require_simulations=True) is None

    def test_the_engine_may_still_close_an_all_cached_generation(
        self, repo_and_sims
    ):
        repo, sims, _ = repo_and_sims
        _counts(sims, active=0, total=0, done=0)

        assert repo.close_if_finished(GEN_ID) == EnumStatus.DONE


class TestASeedWithARepeatInFlight:
    """R19's counting half: a repeated seed is pending, not failed.

    The failed attempt puts the seed in the terminal set. Deriving pending as
    "neither done nor terminal" then lost its waiting clone entirely, and the
    record described a generation as ``invalid`` while its repeat was running.
    """

    def _result(self, repo_and_sims, docs, plan):
        repo, sims, gens = repo_and_sims
        sims.find.return_value = docs
        gens.find_one.return_value = {"expected_seeds": plan}
        return repo.replica_completeness(GEN_ID, ["latency"])

    def test_a_failed_attempt_with_a_waiting_clone_is_pending(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.DONE,
             "random_seed": 1, "network_metrics": {"latency": 1.0}},
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 2, "network_metrics": {}},
            {"individual_id": "ind-a", "status": EnumStatus.WAITING,
             "random_seed": 2, "network_metrics": {}},
        ], plan=[1, 2])

        entry = result["ind-a"]
        assert entry["state"] == "pending"
        assert entry["pending"] == 1
        # And **not** offered for repeating again: that would run the same
        # replica twice.
        assert entry["failed_seeds"] == []
        assert entry["failed"] == 0

    def test_a_running_clone_counts_the_same_way(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 1, "network_metrics": {}},
            {"individual_id": "ind-a", "status": EnumStatus.RUNNING,
             "random_seed": 1, "network_metrics": {}},
        ], plan=[1])

        assert result["ind-a"]["state"] == "pending"
        assert result["ind-a"]["failed_seeds"] == []

    def test_a_failed_seed_with_no_clone_is_still_repeatable(self, repo_and_sims):
        result = self._result(repo_and_sims, [
            {"individual_id": "ind-a", "status": EnumStatus.ERROR,
             "random_seed": 1, "network_metrics": {}},
        ], plan=[1])

        assert result["ind-a"]["failed_seeds"] == [1]
        assert result["ind-a"]["state"] == "invalid"


class TestOneResultPerSeedInTheAggregation:
    """The counting and the aggregation have to agree.

    Completeness counts seeds; the aggregation walked ``Done`` documents. Two
    results for one seed therefore weighed it twice in the mean while the
    record still said two of two — a statistical contribution counted twice
    with nothing saying so.
    """

    def _mean(self, repo_and_sims, docs):
        repo, sims, _ = repo_and_sims
        sims.find.return_value = docs
        return repo.get_simulations_metrics_by_individual(GEN_ID, ["latency"])

    def test_two_results_for_one_seed_count_once(self, repo_and_sims):
        from datetime import datetime, timezone

        early = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
        late = datetime(2026, 10, 16, 11, 0, tzinfo=timezone.utc)
        means = self._mean(repo_and_sims, [
            {"individual_id": "ind-a", "random_seed": 1,
             "network_metrics": {"latency": 10.0}, "end_time": early},
            {"individual_id": "ind-a", "random_seed": 1,
             "network_metrics": {"latency": 30.0}, "end_time": late},
            {"individual_id": "ind-a", "random_seed": 2,
             "network_metrics": {"latency": 20.0}, "end_time": late},
        ])

        # The later result for seed 1 wins: it is the one the repetition was
        # for. Mean of 30 and 20, not of 10, 30 and 20.
        assert means["ind-a"]["latency"] == 25.0

    def test_a_result_without_an_end_time_does_not_displace_one(self, repo_and_sims):
        from datetime import datetime, timezone

        means = self._mean(repo_and_sims, [
            {"individual_id": "ind-a", "random_seed": 1,
             "network_metrics": {"latency": 10.0},
             "end_time": datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)},
            {"individual_id": "ind-a", "random_seed": 1,
             "network_metrics": {"latency": 99.0}, "end_time": None},
        ])

        assert means["ind-a"]["latency"] == 10.0

    def test_documents_with_no_seed_are_all_kept(self, repo_and_sims):
        """Legacy documents have nothing to deduplicate by, and silently
        dropping one would change a historical result."""
        means = self._mean(repo_and_sims, [
            {"individual_id": "ind-a", "network_metrics": {"latency": 10.0}},
            {"individual_id": "ind-a", "network_metrics": {"latency": 20.0}},
        ])

        assert means["ind-a"]["latency"] == 15.0

    def test_distinct_seeds_are_all_counted(self, repo_and_sims):
        means = self._mean(repo_and_sims, [
            {"individual_id": "ind-a", "random_seed": 1,
             "network_metrics": {"latency": 10.0}},
            {"individual_id": "ind-a", "random_seed": 2,
             "network_metrics": {"latency": 20.0}},
        ])

        assert means["ind-a"]["latency"] == 15.0
