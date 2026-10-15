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

    def _sims(self, repo_and_sims, docs):
        repo, sims, _ = repo_and_sims
        sims.find.return_value = docs
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
