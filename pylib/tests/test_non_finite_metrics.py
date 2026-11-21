"""A non-finite metric is not a replica, and a stored vector is judged by its evidence.

Found in the 2026-11-18 review. ``statistics`` returns NaN for a column that
is absent or empty; the master publishes the simulation as ``Done``; the
completeness count took the NaN as present (``metric in measured``) and the
aggregate averaged it, so one NaN made the individual's objective NaN with a
record saying ``complete``. A NaN vector is never dominated, so it reached
front 0.

The second half is the reader side: fronts and indicators used to drop a
vector when any objective was ≥ 1e8, which also drops a legitimate value of
that size and keeps a placeholder below it.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest
from bson import ObjectId

from pylib import moo_metrics
from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.generation import GenerationRepository, _completeness_of
from pylib.evaluation_validity import is_excluded, is_valid_evaluation

GEN_ID = ObjectId("507f1f77bcf86cd799450201")
NAN = float("nan")


def _done(seed, **metrics):
    return {"individual_id": "a", "status": EnumStatus.DONE,
            "random_seed": seed, "network_metrics": metrics}


class TestCompleteness:
    def test_a_nan_metric_is_not_counted_as_present(self):
        record = _completeness_of(
            [_done(1, latency=1.0, energy=NAN), _done(2, latency=2.0, energy=3.0)],
            ["latency", "energy"], [1, 2])
        assert record["per_metric"] == {"latency": 2, "energy": 1}
        assert record["thin_metrics"] == ["energy"]
        assert record["state"] == "partial"
        assert "non-finite" in record["reason"]

    @pytest.mark.parametrize("bad", [NAN, float("inf"), None, "12"])
    def test_any_non_number_is_absent(self, bad):
        record = _completeness_of([_done(1, latency=bad)], ["latency"], [1])
        assert record["per_metric"]["latency"] == 0
        assert record["state"] == "partial"

    def test_finite_metrics_are_still_complete(self):
        record = _completeness_of([_done(1, latency=0.0)], ["latency"], [1])
        assert record["state"] == "complete"


@pytest.fixture
def repo():
    db: dict = {}

    class FakeDb:
        def __getitem__(self, name):
            return db.setdefault(name, MagicMock())

    connection = MagicMock()

    @contextmanager
    def connect():
        yield FakeDb()

    connection.connect = connect
    return GenerationRepository(connection), FakeDb()["simulations"]


class TestAggregation:
    def test_a_nan_replica_does_not_poison_the_mean(self, repo):
        repository, sims = repo
        sims.find.return_value = [
            _done(1, latency=10.0), _done(2, latency=NAN), _done(3, latency=20.0)]
        result = repository.get_simulations_metrics_by_individual(GEN_ID, ["latency"])
        assert result["a"]["latency"] == 15.0

    def test_a_metric_that_is_nan_everywhere_is_absent(self, repo):
        repository, sims = repo
        sims.find.return_value = [_done(1, latency=NAN), _done(2, latency=NAN)]
        result = repository.get_simulations_metrics_by_individual(GEN_ID, ["latency"])
        # Absent, so the engine's extraction refuses the individual instead of
        # sorting a NaN.
        assert "latency" not in result.get("a", {})


class TestValidityRule:
    def test_records(self):
        assert is_valid_evaluation(None)
        assert is_valid_evaluation({"state": "complete", "objective_used": True})
        assert not is_valid_evaluation({"state": "infeasible"})
        assert not is_valid_evaluation({"state": "complete", "refused": "nan"})
        assert not is_valid_evaluation({"state": "partial", "objective_used": False})
        assert not is_valid_evaluation({"state": "complete", "valid": False})

    def test_a_large_measurement_with_evidence_is_kept(self):
        # Energy summed over a long run already reaches 1.8e7; 2e8 is a value
        # the magnitude rule would have thrown away.
        assert not is_excluded([2e8, 1.0], {"state": "complete", "objective_used": True})

    def test_a_penalty_is_dropped_by_its_record_not_its_size(self):
        assert is_excluded([5.0, 1.0], {"state": "infeasible"})

    def test_legacy_documents_keep_the_magnitude_rule(self):
        assert is_excluded([1e9, 1.0], None)
        assert not is_excluded([1e7, 1.0], None)

    def test_non_finite_is_always_excluded(self):
        assert is_excluded([NAN, 1.0], {"state": "complete", "objective_used": True})
        assert is_excluded([float("inf"), 1.0], None)


class TestReferenceFront:
    def test_records_decide_and_rows_without_one_fall_back(self):
        rows = [[2e8, 1.0], [1.0, 2e8], [0.5, 0.5], [1e9, 1e9]]
        records = [
            {"valid": True},               # large, but a measurement
            {"valid": False},              # a penalty that happens to be small-ish
            None,                          # legacy row, small: kept
            None,                          # legacy row, large: dropped
        ]
        front = moo_metrics.sanitize_reference_front(rows, records=records)
        kept = {tuple(r) for r in front.tolist()}
        assert (1.0, 2e8) not in kept and (1e9, 1e9) not in kept
        assert (0.5, 0.5) in kept

    def test_a_large_valid_row_survives(self):
        front = moo_metrics.sanitize_reference_front(
            [[2e8, 0.1], [1.0, 2.0]], records=[{"valid": True}, {"valid": True}])
        assert [2e8, 0.1] in front.tolist()

    def test_without_records_the_behaviour_is_unchanged(self):
        front = moo_metrics.sanitize_reference_front([[1e9, 1e9], [1.0, 2.0]])
        assert np.array_equal(front, np.array([[1.0, 2.0]]))

    def test_a_nan_row_is_dropped(self):
        front = moo_metrics.sanitize_reference_front([[NAN, 1.0], [1.0, 2.0]])
        assert front.tolist() == [[1.0, 2.0]]
