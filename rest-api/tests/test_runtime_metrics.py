"""Tests for the runtime-metrics endpoints (summary embedding + series)."""
from datetime import datetime

from bson import ObjectId

from pylib.telemetry.artifact import serialize_samples
from tests.conftest import EXP_ID, FILE_ID, sample_experiment

BASE = "/api/v1/experiments"

_STARTED = datetime(2026, 7, 1, 12, 0, 0)
_FINISHED = datetime(2026, 7, 1, 12, 30, 0)


def _samples(n_per_series: int = 4) -> list[dict]:
    out = []
    for i in range(n_per_series):
        ts = _STARTED.timestamp() + 15 * i
        out.append({"timestamp": ts, "metric": "cpu_percent", "value": 10.0 + i,
                    "unit": "percent", "scope": "aggregate", "labels": {}})
        out.append({"timestamp": ts, "metric": "memory_bytes", "value": 1000.0 * (i + 1),
                    "unit": "bytes", "scope": "container", "labels": {"name": "cooja1"}})
    return out


def _artifact_bytes(n_per_series: int = 4) -> tuple[bytes, str]:
    payload = serialize_samples(_samples(n_per_series), prefer_parquet=False)
    return payload.data, payload.content_type


def _experiment_with_metrics(n_per_series: int = 4) -> tuple[dict, bytes]:
    data, content_type = _artifact_bytes(n_per_series)
    doc = sample_experiment()
    doc["runtime_metrics"] = {
        "status": "completed",
        "started_at": _STARTED,
        "finished_at": _FINISHED,
        "collection_finished_at": _FINISHED,
        "collection": {"source": "prometheus", "query_step": "15s"},
        "artifact": {
            "storage": "gridfs",
            "file_id": ObjectId(FILE_ID),
            "filename": "runtime-metrics.csv.gz",
            "content_type": content_type,
            "compression": "gzip",
            "size_bytes": len(data),
            "sha256": "deadbeef",
            "schema_version": 1,
        },
        "summary": {
            "duration_seconds": 1800.0,
            "cpu": {"average_percent": 11.5, "maximum_percent": 13.0},
            "memory": {"average_bytes": 2500.0, "maximum_bytes": 4000.0},
        },
    }
    return doc, data


# ── summary embedded in GET /experiments/{id} ─────────────────────────────────
class TestExperimentSummaryEmbedding:
    def test_returns_summary_and_stringified_file_id(self, client, mock_factory):
        doc, _ = _experiment_with_metrics()
        mock_factory.experiment_repo.get.return_value = doc

        resp = client.get(f"{BASE}/{EXP_ID}")

        assert resp.status_code == 200
        rm = resp.json()["runtime_metrics"]
        assert rm["status"] == "completed"
        assert rm["summary"]["cpu"]["maximum_percent"] == 13.0
        assert rm["summary"]["memory"]["average_bytes"] == 2500.0
        assert rm["summary"]["duration_seconds"] == 1800.0
        assert rm["artifact"]["file_id"] == FILE_ID
        # heavy internals must never be embedded
        assert "collection" not in rm

    def test_absent_block_maps_to_null(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()

        resp = client.get(f"{BASE}/{EXP_ID}")

        assert resp.status_code == 200
        assert resp.json()["runtime_metrics"] is None


# ── GET /experiments/{id}/runtime-metrics ─────────────────────────────────────
class TestRuntimeMetricsSeries:
    def test_reconstructs_series_from_artifact(self, client, mock_factory):
        doc, data = _experiment_with_metrics()
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.fs_handler.read_file_content.return_value = data

        resp = client.get(f"{BASE}/{EXP_ID}/runtime-metrics")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["downsampled"] is False
        assert body["total_samples"] == 8
        assert len(body["series"]) == 2

        by_metric = {s["metric"]: s for s in body["series"]}
        cpu = by_metric["cpu_percent"]
        assert cpu["scope"] == "aggregate"
        assert cpu["unit"] == "percent"
        assert cpu["name"] == "aggregate"
        assert [p[1] for p in cpu["points"]] == [10.0, 11.0, 12.0, 13.0]

        mem = by_metric["memory_bytes"]
        assert mem["name"] == "cooja1"
        assert mem["labels"] == {"name": "cooja1"}

        mock_factory.fs_handler.read_file_content.assert_called_once_with(FILE_ID)

    def test_downsamples_to_max_points(self, client, mock_factory):
        doc, data = _experiment_with_metrics(n_per_series=50)
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.fs_handler.read_file_content.return_value = data

        resp = client.get(f"{BASE}/{EXP_ID}/runtime-metrics?max_points=10")

        assert resp.status_code == 200
        body = resp.json()
        assert body["downsampled"] is True
        for s in body["series"]:
            assert len(s["points"]) <= 10
        # bucket averaging preserves the overall mean of the series
        cpu = next(s for s in body["series"] if s["metric"] == "cpu_percent")
        avg = sum(p[1] for p in cpu["points"]) / len(cpu["points"])
        assert abs(avg - (10.0 + 59.0) / 2) < 1.0

    def test_non_completed_status_returns_empty_series(self, client, mock_factory):
        doc = sample_experiment()
        doc["runtime_metrics"] = {"status": "collecting",
                                  "started_at": _STARTED, "finished_at": None}
        mock_factory.experiment_repo.get.return_value = doc

        resp = client.get(f"{BASE}/{EXP_ID}/runtime-metrics")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "collecting"
        assert body["series"] == []
        mock_factory.fs_handler.read_file_content.assert_not_called()

    def test_missing_block_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = sample_experiment()

        resp = client.get(f"{BASE}/{EXP_ID}/runtime-metrics")

        assert resp.status_code == 404

    def test_missing_experiment_returns_404(self, client, mock_factory):
        mock_factory.experiment_repo.get.return_value = None

        resp = client.get(f"{BASE}/{EXP_ID}/runtime-metrics")

        assert resp.status_code == 404

    def test_unreadable_artifact_returns_500(self, client, mock_factory):
        doc, _ = _experiment_with_metrics()
        mock_factory.experiment_repo.get.return_value = doc
        mock_factory.fs_handler.read_file_content.side_effect = RuntimeError("gridfs down")

        resp = client.get(f"{BASE}/{EXP_ID}/runtime-metrics")

        assert resp.status_code == 500
        assert "telemetry artifact" in resp.json()["detail"]
