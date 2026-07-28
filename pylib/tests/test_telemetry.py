"""Tests for the runtime telemetry pipeline (pylib.telemetry).

Covers normalization of Prometheus matrix responses, summary statistics,
artifact round-trips (Parquet and CSV.gz) and the idempotent collect-and-store
orchestration against fake repositories — no live Prometheus/MongoDB needed.
"""
import hashlib
import importlib.util
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from bson import ObjectId

from pylib.telemetry import collector as collector_mod
from pylib.telemetry.artifact import (
    CSV_GZ_CONTENT_TYPE,
    PARQUET_CONTENT_TYPE,
    deserialize_samples,
    serialize_samples,
)
from pylib.telemetry.collector import (
    MetricSpec,
    collect_and_store,
    collect_samples,
    default_metric_specs,
    summarize_samples,
)

HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None

START = datetime(2026, 7, 1, 12, 0, 0)
END = START + timedelta(seconds=60)

SPECS = [
    MetricSpec("cpu_percent", "q_cpu_agg", "percent", "aggregate"),
    MetricSpec("memory_bytes", "q_mem_agg", "bytes", "aggregate"),
]


class FakePrometheusClient:
    """Maps PromQL query → canned /query_range matrix result."""

    def __init__(self, results: dict[str, list[dict]], fail: bool = False, available: bool = True):
        self.results = results
        self.fail = fail
        self.available = available

    def is_available(self, timeout: float = 5.0) -> bool:
        return self.available

    def query_range(self, query, start, end, step="15s"):
        if self.fail:
            raise RuntimeError("prometheus unreachable")
        return self.results.get(query, [])


def _matrix(labels: dict, values: list[tuple[float, str]]) -> dict:
    return {"metric": labels, "values": [[t, v] for t, v in values]}


def _fake_results() -> dict[str, list[dict]]:
    t0 = START.timestamp()
    return {
        "q_cpu_agg": [_matrix({}, [(t0, "10.0"), (t0 + 15, "50.0"), (t0 + 30, "30.0")])],
        "q_mem_agg": [_matrix({}, [(t0, "1000"), (t0 + 15, "3000")])],
    }


# ── normalization ──────────────────────────────────────────────────────────────
class TestCollectSamples:
    def test_flattens_matrix_to_sample_rows(self):
        client = FakePrometheusClient(_fake_results())
        samples = collect_samples(client, SPECS, START, END)

        cpu = [s for s in samples if s["metric"] == "cpu_percent"]
        mem = [s for s in samples if s["metric"] == "memory_bytes"]
        assert len(cpu) == 3 and len(mem) == 2
        assert cpu[0]["unit"] == "percent"
        assert cpu[0]["scope"] == "aggregate"
        assert cpu[0]["labels"] == {}
        # sorted by (metric, scope, timestamp)
        assert [s["value"] for s in cpu] == [10.0, 50.0, 30.0]

    def test_preserves_series_labels(self):
        results = {"q_cpu_agg": [
            _matrix({"name": "cooja1"}, [(START.timestamp(), "1.0")]),
            _matrix({"name": "cooja2"}, [(START.timestamp(), "2.0")]),
        ]}
        client = FakePrometheusClient(results)
        samples = collect_samples(client, SPECS[:1], START, END)
        assert {s["labels"]["name"] for s in samples} == {"cooja1", "cooja2"}

    def test_skips_non_numeric_values(self):
        results = {"q_cpu_agg": [_matrix({}, [(START.timestamp(), "NaN-ish")])]}
        # float("NaN") would parse; a truly malformed string must be dropped
        client = FakePrometheusClient(results)
        assert collect_samples(client, SPECS[:1], START, END) == []


# ── summary statistics ─────────────────────────────────────────────────────────
class TestSummarizeSamples:
    def test_cpu_and_memory_stats_over_all_aggregate_samples(self):
        client = FakePrometheusClient(_fake_results())
        samples = collect_samples(client, SPECS, START, END)
        summary = summarize_samples(samples, START, END)

        assert summary["duration_seconds"] == 60.0
        assert summary["cpu"]["average_percent"] == pytest.approx(30.0)
        assert summary["cpu"]["maximum_percent"] == pytest.approx(50.0)
        assert summary["memory"]["average_bytes"] == pytest.approx(2000.0)
        assert summary["memory"]["maximum_bytes"] == pytest.approx(3000.0)

    def test_container_scope_does_not_leak_into_summary(self):
        samples = [{"timestamp": 1.0, "metric": "cpu_percent", "value": 99.0,
                    "unit": "percent", "scope": "container", "labels": {"name": "cooja1"}}]
        summary = summarize_samples(samples, START, END)
        assert "cpu" not in summary

    def test_empty_samples_yield_duration_only(self):
        summary = summarize_samples([], START, END)
        assert summary == {"duration_seconds": 60.0}


# ── artifact round-trip ────────────────────────────────────────────────────────
def _samples() -> list[dict]:
    return [
        {"timestamp": 1.5, "metric": "cpu_percent", "value": 42.0,
         "unit": "percent", "scope": "aggregate", "labels": {}},
        {"timestamp": 2.5, "metric": "memory_bytes", "value": 1024.0,
         "unit": "bytes", "scope": "container", "labels": {"name": "cooja1"}},
    ]


class TestArtifactRoundTrip:
    def test_csv_gz_round_trip(self):
        payload = serialize_samples(_samples(), prefer_parquet=False)
        assert payload.content_type == CSV_GZ_CONTENT_TYPE
        assert payload.filename.endswith(".csv.gz")
        assert deserialize_samples(payload.data, payload.content_type) == _samples()

    @pytest.mark.skipif(not HAS_PYARROW, reason="pyarrow not installed")
    def test_parquet_round_trip(self):
        payload = serialize_samples(_samples(), prefer_parquet=True)
        assert payload.content_type == PARQUET_CONTENT_TYPE
        assert payload.compression == "snappy"
        assert deserialize_samples(payload.data, payload.content_type) == _samples()

    def test_unknown_content_type_rejected(self):
        with pytest.raises(ValueError):
            deserialize_samples(b"", "text/plain")


# ── collect_and_store orchestration ────────────────────────────────────────────
class FakeExperimentRepo:
    def __init__(self):
        self.blocks: dict[str, dict] = {}
        self.claimed: set[str] = set()

    def claim_runtime_metrics_collection(self, experiment_id, started_at, finished_at):
        if experiment_id in self.claimed:
            return False
        self.claimed.add(experiment_id)
        return True

    def set_runtime_metrics(self, experiment_id, block):
        self.blocks[experiment_id] = block
        return True


class FakeFSHandler:
    def __init__(self):
        self.uploads: list[tuple[bytes, str]] = []

    def upload_bytes(self, data, name):
        self.uploads.append((data, name))
        return ObjectId()


@pytest.fixture
def fake_mongo():
    return SimpleNamespace(experiment_repo=FakeExperimentRepo(), fs_handler=FakeFSHandler())


EXP_ID = "507f1f77bcf86cd799439011"


class TestCollectAndStore:
    def _run(self, fake_mongo, monkeypatch, client):
        monkeypatch.setattr(collector_mod, "PrometheusClient", lambda url: client)
        return collect_and_store(
            fake_mongo, EXP_ID, START, END,
            prometheus_url="http://prom:9090", step="15s", specs=SPECS,
        )

    def test_completed_stores_artifact_and_summary(self, fake_mongo, monkeypatch):
        status = self._run(fake_mongo, monkeypatch, FakePrometheusClient(_fake_results()))
        assert status == "completed"

        block = fake_mongo.experiment_repo.blocks[EXP_ID]
        assert block["status"] == "completed"
        assert block["started_at"] == START and block["finished_at"] == END
        assert block["collection_finished_at"] is not None
        assert block["summary"]["cpu"]["maximum_percent"] == pytest.approx(50.0)

        artifact = block["artifact"]
        data, name = fake_mongo.fs_handler.uploads[0]
        assert artifact["storage"] == "gridfs"
        assert isinstance(artifact["file_id"], ObjectId)
        assert artifact["size_bytes"] == len(data)
        assert artifact["sha256"] == hashlib.sha256(data).hexdigest()
        assert artifact["schema_version"] == 1
        assert EXP_ID in name

        # the artifact reconstructs the collected series exactly
        samples = deserialize_samples(data, artifact["content_type"])
        assert len(samples) == 5

    def test_second_invocation_is_skipped(self, fake_mongo, monkeypatch):
        client = FakePrometheusClient(_fake_results())
        assert self._run(fake_mongo, monkeypatch, client) == "completed"
        assert self._run(fake_mongo, monkeypatch, client) == "skipped"
        assert len(fake_mongo.fs_handler.uploads) == 1

    def test_no_samples_marks_no_data(self, fake_mongo, monkeypatch):
        status = self._run(fake_mongo, monkeypatch, FakePrometheusClient({}))
        assert status == "no_data"
        block = fake_mongo.experiment_repo.blocks[EXP_ID]
        assert block["status"] == "no_data"
        assert "artifact" not in block
        assert block["summary"]["duration_seconds"] == 60.0
        assert fake_mongo.fs_handler.uploads == []

    def test_prometheus_failure_marks_failed(self, fake_mongo, monkeypatch):
        status = self._run(fake_mongo, monkeypatch, FakePrometheusClient({}, fail=True))
        assert status == "failed"
        block = fake_mongo.experiment_repo.blocks[EXP_ID]
        assert block["status"] == "failed"
        assert "prometheus unreachable" in block["error"]

    def test_unreachable_prometheus_skips_without_claiming(self, fake_mongo, monkeypatch):
        """Local run without the monitoring stack: one warning, nothing persisted,
        and no claim — so a later backfill sweep can still collect."""
        client = FakePrometheusClient(_fake_results(), available=False)
        assert self._run(fake_mongo, monkeypatch, client) == "unavailable"
        assert EXP_ID not in fake_mongo.experiment_repo.blocks
        assert fake_mongo.fs_handler.uploads == []
        # Prometheus comes back → collection succeeds on retry
        client.available = True
        assert self._run(fake_mongo, monkeypatch, client) == "completed"


class TestDefaultPrometheusUrl:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_URL", "http://custom:9999")
        monkeypatch.setenv("IS_DOCKER", "True")
        assert collector_mod.default_prometheus_url() == "http://custom:9999"

    def test_docker_uses_service_name(self, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_URL", raising=False)
        monkeypatch.setenv("IS_DOCKER", "True")
        assert collector_mod.default_prometheus_url() == "http://prometheus:9090"

    def test_local_uses_localhost(self, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_URL", raising=False)
        monkeypatch.delenv("IS_DOCKER", raising=False)
        assert collector_mod.default_prometheus_url() == "http://localhost:9090"


class TestDefaultMetricSpecs:
    def test_covers_cpu_and_memory_in_both_scopes(self):
        specs = default_metric_specs()
        combos = {(s.metric, s.scope) for s in specs}
        assert combos == {
            ("cpu_percent", "aggregate"),
            ("memory_bytes", "aggregate"),
            ("cpu_percent", "container"),
            ("memory_bytes", "container"),
        }
