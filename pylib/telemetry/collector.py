"""Collection pipeline: Prometheus → normalized samples → summary + artifact.

``collect_and_store`` is the single entry point used by the watcher. It is
idempotent per experiment: the first caller claims the ``runtime_metrics``
block (status "collecting"); concurrent or repeated invocations are no-ops.
"""
import hashlib
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime

from pylib.telemetry.artifact import ARTIFACT_SCHEMA_VERSION, serialize_samples
from pylib.telemetry.prometheus import PrometheusClient

logger = logging.getLogger(__name__)

# cAdvisor series for the SimLab stack. ``name!=""`` drops the synthetic
# cgroup-hierarchy rows cAdvisor also exports; the simlab.group docker label
# keeps the scope to the containers doing experiment work.
DEFAULT_CONTAINER_FILTER = (
    'container_label_simlab_group=~"simulation|backend",name!=""'
)

STATUS_COLLECTING = "collecting"
STATUS_COMPLETED = "completed"
STATUS_NO_DATA = "no_data"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class MetricSpec:
    metric: str   # SimLab metric name stored in the artifact
    query: str    # PromQL query (range)
    unit: str     # unit of the resulting values
    scope: str    # "aggregate" | "container"


def default_metric_specs(container_filter: str = DEFAULT_CONTAINER_FILTER) -> list[MetricSpec]:
    cpu = f"container_cpu_usage_seconds_total{{{container_filter}}}"
    mem = f"container_memory_usage_bytes{{{container_filter}}}"
    return [
        # Whole-stack aggregates: these feed the summary statistics.
        MetricSpec("cpu_percent", f"sum(rate({cpu}[1m])) * 100", "percent", "aggregate"),
        MetricSpec("memory_bytes", f"sum({mem})", "bytes", "aggregate"),
        # Per-container series: preserved for detailed charts / later analysis.
        MetricSpec("cpu_percent", f"sum by (name) (rate({cpu}[1m])) * 100", "percent", "container"),
        MetricSpec("memory_bytes", f"sum by (name) ({mem})", "bytes", "container"),
    ]


def collect_samples(
    client: PrometheusClient,
    specs: list[MetricSpec],
    start: datetime,
    end: datetime,
    step: str = "15s",
) -> list[dict]:
    """Query every spec over [start, end] and normalize to flat sample rows."""
    samples: list[dict] = []
    for spec in specs:
        for series in client.query_range(spec.query, start, end, step):
            labels = dict(series.get("metric") or {})
            for ts, raw in series.get("values") or []:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue  # NaN/absent samples are not preserved
                samples.append({
                    "timestamp": float(ts),
                    "metric": spec.metric,
                    "value": value,
                    "unit": spec.unit,
                    "scope": spec.scope,
                    "labels": labels,
                })
    samples.sort(key=lambda s: (s["metric"], s["scope"], s["timestamp"]))
    return samples


def _stats(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), max(values)


def summarize_samples(
    samples: list[dict],
    started_at: datetime,
    finished_at: datetime,
) -> dict:
    """Aggregate statistics over ALL samples of the aggregate-scope series.

    New metrics can be summarized later by adding keys next to ``cpu`` and
    ``memory`` without breaking existing documents.
    """
    summary: dict = {
        "duration_seconds": max((finished_at - started_at).total_seconds(), 0.0),
    }
    cpu = [s["value"] for s in samples
           if s["metric"] == "cpu_percent" and s["scope"] == "aggregate"]
    mem = [s["value"] for s in samples
           if s["metric"] == "memory_bytes" and s["scope"] == "aggregate"]
    if cpu:
        avg, peak = _stats(cpu)
        summary["cpu"] = {"average_percent": avg, "maximum_percent": peak}
    if mem:
        avg, peak = _stats(mem)
        summary["memory"] = {"average_bytes": avg, "maximum_bytes": peak}
    return summary


def collect_and_store(
    mongo,
    experiment_id: str,
    started_at: datetime,
    finished_at: datetime,
    prometheus_url: str | None = None,
    step: str | None = None,
    specs: list[MetricSpec] | None = None,
) -> str:
    """Collect telemetry for one finished experiment and persist it.

    Returns the final ``runtime_metrics.status`` ("completed", "no_data",
    "failed") or "skipped" when another collector already owns the block.
    """
    prometheus_url = prometheus_url or os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    step = step or os.getenv("TELEMETRY_QUERY_STEP", "15s")
    specs = specs or default_metric_specs(
        os.getenv("TELEMETRY_CONTAINER_FILTER", DEFAULT_CONTAINER_FILTER)
    )

    if not mongo.experiment_repo.claim_runtime_metrics_collection(
        experiment_id, started_at, finished_at
    ):
        logger.info("[telemetry] runtime metrics already handled for %s", experiment_id)
        return "skipped"

    base = {
        "status": STATUS_COLLECTING,
        "started_at": started_at,
        "finished_at": finished_at,
        "collection": {
            "source": "prometheus",
            "prometheus_url": prometheus_url,
            "query_step": step,
            "metrics": [asdict(s) for s in specs],
        },
    }
    try:
        client = PrometheusClient(prometheus_url)
        samples = collect_samples(client, specs, started_at, finished_at, step)
        summary = summarize_samples(samples, started_at, finished_at)
        base["collection"]["sample_count"] = len(samples)

        if not samples:
            block = {**base, "status": STATUS_NO_DATA,
                     "collection_finished_at": datetime.now(), "summary": summary}
            mongo.experiment_repo.set_runtime_metrics(experiment_id, block)
            logger.warning("[telemetry] no Prometheus samples for experiment %s", experiment_id)
            return STATUS_NO_DATA

        payload = serialize_samples(samples)
        file_id = mongo.fs_handler.upload_bytes(
            payload.data, f"runtime-metrics-{experiment_id}-{payload.filename}"
        )
        block = {
            **base,
            "status": STATUS_COMPLETED,
            "collection_finished_at": datetime.now(),
            "artifact": {
                "storage": "gridfs",
                "file_id": file_id,
                "filename": payload.filename,
                "content_type": payload.content_type,
                "compression": payload.compression,
                "size_bytes": len(payload.data),
                "sha256": hashlib.sha256(payload.data).hexdigest(),
                "schema_version": ARTIFACT_SCHEMA_VERSION,
            },
            "summary": summary,
        }
        mongo.experiment_repo.set_runtime_metrics(experiment_id, block)
        logger.info(
            "[telemetry] stored %d samples (%d bytes, %s) for experiment %s",
            len(samples), len(payload.data), payload.filename, experiment_id,
        )
        return STATUS_COMPLETED
    except Exception as e:
        logger.exception("[telemetry] collection failed for experiment %s", experiment_id)
        mongo.experiment_repo.set_runtime_metrics(experiment_id, {
            **base,
            "status": STATUS_FAILED,
            "collection_finished_at": datetime.now(),
            "error": str(e),
        })
        return STATUS_FAILED
