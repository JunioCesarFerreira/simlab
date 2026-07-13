"""Runtime (computational) telemetry for SimLab experiments.

Collects Prometheus/cAdvisor metrics for the exact ``[start_time, end_time]``
window of an experiment, preserves the raw time series as an immutable GridFS
artifact (Parquet preferred, CSV.gz fallback) and stores a small summary block
(``runtime_metrics``) on the experiment document.
"""
from pylib.telemetry.prometheus import PrometheusClient, PrometheusError
from pylib.telemetry.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    SAMPLE_COLUMNS,
    serialize_samples,
    deserialize_samples,
)
from pylib.telemetry.collector import (
    MetricSpec,
    default_metric_specs,
    collect_samples,
    summarize_samples,
    collect_and_store,
)
from pylib.telemetry.watcher import start_runtime_metrics_watcher

__all__ = [
    "PrometheusClient",
    "PrometheusError",
    "ARTIFACT_SCHEMA_VERSION",
    "SAMPLE_COLUMNS",
    "serialize_samples",
    "deserialize_samples",
    "MetricSpec",
    "default_metric_specs",
    "collect_samples",
    "summarize_samples",
    "collect_and_store",
    "start_runtime_metrics_watcher",
]
