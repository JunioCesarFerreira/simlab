"""Serialization of the raw telemetry artifact.

The artifact is a flat table of samples — one row per (timestamp, series)
pair — sufficient to reconstruct every original Prometheus time series:

    timestamp  float   epoch seconds of the sample
    metric     str     SimLab metric name (e.g. "cpu_percent")
    value      float   sample value
    unit       str     value unit (e.g. "percent", "bytes")
    scope      str     "aggregate" (whole SimLab stack) or "container"
    labels     str     JSON-encoded Prometheus labels of the series

Parquet (snappy) is the preferred format; when pyarrow is not available the
writer transparently falls back to gzip-compressed CSV. Readers dispatch on
the stored ``content_type``, so mixed deployments remain compatible.
"""
import csv
import gzip
import io
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1
SAMPLE_COLUMNS = ("timestamp", "metric", "value", "unit", "scope", "labels")

PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
CSV_GZ_CONTENT_TYPE = "application/gzip"
PARQUET_FILENAME = "runtime-metrics.parquet"
CSV_GZ_FILENAME = "runtime-metrics.csv.gz"


@dataclass(frozen=True)
class ArtifactPayload:
    data: bytes
    filename: str
    content_type: str
    compression: str


def _labels_str(sample: dict) -> str:
    labels = sample.get("labels") or {}
    if isinstance(labels, str):
        return labels
    return json.dumps(labels, sort_keys=True)


def _serialize_parquet(samples: list[dict]) -> ArtifactPayload:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({
        "timestamp": pa.array([float(s["timestamp"]) for s in samples], pa.float64()),
        "metric": pa.array([s["metric"] for s in samples], pa.string()),
        "value": pa.array([float(s["value"]) for s in samples], pa.float64()),
        "unit": pa.array([s.get("unit", "") for s in samples], pa.string()),
        "scope": pa.array([s.get("scope", "") for s in samples], pa.string()),
        "labels": pa.array([_labels_str(s) for s in samples], pa.string()),
    })
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    return ArtifactPayload(buf.getvalue(), PARQUET_FILENAME, PARQUET_CONTENT_TYPE, "snappy")


def _serialize_csv_gz(samples: list[dict]) -> ArtifactPayload:
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(SAMPLE_COLUMNS)
    for s in samples:
        writer.writerow([
            float(s["timestamp"]), s["metric"], float(s["value"]),
            s.get("unit", ""), s.get("scope", ""), _labels_str(s),
        ])
    data = gzip.compress(text.getvalue().encode("utf-8"))
    return ArtifactPayload(data, CSV_GZ_FILENAME, CSV_GZ_CONTENT_TYPE, "gzip")


def serialize_samples(samples: list[dict], prefer_parquet: bool = True) -> ArtifactPayload:
    """Serialize samples to Parquet when pyarrow is available, else CSV.gz."""
    if prefer_parquet:
        try:
            return _serialize_parquet(samples)
        except ImportError:
            logger.warning("pyarrow not available — falling back to CSV.gz artifact")
    return _serialize_csv_gz(samples)


def _deserialize_parquet(data: bytes) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(data))
    cols = {name: table.column(name).to_pylist() for name in SAMPLE_COLUMNS}
    return [
        {
            "timestamp": cols["timestamp"][i],
            "metric": cols["metric"][i],
            "value": cols["value"][i],
            "unit": cols["unit"][i],
            "scope": cols["scope"][i],
            "labels": json.loads(cols["labels"][i] or "{}"),
        }
        for i in range(table.num_rows)
    ]


def _deserialize_csv_gz(data: bytes) -> list[dict]:
    text = gzip.decompress(data).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [
        {
            "timestamp": float(row["timestamp"]),
            "metric": row["metric"],
            "value": float(row["value"]),
            "unit": row["unit"],
            "scope": row["scope"],
            "labels": json.loads(row["labels"] or "{}"),
        }
        for row in reader
    ]


def deserialize_samples(data: bytes, content_type: str) -> list[dict]:
    """Read an artifact back into sample dicts (labels decoded to dict)."""
    if content_type == PARQUET_CONTENT_TYPE:
        return _deserialize_parquet(data)
    if content_type == CSV_GZ_CONTENT_TYPE:
        return _deserialize_csv_gz(data)
    raise ValueError(f"Unsupported artifact content_type: {content_type!r}")
