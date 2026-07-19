# Runtime Metrics — computational telemetry per experiment

For every experiment executed by SimLab, the system preserves a history of
computational metrics (CPU / memory of the SimLab containers) corresponding
exactly to the execution window `[start_time, end_time]`.

The raw time series are preserved **integrally** as an immutable artifact in
GridFS; the experiment document stores only a small summary and a reference
to that artifact. This keeps experiment documents small, keeps the front-end
light, and allows future analyses without re-querying Prometheus (whose
retention is limited).

## Data flow

```text
Experiment starts            → mo-engine sets start_time (status Running)
Experiment finishes          → strategy sets end_time (status Done/Error)
Change Stream fires          → pylib.telemetry watcher (inside mo-engine)
Wait TELEMETRY_COLLECTION_DELAY_SECONDS   (last Prometheus scrape lands)
query_range [start, end]     → cAdvisor CPU/memory series (aggregate + per container)
Normalize samples            → (timestamp, metric, value, unit, scope, labels)
Compute summary              → duration, CPU avg/peak %, memory avg/peak bytes
Serialize artifact           → Parquet/snappy (CSV.gz fallback without pyarrow)
Store in GridFS              → file_id, sha256, size
Update experiment document   → runtime_metrics block (summary + artifact ref)
```

A startup sweep backfills experiments that finished while the watcher was
down, bounded by `TELEMETRY_BACKFILL_HOURS` (default 6 h) so runs whose data
Prometheus no longer retains are not churned. Collection is idempotent: an
atomic claim on the `runtime_metrics` field guarantees a single collector per
experiment.

## Experiment document (`runtime_metrics`)

```json
{
  "runtime_metrics": {
    "status": "completed",            // collecting | completed | no_data | failed
    "started_at": "...",
    "finished_at": "...",
    "collection_finished_at": "...",
    "collection": { "source": "prometheus", "query_step": "15s", "sample_count": 4200, "metrics": [ ... ] },
    "artifact": {
      "storage": "gridfs",
      "file_id": "...",
      "filename": "runtime-metrics.parquet",
      "content_type": "application/vnd.apache.parquet",
      "compression": "snappy",
      "size_bytes": 2138456,
      "sha256": "...",
      "schema_version": 1
    },
    "summary": {
      "duration_seconds": 1823,
      "cpu":    { "average_percent": 47.8, "maximum_percent": 91.6 },
      "memory": { "average_bytes": 918273645, "maximum_bytes": 1325481984 }
    }
  }
}
```

The summary is computed over **all** aggregate-scope samples of the window.
New metric summaries can be added alongside `cpu` / `memory` without breaking
existing documents.

## Artifact schema (v1)

One row per sample; enough to reconstruct every original series:

| column      | type   | description                                    |
| ----------- | ------ | ---------------------------------------------- |
| `timestamp` | float  | epoch seconds                                  |
| `metric`    | string | `cpu_percent`, `memory_bytes`, …               |
| `value`     | float  | sample value                                   |
| `unit`      | string | `percent`, `bytes`, …                          |
| `scope`     | string | `aggregate` (whole stack) or `container`       |
| `labels`    | string | JSON-encoded Prometheus labels of the series   |

The default queries cover the SimLab containers selected by
`container_label_simlab_group=~"simulation|backend"` (override with
`TELEMETRY_CONTAINER_FILTER`), in both scopes.

## API

- `GET /experiments/{id}` — embeds the `runtime_metrics` summary + artifact
  reference (never the series).
- `GET /experiments/{id}/runtime-metrics?max_points=N` — loads the artifact
  from GridFS, reconstructs each series and downsamples it (bucket average)
  to at most `N` points (default 1000). Returns
  `{status, summary, series[], downsampled, total_samples}`.
- `GET /files/{file_id}/as/{extension}` — raw artifact download (generic
  files endpoint).

## Front-end

The experiment detail page shows a **Runtime Metrics** section with summary
tiles (duration, CPU avg/peak, memory avg/peak) loaded with the page, and a
*Show charts* button that fetches the series endpoint on demand and renders
CPU / memory line charts (aggregate emphasized, one thin line per container).

## Configuration (mo-engine environment)

| variable                             | default                  | purpose                          |
| ------------------------------------ | ------------------------ | -------------------------------- |
| `PROMETHEUS_URL`                     | see below                | Prometheus base URL              |
| `TELEMETRY_ENABLED`                  | `True`                   | disable collection entirely      |
| `TELEMETRY_QUERY_STEP`               | `15s`                    | `query_range` resolution         |
| `TELEMETRY_COLLECTION_DELAY_SECONDS` | `30`                     | wait for the final scrape        |
| `TELEMETRY_BACKFILL_HOURS`           | `6`                      | startup sweep window             |
| `TELEMETRY_CONTAINER_FILTER`         | simlab group filter      | PromQL label selector            |

When `PROMETHEUS_URL` is unset, the default depends on where the process runs:
`http://prometheus:9090` inside Docker (`IS_DOCKER=True`) and
`http://localhost:9090` on the host (matching the port published by both the
main stack and `debug/docker/mongo-cooja`). Before collecting, the client
checks Prometheus's health endpoint; if unreachable, collection is skipped
with a single warning and nothing is persisted, so a later backfill sweep can
retry once the monitoring stack is up.
