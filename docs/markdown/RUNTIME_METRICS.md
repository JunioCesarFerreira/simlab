# Runtime Metrics — computational telemetry per experiment

For every experiment executed by SimLab, the system preserves a history of
computational metrics (CPU / memory of this machine's SimLab containers)
corresponding to the execution window `[start_time, end_time]`. Containers of
remote hosts are not measured.

The queried time series are preserved **integrally**, at the query step, as an
artifact in GridFS that is never modified (a new collection writes a new one); the experiment document stores only a small summary and a reference
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

A **periodic** sweep (also runs once at startup) backfills experiments that
finished while the watcher was down. It also selects runs left in a transient
`unavailable`/`failed` state — e.g. Prometheus was down at finish time and has
since come back — but today it [does not manage to collect them](#known-issues).
It is bounded by `TELEMETRY_BACKFILL_HOURS` (default 6 h) so
runs whose data Prometheus no longer retains are not churned, and its cadence
is `TELEMETRY_SWEEP_INTERVAL_SECONDS` (default 600 s; `0` = startup only).
Collection is idempotent: an atomic claim on the `runtime_metrics` field
guarantees a single collector per experiment. A manual collection overwrites
the block instead of claiming it. A claim left `collecting` by a crash is
released by the sweep after `TELEMETRY_STALE_CLAIM_SECONDS`.

When Prometheus is unreachable the state is **persisted** as `unavailable`
(rather than silently dropped) so the experiment page can surface it and offer
a manual retry. An operator can also trigger a collection on demand via
`POST /experiments/{id}/runtime-metrics/collect` (see API below).

## Experiment document (`runtime_metrics`)

```jsonc
{
  "runtime_metrics": {
    "status": "completed",            // collecting | completed | no_data | failed | unavailable
    "started_at": "...",
    "finished_at": "...",
    "collection_finished_at": "...",
    "collection": { "source": "prometheus", "prometheus_url": "...", "query_step": "15s",
                    "container_filter": "...", "sample_count": 4200, "metrics": [ ... ] },
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

The summary is computed over **all** finite aggregate-scope samples of the window.
CPU is in percent of one core and summed over the containers, so it can exceed
100.
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
`container_label_simlab_group=~"simulation|backend",name!=""` (override with
`TELEMETRY_CONTAINER_FILTER`), in both scopes.

## API

- `GET /experiments/{id}` — embeds the `runtime_metrics` summary + artifact
  reference (never the series).
- `GET /experiments/{id}/runtime-metrics?max_points=N` — loads the artifact
  from GridFS, reconstructs each series and downsamples it (bucket average)
  to at most `N` points (default 1000). Returns
  `{status, started_at, finished_at, summary, series[], downsampled, total_samples}`.
- `POST /experiments/{id}/runtime-metrics/collect` — manually (re)collect from
  Prometheus, overwriting any existing block. Body `{start?, end?}` (ISO; with
  an offset, or naive in the server's local time, which is UTC in the
  containers) overrides the window; omitted, the experiment's own
  `[start_time, end_time]` is used. Returns `{status, runtime_metrics}`. `409`
  while a collection is already in progress. This runs inside the REST API, so
  it must reach Prometheus (`PROMETHEUS_URL`, monitoring network).
- `GET /files/{file_id}/as/{extension}` — raw artifact download (generic
  files endpoint).

## Front-end

The experiment detail page shows a **Runtime Metrics** section for every
finished run. When a completed collection exists it renders summary tiles
(duration, CPU avg/peak, memory avg/peak) and a *Show charts* button that
fetches the series endpoint on demand (aggregate emphasized, one thin line per
container). When no data was recorded (`unavailable`/`no_data`/`failed`, or no
block at all) it shows an alert explaining the situation and a *Collect from
Prometheus* shortcut: the operator confirms/edits the `[start, end]` window and
the front calls the collect endpoint, then refreshes.

## Configuration (mo-engine environment)

| variable                             | default                  | purpose                          |
| ------------------------------------ | ------------------------ | -------------------------------- |
| `PROMETHEUS_URL`                     | see below                | Prometheus base URL              |
| `TELEMETRY_ENABLED`                  | `True`                   | disable collection entirely      |
| `TELEMETRY_QUERY_STEP`               | `15s`                    | `query_range` resolution         |
| `TELEMETRY_COLLECTION_DELAY_SECONDS` | `30`                     | wait for the final scrape        |
| `TELEMETRY_BACKFILL_HOURS`           | `6`                      | sweep window (recent runs only)  |
| `TELEMETRY_SWEEP_INTERVAL_SECONDS`   | `600`                    | periodic sweep cadence (0=once)  |
| `TELEMETRY_STALE_CLAIM_SECONDS`      | `900`                    | release stale `collecting` claims|
| `TELEMETRY_CONTAINER_FILTER`         | simlab group filter      | PromQL label selector            |

`docker-compose.yaml` passes only `PROMETHEUS_URL`: any other variable has to
be added to the service's `environment:` block.

`PROMETHEUS_URL` is also set on the **rest-api** service, which serves the
manual collect endpoint and therefore queries Prometheus directly.

When `PROMETHEUS_URL` is unset, the default depends on where the process runs:
`http://prometheus:9090` inside Docker (`IS_DOCKER=True`) and
`http://localhost:9090` on the host (matching the port published by both the
main stack and `debug/docker/mongo-cooja`). Before collecting, the client
checks Prometheus's health endpoint; if unreachable, the query is skipped with
a single warning and the state is persisted as `unavailable`.

## Retries and missing samples

- **`unavailable` and `failed` are retried by the sweep**, every
  `TELEMETRY_SWEEP_INTERVAL_SECONDS`, for the runs that ended in the last
  `TELEMETRY_BACKFILL_HOURS`: it overwrites the block it finds (`force`). Older
  runs are retried with *Collect from Prometheus* on the experiment page, or the
  collect endpoint.
- **A NaN sample** (Prometheus answers `NaN` for a window without data) is kept
  in the raw artifact and left out of the summary's averages and maxima; the
  summary's `non_finite_samples` says how many were left out.

## Known issues

None known.
