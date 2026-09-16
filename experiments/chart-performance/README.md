# Chart loading review — 2026-09-13

The review found unnecessary work in the metrics endpoint, repeated queries
to MongoDB, and processing that blocked the UI.

## Changes

- `/hv-gd` computed cumulative HV over the entire history even when the
  frontend only used the selected population's series. Details and
  comparison now send `include_cumulative=false`. The response keeps
  `hv_cumulative: []` in that case; the API default remains `true` to
  preserve existing clients. Selecting Archive still computes the exact
  Archive series in `hv`.
- Dominance filtering moved from quadratic Python loops to the already
  installed `moocore`. Cumulative HV is no longer recomputed when the set
  stays the same and is reused as the population's HV in Archive mode.
- Individuals are fetched in one query per experiment, grouped in memory.
  `/hv-gd` projects only generation, identifier, and objectives; `/full`
  and the generation listing also stop querying each generation
  separately. The `experiment_id` indexes already existed.
- The API reuses the MongoDB pool and closes it on shutdown. The other
  services keep their previous connection lifecycle.
- Metrics cache keyed by data content, capped at 32 responses and 8 MiB
  of JSON per process. Objectives, references, survivors, and options are
  part of the key; changes to an existing generation invalidate the
  result. Database queries still run to verify current data. Identical
  concurrent requests share the computation.
- Frontend polling waits for the response before scheduling the next
  query, pauses fetching while the tab is hidden, and cancels requests on
  leaving/switching experiments. Stale responses no longer replace the
  current screen.
- HV/GD/IGD track data changes; updates during a computation are batched
  into a new fetch. Update failures preserve the existing chart and offer
  a retry.
- Pareto ranks are computed in a Web Worker, with front lookup and without
  storing the quadratic dominance graph. Large snapshots use `shallowRef`.
  Indicator charts use canvas, with no initial animation nor a persistent
  symbol per point.
- HTTP compression on the API and Nginx; caching of the hashed static
  files produced by Vite.

## Local measurements

Deterministic synthetic data: 30 generations × 50 points on the DTLZ2
front, seed 20260913. Simulated repositories, no network or real MongoDB.
Times measure the Python endpoint, not production latency nor browser
painting.

| Scenario | Observed time |
| --- | ---: |
| Previous code, 3 objectives, including cumulative | 5.53 s |
| Optimized, same scenario and all series | 36 ms |
| Repeat served from cache, same scenario | 2 ms |
| Optimized, 6 objectives, still including cumulative | exceeded 45 s |
| Optimized, 6 objectives, only the series shown in Offspring | 33 ms |

In the 3-objective scenario, HV, cumulative HV, GD, IGD, and IGD+ had zero
maximum numerical difference versus the previous code. Individual queries
dropped from 30 to 1 per request, not counting the savings in connections
and traffic with a real MongoDB.

Final run results: [3 objectives](3d.json) and [6 objectives](6d.json).

## Reproduction

From the repository root:

```bash
# Preserves the previous endpoint in a temporary file; does not touch Git.
git show HEAD:rest-api/api/endpoints/experiment.py > /tmp/simlab-hvgd-baseline.py
rest-api/.venv/bin/python experiments/chart-performance/benchmark.py \
  --include-cumulative --baseline-file /tmp/simlab-hvgd-baseline.py

# Path now used by the frontend, with six objectives.
rest-api/.venv/bin/python experiments/chart-performance/benchmark.py --dimensions 6
```

The file used as baseline must contain the code prior to this review. The
benchmark prints times, query counts, and numerically compares the series
when given `--baseline-file`.

Validation: 161 API tests, 172 shared-library tests (1 skipped), 98
frontend tests, lint, and build. Vite still shows the large-chunk warning
for the charting libraries. The tests cover rank equivalence, min/max
orientation, population modes, cache, batch queries, and out-of-order
responses.

To apply this to a running environment, the API and frontend need to be
rebuilt/restarted. The database and experiments were not modified during
the review. The first exact Archive computation may still exceed the
timeout on large histories with many objectives; this review does not
introduce approximations or sampling of the indicators. Latency
measurement with real data remains dependent on the runtime environment.
