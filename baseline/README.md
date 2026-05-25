# Script-Based Baseline Orchestrator

This directory contains the **lightweight baseline orchestrator** referenced
in §5.6 / Table 6 of the SimLab paper. It is deliberately minimal — a
shell script over Docker Compose that launches `C_MAX` Cooja containers,
dispatches simulations to them via SSH/SCP, and collects raw logs.

It exists **only** for comparison against the full SimLab architecture. It
is *not* meant to be reused as a production tool.

## What this baseline does NOT have

| Feature                  | SimLab full | Baseline (this dir) |
|--------------------------|-------------|---------------------|
| Bounded concurrency      | Explicit `$C_{\max}$` enforced by master-node | Manual / implicit via shell loop |
| Resume after failure     | Automatic from persisted state | Manual re-run; previous results overwritten |
| Traceability             | Full (state + events + artifacts) | Partial (logs and files only) |
| Experiment replay        | State-machine semantics | Hard, due to implicit order |
| Artifact management      | Versioned in MongoDB/GridFS | Ad hoc directories |

These deficiencies are the *point*. The baseline exists to make Table 6 of
the paper reproducible.

## Files

* `docker-compose.baseline.yaml` — 10 standalone Cooja containers, each
  exposing SSH on a different host port (12231–12240). No MongoDB, no
  REST API, no mo-engine, no master-node.
* `run_baseline.sh` — orchestration script. Loops over input bundles,
  dispatches them in parallel to free workers via plain `sshpass` +
  `scp`, collects `COOJA.testlog` from each container.
* `prepare_workload.py` — helper that builds the 30-bundle ×
  50-fixed-mote workload from §5.2 using the same Cooja template the
  mo-engine uses.

## Quick start

```bash
# 1. Generate the input bundles (30 simulations × 50 motes, per §5.2)
python3 prepare_workload.py \
    --count 30 \
    --motes 50 \
    --duration 180 \
    --out ./inputs \
    --firmware-dir ../firmware/rpl-udp-csma

# 2. Start the workers and dispatch
./run_baseline.sh ./inputs ./results 10

# 3. Inspect results
ls results/
# sim_000.log   sim_000.elapsed   sim_000.stdout   sim_000.stderr
# sim_001.log   ...

# 4. Tear down
docker compose -f docker-compose.baseline.yaml down
```

## Required dependencies

* Docker + Docker Compose
* `sshpass` (for non-interactive SSH auth — install with `apt install sshpass`)
* Python 3.12+ (only for `prepare_workload.py`; the orchestrator itself is pure shell)

## Container image

This baseline uses the exact same Cooja image as the full SimLab stack:

```
juniocesarferreira/simulation-cooja:v1.1
```

That guarantees the comparison in Table 6 is not biased by simulator
version drift.

## Limitations to flag in the paper

* No status persistence — if `run_baseline.sh` is killed, the simulations
  already dispatched continue inside their containers and their output
  is *lost from the orchestrator's perspective*.
* No idempotency — re-running the script re-uploads inputs and re-runs
  simulations, overwriting `results/`.
* No load balancing — the round-robin worker assignment in the script
  does not consider container load or fairness.
* SSH/SCP is the only transport (matches the full stack's choice, so
  the comparison stays apples-to-apples).
* JVM tuning (`COOJA_JVM_XMS`, `COOJA_JVM_XMX`) defaults to `4g`/`4g`
  to match the full stack ([../docker-compose.yaml#L198-L199](../docker-compose.yaml#L198-L199));
  override with environment variables if needed.

## Reproducing Table 6 of the paper

1. Provision a host whose resources match §4.3 (12-core EPYC Milan, 94 GB
   RAM, Docker 27.3.1, etc.).
2. Run the full SimLab stack with the same 30 × 50-mote workload (e.g. via
   `debug/requests/post-random-search-experiment-p2.json`) and record:
   total wall-clock time, number of simulations that completed cleanly,
   whether restart was tested.
3. Run this baseline on the same workload. Record the same metrics.
4. Kill `run_baseline.sh` mid-execution to demonstrate the lack of
   recovery; compare against the full stack's automatic resume
   ([../master-node/master-node.py#L124-L146](../master-node/master-node.py#L124-L146)).
