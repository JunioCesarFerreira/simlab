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
| Bounded concurrency      | Explicit: one reservation per container, enforced by the master-node | Manual / implicit via shell loop (`C_MAX`) |
| Resume after failure     | Automatic from persisted state | Manual re-run; previous results overwritten |
| Traceability             | State, configuration and artifacts per simulation | Partial (logs and files only) |
| Experiment replay        | State-machine semantics | Hard, due to implicit order |
| Artifact management      | Stored per simulation in MongoDB/GridFS | Ad hoc directories |

These deficiencies are the *point*. The baseline exists to make Table 6 of
the paper reproducible. Read [Known issues](#known-issues) before timing it.

## Files

* `docker-compose.baseline.yaml` — 10 standalone Cooja containers, each
  exposing SSH on a different host port (12231–12240), on every interface.
  No MongoDB, no REST API, no mo-engine, no master-node.
* `run_baseline.sh` — orchestration script. Loops over input bundles, keeps at
  most `C_MAX` running, assigns the containers round-robin via plain
  `sshpass` + `scp`, and collects `COOJA.testlog` from each container.
* `prepare_workload.py` — helper that builds the 30-bundle ×
  50-fixed-mote workload from §5.2 using the same Cooja template the
  mo-engine uses. The positions are uniformly random; the first mote is the
  root.

A bundle is a directory with `simulation.csc`, optionally `positions.dat`, and
the firmware files, all in the same directory (no subdirectories).

## Quick start

```bash
# 1. Generate the input bundles (30 simulations × 50 motes, per §5.2).
#    --duration is in simulated minutes.
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
* Python 3.12+ with `numpy` and `pandas`, which `pylib` imports (only for
  `prepare_workload.py`; the orchestrator itself is pure shell)

## Container image

This baseline uses the same Cooja image tag as the full SimLab stack:

```
juniocesarferreira/simulation-cooja:v1.1
```

That keeps simulator version drift out of the comparison in Table 6, as long as
the tag resolves to the same digest for both.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `COOJA_JVM_XMS`, `COOJA_JVM_XMX` | `4g`, `4g` | JVM heap, as in [docker-compose.yaml](../docker-compose.yaml) |
| `SIM_TIMEOUT_SEC` | `10800` | Wall-clock limit of one run |

`C_MAX` is the script's third argument (default 10, at most the 10 services of
the compose file).

## Limitations to flag in the paper

* No status persistence — if `run_baseline.sh` is killed, the simulations
  already dispatched may continue inside their containers and their output
  is *lost from the orchestrator's perspective*.
* No idempotency — re-running the script re-uploads inputs and re-runs
  simulations, overwriting `results/`.
* No load balancing — the round-robin worker assignment in the script
  does not consider container load or fairness.
* SSH/SCP is the only transport (as in the full stack, which adds a second
  hop for remote hosts).

## Known issues

* **Two simulations can land on one container.** A slot frees when *any* run
  ends, but the next container is always the next one in the round-robin, which
  may still be busy. Runs of different lengths therefore overwrite each other's
  files and distort the times.
* **A stale result can be collected.** Nothing is cleaned between runs: if a
  run fails, the previous `COOJA.testlog` is copied as its result, and an old
  `positions.dat` or firmware file stays in place.
* **The timeout kills only the local `ssh`.** The JVM can keep running in the
  container.
* The help text of `--duration` says seconds; the value is in minutes.

## Reproducing Table 6 of the paper

1. Provision a host whose resources match §4.3 (12-core EPYC Milan, 94 GB
   RAM, Docker 27.3.1, etc.).
2. Run the full SimLab stack with the same 30 × 50-mote workload and record:
   total wall-clock time, number of simulations that completed cleanly,
   whether restart was tested. No payload in
   [debug/requests](../debug/requests) is that workload:
   `post-random-search-experiment-p2.json` has 20 individuals, 5 generations
   and a P2 scenario, so it has to be adapted.
3. Run this baseline on the same workload. Record the same metrics.
4. Kill `run_baseline.sh` mid-execution to demonstrate the lack of
   recovery; compare against the full stack's automatic resume
   ([recovery.py](../master-node/lib/recovery.py), run at the master-node's
   startup).
