# master-node

The **master-node** is the simulation execution orchestrator in SimLab. It bridges MongoDB (where experiments and generations are defined) and a pool of [Cooja](https://github.com/contiki-ng/cooja) simulator containers, managing the full lifecycle of every simulation job: queuing, file transfer, execution, log collection, metric extraction, and status reporting.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          master-node                            │
│                                                                 │
│  ┌──────────────┐    shared     ┌─────────────────────────────┐ │
│  │  mongowatch  │──────queue───▶│  simulation_worker (×N)     │ │
│  │  (thread)    │               │  - prepare files            │ │
│  └──────────────┘               │  - SSH/SCP transfer         │ │
│         ▲                       │  - run Cooja                │ │
│         │                       │  - collect logs & metrics   │ │
│  MongoDB Change                 └──────────────┬──────────────┘ │
│  Stream (WAITING                               │                │
│  generations)                                  │ mark done/error│
│                                                ▼                │
│                                  MongoDB (simulations,          │
│                                  generations, GridFS)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Startup Sequence

When `master-node.py` starts:

1. **Load settings** from environment variables.
2. **Recover stuck simulations** — any simulation left in `RUNNING` from a previous crash is marked `ERROR`, and its generation is re-evaluated.
3. **Start N worker threads** — one per Cooja container.
4. **Drain the pending queue** — all simulations already in `WAITING` state are enqueued immediately.
5. **Start the watcher thread** — listens to MongoDB Change Streams for new `WAITING` generations and enqueues their simulations in real time.
6. **Block** on the main thread (keyboard interrupt exits cleanly).

---

## Components

### `master-node.py`

The entry point. Holds the `Settings` dataclass, all core functions, and `main()`.

| Function | Responsibility |
|---|---|
| `Settings.from_env()` | Reads all configuration from environment variables |
| `recover_stuck_simulations()` | Marks crashed RUNNING simulations as ERROR on startup |
| `prepare_simulation_files()` | Downloads `.csc`, `.dat`, and source files from GridFS to a temp dir |
| `run_cooja_simulation()` | Executes Cooja via SSH, tails stdout/stderr, retrieves the log |
| `simulation_worker()` | Thread loop: dequeues a simulation, runs it (real or synthetic mode) |
| `start_workers()` | Spawns N worker threads, one per container |
| `load_initial_waiting_jobs()` | Enqueues all WAITING simulations present at startup |
| `_check_and_close_generation()` | After each simulation completes, checks if the whole generation is DONE or ERROR |
| `main()` | Orchestrates startup, watcher, and main loop |

### `lib/mongowatch.py`

Wraps the MongoDB Change Stream subscription for generations.

- Watches the `generations` collection for documents that transition to `WAITING`.
- For each event, fetches all pending simulations of that generation and puts them on the shared queue.
- Marks the generation as `RUNNING` once its simulations are enqueued.

### `lib/sshscp.py`

Thin wrappers around Paramiko and SCP:

- `create_ssh_client()` — opens an SSH connection to a Cooja container.
- `send_files_scp()` — transfers local files to the container's working directory.

### `lib/synthetic_data.py`

Optional execution mode (`ENABLE_DATA_SYNTHETIC=true`) that replaces actual Cooja runs with benchmark functions.
Supported benchmarks: **DTLZ2** (default), **ZDT1**, **SCH1**.
Useful for testing the optimization pipeline without launching the simulator.

---

## Simulation Execution Flow

```mermaid
flowchart TD
    START([master-node starts]) --> RECOVER[Recover stuck simulations\nRUNNING → ERROR]
    RECOVER --> WORKERS[Start N worker threads\none per Cooja container]
    WORKERS --> DRAIN[Enqueue all WAITING\nsimulations from DB]
    DRAIN --> WATCH[Start mongowatch thread\nChange Stream on generations]
    WATCH --> IDLE([Main thread sleeps\nwaiting for interrupt])

    %% mongowatch path
    WATCH -->|New WAITING generation| ENQUEUE[Fetch pending simulations\nfor that generation]
    ENQUEUE --> MARK_RUN_GEN[Mark generation RUNNING]
    MARK_RUN_GEN --> QUEUE[(Shared simulation queue)]

    %% worker path
    QUEUE -->|sim dequeued| SYNTH{ENABLE_DATA\nSYNTHETIC?}
    SYNTH -->|yes| BENCH[Run benchmark function\nDTLZ2 / ZDT1 / SCH1]
    BENCH --> DONE_SIM

    SYNTH -->|no| PREP[prepare_simulation_files\nDownload .csc + .dat +\nsource files from GridFS]
    PREP -->|download failed| ERR_PREP[mark_error simulation]
    ERR_PREP --> CLOSE_GEN

    PREP -->|ok| SCP[Send files via SCP\nto Cooja container]
    SCP --> SSH[SSH: run Cooja JAR\n--no-gui simulation.csc]
    SSH --> MONITOR[Monitor stdout/stderr\nuntil process exits]
    MONITOR --> GETLOG[SCP: retrieve COOJA.testlog]
    GETLOG --> UPLOAD_LOG[Upload log to GridFS]
    UPLOAD_LOG --> CONVERT[Convert log → CSV\ncooja_files.convert_cooja_log_to_csv]
    CONVERT --> UPLOAD_CSV[Upload CSV to GridFS]
    UPLOAD_CSV --> METRICS{CSV exists\nand non-empty?}
    METRICS -->|yes| CALC[Calculate network metrics\nstatistics.evaluate_config]
    CALC --> DONE_SIM[mark_done simulation\nlog_id + csv_id + metrics]
    METRICS -->|no| ERR_CSV[mark_error simulation\nCSV missing or empty]
    ERR_CSV --> CLOSE_GEN

    DONE_SIM --> CLEANUP[Delete local temp files]
    CLEANUP --> CLOSE_GEN[_check_and_close_generation]

    CLOSE_GEN --> ALL_DONE{All simulations\nDONE?}
    ALL_DONE -->|yes| GEN_DONE[mark_done generation\n→ triggers mo-engine\nChange Stream]
    ALL_DONE -->|no active left| GEN_ERR[mark_error generation]
    ALL_DONE -->|still active| WAIT([Wait for remaining\nsimulations])
```

---

## Generation State Machine

```mermaid
stateDiagram-v2
    [*] --> WAITING : inserted by mo-engine
    WAITING --> RUNNING : mongowatch enqueues simulations
    RUNNING --> DONE : all simulations DONE
    RUNNING --> ERROR : no active simulations left\nbut at least one ERROR
```

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/?replicaSet=rs0` | MongoDB connection string (replica set required for Change Streams) |
| `DB_NAME` | `simlab` | Database name |
| `IS_DOCKER` | `false` | If `true`, hostnames are `cooja1..coojaΝ` on port 22; otherwise `localhost` on ports `2231+i` |
| `NUMBER_OF_CONTAINERS` | `3` | Number of Cooja containers (= number of worker threads) |
| `SIM_TIMEOUT_SEC` | `3600` | SSH command timeout and stuck-simulation cutoff in seconds. Set to `0` to disable recovery |
| `ENABLE_DATA_SYNTHETIC` | `false` | Replace Cooja execution with a benchmark function |
| `BENCH` | `DTLZ2` | Benchmark to use in synthetic mode: `DTLZ2`, `ZDT1`, or `SCH1` |
| `NOISE_STD` | `0.0` | Standard deviation of Gaussian noise added to synthetic objectives |

---

## File Layout

```
master-node/
├── master-node.py          # Entry point and core orchestration logic
└── lib/
    ├── mongowatch.py       # MongoDB Change Stream watcher for generations
    ├── sshscp.py           # SSH/SCP helpers (Paramiko)
    └── synthetic_data.py   # Benchmark-based simulation replacement
```

---

## Dependencies

- **paramiko** + **scp** — SSH client and file transfer
- **pandas** — CSV parsing for log conversion
- **pymongo** + **bson** — MongoDB driver
- **pylib** (internal) — shared repositories, models, metrics, and Cooja file utilities

---

## Interaction with Other Services

| Service | How it interacts |
|---|---|
| **mo-engine** | Inserts `Generation` documents with status `WAITING`, which triggers the Change Stream watcher |
| **MongoDB** | Persistence for simulations, generations, and file storage (GridFS) |
| **Cooja containers** | Receives simulation files via SCP, runs Cooja, returns `COOJA.testlog` |
| **rest-api** | Reads simulation and generation status written by master-node; no direct communication |
