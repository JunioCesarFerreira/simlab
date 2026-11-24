# Util

Utility tools for development and experiment management.

| Script | Purpose | Writes |
|---|---|---|
| [`gdcc.py`](#gdccpy--generatordockercomposecooja) | Generate Cooja compose services | stdout |
| [`mdr.py`](#mdrpy--monitordockerregister) | Record container statistics | a CSV |
| [`mongo_backup.py`](#mongo_backuppy--mongodb-backup) | Export the database to JSON | local files |
| [`gridfs_compact.py`](#gridfs_compactpy--gridfs-compaction) | Compress or restore stored files | GridFS |
| [`create_exp.py`](#create_exppy--create-experiment) | Template to create an experiment | through the API |
| [`run_pareto_analysis.py`](#run_pareto_analysispy--run-pareto-analysis) | Plot every finished experiment | through the API |
| [`register_host.py`](#register_hostpy--register-a-remote-cooja-host) | Register and manage Cooja hosts | MongoDB |
| [`backfill_dodag.py`](#backfill_dodagpy--recompute-dodag-summaries) | Recompute DODAG summaries | MongoDB, with `--apply` |
| [`recover_result.py`](#recover_resultpy--recover-a-result-that-was-produced-and-never-collected) | Resolve a result that was not collected | MongoDB, GridFS |

## Before running

- **MongoDB from the host.** Every script defaults to
  `mongodb://localhost:27017/?directConnection=true`, which reaches the stack's
  replica set from the host. `MONGO_URI` or `--uri` override it.
- **API key.** The scripts that call the API default to `simlab-api-key42`, the
  default of `docker-compose.yaml`. Set `SIMLAB_API_KEY` to the key of your
  deployment.
- **Dependencies.** There is no `requirements.txt` here. `pymongo` and
  `requests` cover the simple scripts; those that import `pylib` run in the
  master-node's environment (`master-node/requirements.txt`).
- The paths in the examples are relative to this directory, except where they
  start with `util/`.

## Scripts

### `gdcc.py` — GeneratorDockerComposeCooja

Prints a docker-compose snippet with multiple Cooja simulation containers
([output_gdcc.yaml](output_gdcc.yaml) is a saved sample).

```bash
python gdcc.py <N>
```

| Argument | Description |
|---|---|
| `N` | Number of Cooja containers to generate |

> **Edit the output before using it.** It publishes SSH (`root`/`root`) on every
> interface, as `"2231:22"`: write `"127.0.0.1:2231:22"`. It also has no
> `simlab.*` labels, without which the master-node does not discover the
> containers and the telemetry does not see them; copy them from a `cooja`
> service of [docker-compose.yaml](../docker-compose.yaml).

---

### `mdr.py` — MonitorDockerRegister

Collects Docker container stats (CPU, memory, network I/O) and saves them to a CSV file.

```bash
python mdr.py [--interval N] [--duration N] [--output FILE]
```

| Argument | Default | Description |
|---|---|---|
| `--interval` | `5` | Sampling interval in seconds |
| `--duration` | `300` | Total collection duration in seconds |
| `--output` | `docker_stats.csv` | Output CSV file path |

---

### `mongo_backup.py` — MongoDB Backup

Dumps every collection in the SimLab database to individual JSON files organized by timestamp.

> **An export, not a restorable backup.** Ids and dates become strings and
> there is no restore script; use `mongodump` for a backup. The export includes
> the GridFS chunks, as stored (possibly compressed), and the sealed host
> credentials.

```bash
python mongo_backup.py [--uri URI] [--db DB] [--output DIR]
```

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--uri` | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | MongoDB connection URI |
| `--db` | `DB_NAME` | `simlab` | Database name |
| `--output` | — | `mongo_backup` | Root output directory |

Output structure: `<output>/<timestamp>/<collection>/<_id>.json`

---

### `gridfs_compact.py` — GridFS Compaction

Re-encodes stored GridFS files with zstd, in place and reversibly. New uploads
are already compressed by `pylib/db/gridfs.py`; this backfills the files that
predate it.

```bash
python gridfs_compact.py --dry-run
python gridfs_compact.py --filename sim_result.log --limit 100
python gridfs_compact.py                      # everything eligible
python gridfs_compact.py --revert             # back to plain bytes
python gridfs_compact.py --recover            # restore orphaned backups
```

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--uri` | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | MongoDB connection URI |
| `--db` | `DB_NAME` | `simlab` | Database name |
| `--filename` | — | all | Only files with this exact filename |
| `--limit` | — | all | Stop after N files |
| `--level` | — | `12` | zstd level |
| `--dry-run` | — | off | Report what would be converted, change nothing |
| `--revert` | — | off | Decompress back to plain bytes |
| `--recover` | — | off | Restore backups left by an interrupted run |

Every file keeps its `_id`, so no document that references it is touched —
which is what makes `--revert` possible. Per file the tool hashes the original,
checks the new frame decodes back to that hash *before* deleting anything,
stores an untouched backup, rewrites under the same `_id`, verifies the
read-back, and only then drops the backup. A crash in between leaves the backup
for `--recover`; an interrupted run can simply be restarted, since files that
already carry a `metadata.compression` block are skipped.

> **Run it with the stack idle.** Each file is briefly deleted and re-inserted
> under the same `_id`; a reader hitting that window gets a missing file. The
> tool also holds one file fully in memory at a time, so peak usage tracks the
> largest file rather than the collection.

---

### `create_exp.py` — Create Experiment

Script template for creating experiments via the SimLab REST API. Edit the
`body` dict before running: it is an example, with ids of another database
(`duration` is in simulated minutes). The endpoint, key and TLS check come from
`SIMLAB_API_BASE`, `SIMLAB_API_KEY`, `SIMLAB_CA_BUNDLE` and `SIMLAB_TLS_VERIFY`.

```bash
python create_exp.py
```

---

### `run_pareto_analysis.py` — Run Pareto Analysis

Queries MongoDB for all completed experiments (`status: Done`) and runs `plot_pareto_results.py` for each one, uploading the generated plots back via the REST API.

Experiments with a number of objectives other than 3 are skipped (limitation of the analysis script).
It passes no analytical front, so synthetic experiments are also measured
against their own final front.

```bash
python run_pareto_analysis.py [--uri URI] [--db DB] [--api-base URL] [--api-key KEY] [--python PYTHON]
```

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--uri` | `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | MongoDB connection URI |
| `--db` | `DB_NAME` | `simlab` | Database name |
| `--api-base` | `SIMLAB_API_BASE` | `https://localhost/api/v1` | SimLab REST API base URL (through the nginx proxy) |
| `--api-key` | `SIMLAB_API_KEY` | `simlab-api-key42` | SimLab API key |
| `--python` | `SIMLAB_PYTHON` | `python3` | Interpreter for `plot_pareto_results.py`; it needs [pareto-analysis](../pareto-analysis/README.md)'s dependencies |

**Example (remote server):**

MongoDB is bound to `127.0.0.1` on the SimLab host, so reach it through an SSH
tunnel (`ssh -L 27017:localhost:27017 server`). The REST API goes through the
reverse proxy over HTTPS.

```bash
MONGO_URI='mongodb://localhost:27017/?directConnection=true' \
SIMLAB_API_BASE=https://server/api/v1 \
SIMLAB_CA_BUNDLE=../nginx/certs/simlab.crt \
SIMLAB_API_KEY=my-key \
python run_pareto_analysis.py
```

`SIMLAB_CA_BUNDLE` is only needed when the server uses the self-signed
certificate; drop it once a CA-issued one is installed. As a last resort,
`SIMLAB_TLS_VERIFY=false` skips verification.

---

### `register_host.py` — Register a remote Cooja host

Registers a host and its SSH access, from the command line instead of the
Infrastructure page, and asks the master-node to discover its workers.

```bash
python util/register_host.py list
python util/register_host.py add vm7 --address host.example --port 22 \
    --user simlab --key ~/.ssh/simlab_vm7
```

| Command | What it does | Options |
|---|---|---|
| `list` | hosts and their occupancy | `-v` (workers and last check), `--json` |
| `add <name>` | register a host and request a check | see below |
| `check <name>` | request a connectivity check | — |
| `discover <name>` | request container discovery | — |
| `enable <name>` | add the host to the pool | `--off` drains it out |

Options of `add`: `--address` and `--user` (required), `--port` (22), `--key`
or `--password-stdin`, `--container-user` (`root`), `--container-port` (22),
`--container-password-stdin`, `--address-mode` (`container_ip`),
`--label-filter`, `--name-prefix`, `--disabled`, `--replace`, `--no-check`.

Two things to know before using it:

- **It talks to MongoDB directly**, not through the API. The page's route goes
  through validation the API performs; this one does not, so a value it accepts
  may be one the API would have refused.
- **`list --json` prints the host document as stored, including the sealed
  credential envelope.** The API redacts that field; this does not. The
  ciphertext is useless without the master-node's private key, but it does not
  belong in a log or a ticket.

MongoDB is bound to `127.0.0.1` on the SimLab host, so reach it through an SSH
tunnel. Careful with a replica set behind a tunnel: the driver follows the
addresses the set advertises, which are the ones *inside* the deployment — use
`?directConnection=true` when they are not reachable from where you are.

---

### `backfill_dodag.py` — Recompute DODAG summaries

Walks finished simulations and derives the RPL DODAG from their stored
testlogs. It is a dry run by default and prints what it *would* change.

```bash
python util/backfill_dodag.py [--mongo-uri URI] [--db-name DB] [--status Done] \
    [--stability-window-ms 60000] [--force] [--limit N] [--apply]
```

How the DODAG is read from a log is in
[FIRMWARE_METRICS.md §5.7](../docs/markdown/FIRMWARE_METRICS.md#57-reports-and-the-dodag),
and what a legacy log cannot give back in the script's own docstring: only the approximate, first-contact formation is
recoverable, not the exact parent changes, because those were never logged.

Two limits worth knowing in automation: `--limit` applies *before* the
eligibility filters, so a small limit can examine a batch and change nothing;
and an individual log that fails to parse is reported but does not change the
exit status.

---

### `recover_result.py` — Recover a result that was produced and never collected

When a run finishes and its result cannot be fetched — twice, over two
sessions — the master-node stops rather than losing it. The testlog is copied
on the container to a name the next run's cleanup does not know (where the copy
could be made) and the simulation records `unfetched_result`. With a copy, the
worker goes back to the pool. Where no copy could be made (or the record could
not be written), the worker is **held** out of the pool until something resolves
it. Nothing comes back for the bytes automatically. This is the other half of
that contract.

```bash
python util/recover_result.py list
python util/recover_result.py show <simulation_id>
python util/recover_result.py fetch <simulation_id> [--out DIR] [--original]
python util/recover_result.py publish <simulation_id> [--out DIR]
python util/recover_result.py discard <simulation_id> --reason "..."
python util/recover_result.py release <worker_name>
```

| Command | What it does | What it changes |
|---|---|---|
| `list` | the simulations carrying an uncollected result, and the workers held, so the two can be matched | nothing |
| `show` | one record in full: host, worker, container, attempt, pid, directory, file, stage, reason | nothing |
| `fetch` | opens a session to the worker and brings the preserved testlog back | nothing in the database |
| `publish` | fetches, converts to CSV, stores both in GridFS and records them as a **recovered** result | the simulation |
| `discard` | records the decision to give up, with a reason | the simulation |
| `release` | clears the hold so the health cycle measures the worker again; needed only if the worker is held | the worker |

`--out` defaults to `recovered`.

| Env var | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/?directConnection=true` | MongoDB connection URI |
| `DB_NAME` | `simlab` | Database name |
| `INFRA_KEY_PATH` | the master-node's keystore path | the key that opens the host credential |

`release` takes the worker's name and acts on the first held worker that has
it. Names can repeat across hosts: check with `list` first.

**The order is enforced, not advisory.** `release` refuses while the simulation
still carries an unresolved `unfetched_result`: releasing the worker first lets
the next run delete the bytes, and that is exactly the loss the hold exists to
prevent. Resolve it — `publish` or `discard` — and then release.

**`publish` does not decide anything about the experiment.** It stores the
artifacts and records `recovered_result` with the time, the remote path and the
tool's name. It does **not** compute the metrics and does **not** change the
simulation's status: whether a result collected by hand after the fact should
feed the optimiser is a decision about the experiment, to be taken by a
person. A simulation already in `Done` is refused
outright, because recovering over it would replace a measurement.

Three things to know before using it:

- **It is an operator tool, not an API.** It reads and writes MongoDB directly
  and opens SSH to the worker, so it needs the private key the master-node
  holds: run it where the master-node runs. `docker cp util/recover_result.py
  master_node:/tmp/` and `docker exec master_node python /tmp/recover_result.py`
  works, and is how the procedure was exercised. With a wrong `INFRA_KEY_PATH`
  the keystore loader **creates** a new key at that path, which then opens no
  credential.
- **It cannot recover what was never preserved.** If `remote_file` is null the
  second session never opened, so no copy was made. The original may still be
  under `COOJA.testlog` and `fetch --original` will try it — but any run that
  has since used the worker has deleted it. The worker being held is what makes
  that unlikely rather than impossible.
- **The health cycle will not release the worker before a decision.** A
  `collection_pending` hold is kept until the simulation records
  `recovered_result` or `unfetched_result.discarded`, or is deleted; the
  conditions are in the
  [master-node README](../master-node/README.md#a-worker-is-held-and-stays-held).
