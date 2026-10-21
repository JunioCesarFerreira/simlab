# Util

Utility tools for development and experiment management.

## Scripts

### `gdcc.py` — GeneratorDockerComposeCooja

Generates a docker-compose snippet with multiple Cooja simulation containers.

```bash
python gdcc.py <N>
```

| Argument | Description |
|---|---|
| `N` | Number of Cooja containers to generate |

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

```bash
python mongo_backup.py [--uri URI] [--db DB] [--output DIR]
```

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--uri` | `MONGO_URI` | `mongodb://localhost:27017/?replicaSet=rs0` | MongoDB connection URI |
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

Script template for creating experiments via the SimLab REST API. Edit the `body` dict and the `API_KEY` constant before running.

```bash
python create_exp.py
```

---

### `run_pareto_analysis.py` — Run Pareto Analysis

Queries MongoDB for all completed experiments (`status: Done`) and runs `plot_pareto_results.py` for each one, uploading the generated plots back via the REST API.

Experiments with a number of objectives other than 3 are skipped (limitation of the analysis script).

```bash
python run_pareto_analysis.py [--uri URI] [--db DB] [--api-base URL] [--api-key KEY]
```

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--uri` | `MONGO_URI` | `mongodb://localhost:27017/?replicaSet=rs0` | MongoDB connection URI |
| `--db` | `DB_NAME` | `simlab` | Database name |
| `--api-base` | `SIMLAB_API_BASE` | `https://localhost/api/v1` | SimLab REST API base URL (through the nginx proxy) |
| `--api-key` | `SIMLAB_API_KEY` | `api-password` | SimLab API key |

**Example (remote server):**

MongoDB is bound to `127.0.0.1` on the SimLab host, so reach it through an SSH
tunnel (`ssh -L 27017:localhost:27017 server`). The REST API goes through the
reverse proxy over HTTPS.

```bash
MONGO_URI=mongodb://localhost:27017/?replicaSet=rs0 \
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

Registers a host, its SSH access and its workers, from the command line
instead of the Infrastructure page.

```bash
python util/register_host.py list
python util/register_host.py add vm7 --address host.example --port 22 \
    --user simlab --key ~/.ssh/simlab_vm7
```

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

### `backfill_dodag.py` — Recompute DODAG summaries

Walks finished simulations and derives the RPL DODAG from their stored
testlogs. It is a dry run by default and prints what it *would* change.

What a legacy log cannot give back is in
[FIRMWARE_TRACEABILITY.md](../docs/markdown/FIRMWARE_TRACEABILITY.md) and in
the script's own docstring: only the approximate, first-contact formation is
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
could be made), the simulation records `unfetched_result`, and the worker is
**held** out of the pool until something resolves it. Nothing comes back for it
automatically. This is the other half of that contract.

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
| `release` | clears the hold so the health cycle measures the worker again | the worker |

| Env var | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/?replicaSet=rs0` | MongoDB connection URI |
| `DB_NAME` | `simlab` | Database name |
| `INFRA_KEY_PATH` | the master-node's keystore path | the key that opens the host credential |

**The order is enforced, not advisory.** `release` refuses while the simulation
still carries an unresolved `unfetched_result`: releasing the worker first lets
the next run delete the bytes, and that is exactly the loss the hold exists to
prevent. Resolve it — `publish` or `discard` — and then release.

**`publish` does not decide anything about the experiment.** It stores the
artifacts and records `recovered_result` with the time, the remote path and the
tool's name. It does **not** compute the metrics and does **not** change the
simulation's status: whether a result collected by hand after the fact should
feed the optimiser is a decision about the experiment, and the point of the
hold is that it is taken by a person. A simulation already in `Done` is refused
outright, because recovering over it would replace a measurement.

Three things to know before using it:

- **It is an operator tool, not an API.** It reads and writes MongoDB directly
  and opens SSH to the worker, so it needs the private key the master-node
  holds: run it where the master-node runs. `docker cp util/recover_result.py
  master_node:/tmp/` and `docker exec master_node python /tmp/recover_result.py`
  works, and is how the procedure was exercised.
- **It cannot recover what was never preserved.** If `remote_file` is null the
  second session never opened, so no copy was made. The original may still be
  under `COOJA.testlog` and `fetch --original` will try it — but any run that
  has since used the worker has deleted it. The worker being held is what makes
  that unlikely rather than impossible.
- **The health cycle will not release the worker for you.** A
  `collection_pending` hold is kept until the simulation records
  `recovered_result` or `unfetched_result.discarded`, so a resolver that finds
  the container healthy and idle still leaves the hold in place.
