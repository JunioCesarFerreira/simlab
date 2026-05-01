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
| `--api-base` | `SIMLAB_API_BASE` | `http://localhost:8000/api/v1` | SimLab REST API base URL |
| `--api-key` | `SIMLAB_API_KEY` | `api-password` | SimLab API key |

**Example (remote server):**
```bash
MONGO_URI=mongodb://server:27017/?replicaSet=rs0 \
SIMLAB_API_BASE=http://server:8000/api/v1 \
SIMLAB_API_KEY=my-key \
python run_pareto_analysis.py
```