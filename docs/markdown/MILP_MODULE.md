# MILP Module (milp-engine)

The **milp-engine** solves MILP design models over a parameter grid and hands
the resulting unique topologies to the existing simulation pipeline as a
**batch experiment**. It never talks to Cooja directly:

```
GUI ── POST /milp/sweeps ──▶ REST API ──▶ MongoDB (milp_sweeps)
                                              │ change stream
                                              ▼
                                        milp-engine
                                        │ parameter sweep (Gurobi/HiGHS)
                                        │ genotype dedup + checkpoint
                                        ▼
                              experiment {strategy: "batch", chromosomes}
                                              │ change stream
                                              ▼
                      mo-engine ─▶ master-node ─▶ cooja workers ─▶ Pareto/GUI
```

## Models

| Key | Problem | Parameters |
|---|---|---|
| `milp_p2_mobile` | `problem2` (mobile coverage) | `C0`, `kdecay`, `B`, `w_install`, `duration` |

The catalog lives in `pylib/config/milp_models.py` (single source for the API
and the engine). Genotype convention: bit *i* of a solution mask refers to
`problem["candidates"][i]` — identical to `ChromosomeP2.mask`.

Mobile trajectories are discretized exactly like the simulator's
`positions.dat` (`pylib/cooja_builder/parse_json_pos_dat.py`); see
`milp-engine/lib/trajectory.py`. This keeps MILP topologies and Cooja metrics
consistent.

## Solver backends

- **gurobi** (default): requires a license for real-size instances; the pip
  `gurobipy` ships a size-limited trial (~2000 vars/constraints).
- **highs**: open source, no license, first-class fallback. When a sweep is
  created with `solver.allow_fallback: true` (default), an unavailable Gurobi
  silently falls back to HiGHS.

The engine probes both at startup and publishes the result to
`milp_engine_status`, exposed at `GET /milp/status` and shown in the GUI
(Models page).

## Gurobi licensing in containers

Named-user academic licenses do **not** work inside Docker. Use one of:

1. **WLS (Web License Service)** — recommended; free academic WLS licenses at
   <https://license.gurobi.com>. Either mount the license file:

   ```yaml
   # docker-compose.yaml (milpengine service)
   environment:
     - GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
   volumes:
     - ./gurobi.lic:/opt/gurobi/gurobi.lic:ro
   ```

   or pass the WLS credentials directly:

   ```yaml
   environment:
     - GRB_WLSACCESSID=...
     - GRB_WLSSECRET=...
     - GRB_LICENSEID=...
   ```

   WLS needs outbound internet access from the container.

2. **Compute Server / Token Server** — point `gurobi.lic` at the server
   (`COMPUTESERVER=...` / `TOKENSERVER=...`) and mount it as above.

3. **No license** — the engine still works: gurobipy trial for small
   instances, HiGHS for everything else (`MILP_SOLVER=highs` to prefer it).

## Creating a sweep

Via the GUI: **Models → open a model → New parameter sweep**. Pick a problem
draft (created in the Problem Editor), choose which parameters to sweep,
configure the solver and the batch-experiment options (objectives, MAC
protocols, source repositories), and launch.

Via the API:

```json
POST /api/v1/milp/sweeps
{
  "name": "P2 sweep",
  "model_key": "milp_p2_mobile",
  "problem": { "name": "problem2", "...": "exported problem JSON" },
  "parameter_grid": { "C0": [10, 110, 310, 610, 1010], "kdecay": [0.9, 0.5, 0.25, 0.1], "B": [1, 25, 50, 75, 100] },
  "fixed_parameters": { "w_install": 1e6 },
  "solver": { "backend": "gurobi", "time_limit_s": 300, "mip_gap": 0.01, "allow_fallback": true },
  "batch_options": {
    "objectives": [{ "metric_name": "latency", "goal": "min" }],
    "simulation": { "duration": 180, "random_seeds": [42] },
    "mac_protocols": [0],
    "source_repository_options": { "csma": "<repo id>" },
    "data_conversion_config": { "node_col": "node", "time_col": "root_time_now", "metrics": ["..."] }
  }
}
```

Every grid combination is recorded in the sweep document (`solutions`),
including duplicates and infeasible points — the params→topology mapping is a
result in itself. Progress and dedup state are checkpointed after every solve,
so an interrupted engine resumes exactly where it stopped (including sweeps
orphaned in `Running` after a crash).

When the sweep finishes, the engine creates one batch experiment containing
one chromosome per unique genotype (× selected MAC protocols) and links it to
the sweep (`experiment_id`) and to the campaign, when `campaign_id` was given.

## Operations

- Cancel: `PATCH /milp/sweeps/{id}/cancel` (cooperative; finishes the current
  solve first).
- Delete: `DELETE /milp/sweeps/{id}` (blocked while `Running`).
- Grid guardrail: requests above `MILP_MAX_COMBINATIONS` (default 10000)
  combinations are rejected by the API.

## Tests

```bash
cd milp-engine && python -m pytest      # model, solvers, sweep, runner, handoff
cd rest-api   && .venv/bin/python -m pytest tests/test_milp.py
```

Backend-dependent tests are parametrized over the available solvers and skip
when neither `gurobipy` nor `highspy` is importable.
