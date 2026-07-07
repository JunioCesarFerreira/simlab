# Synthetic Benchmark Mode

SimLab can evaluate individuals against **classical multi-objective benchmark
functions** (DTLZ2, ZDT1, SCH1) instead of running real Cooja simulations. In
this mode the master-node computes objective values analytically from the
chromosome, skipping all container orchestration, SSH, log parsing and CSV
conversion. It is intended for **fast validation of the optimization pipeline**
(NSGA-II/III, Random Search, operators, convergence metrics) without the cost of
network simulation.

There are two ways to enable it:

| Scope | Where it is configured | Precedence |
| ----- | ---------------------- | ---------- |
| **Per-experiment** (recommended) | `parameters.simulation.synthetic` in the experiment document — set through the **Synthetic Instances** GUI editor | Highest |
| **Global fallback** | `ENABLE_DATA_SYNTHETIC` / `BENCH` / `NOISE_STD` environment variables on the master-node | Used only when the experiment has no `synthetic` block |

---

## 1. Per-experiment configuration (GUI)

The **Synthetic Instances** page (sidebar → *Synthetic*) lets you define a
benchmark instance visually and launch an experiment from it — analogous to the
Problem Editor + Launch Wizard flow.

The editor produces an experiment whose `parameters.simulation` contains:

```jsonc
{
  "simulation": {
    "duration": 1,
    "random_seeds": [42],
    "synthetic": {
      "enabled": true,
      "bench": "DTLZ2",      // "DTLZ2" | "ZDT1" | "SCH1"
      "noise_std": 0.0
    }
  }
}
```

The instance is encoded as a **P1 problem** (relay placement) with no mobile
nodes and `min_coverage_percentage = 0.0`, so the optimizer freely generates
relay positions that act as the benchmark's decision variables:

- `number_of_relays = ceil(nVars / 2)` — each relay contributes an `(x, y)` pair.
- Genome length = `2 · number_of_relays` decision variables (the **sink is
  excluded** — it is fixed infrastructure, not a decision variable).
- The `region` bounding box (`parameters.problem.region`, format
  `[xmin, ymin, xmax, ymax]`) normalizes every coordinate to `[0,1]` before it is
  fed to the benchmark function.

When an experiment has `synthetic.enabled = true`:

- The **mo-engine** skips CSC/firmware file generation and the source-repository
  lookup for every simulation (no Cooja artifacts are produced).
- The **master-node** routes the job to `run_synthetic_simulation()` instead of
  the Cooja worker.
- The **GUI** shows an amber `⬡ Synthetic — <BENCH>` badge on the experiment card
  and detail header. For DTLZ2 with `M = 3` the editor renders an interactive 3D
  preview of the theoretical Pareto front (unit quarter-sphere).

You can also enable synthetic mode from the **existing Launch Wizard** (Problem
Editor flow): step 3 (*Simulation*) has an optional *"Synthetic benchmark mode"*
toggle. This reuses the same `parameters.simulation.synthetic` block.

---

## 2. Global fallback (environment variables)

When an experiment does **not** carry a `synthetic` block, the master-node falls
back to environment variables inside its worker loop:

```python
# master-node.py (simplified)
syn_cfg = exp_doc["parameters"]["simulation"].get("synthetic", {})
mode = syn_cfg.get(
    "enabled",
    os.getenv("ENABLE_DATA_SYNTHETIC", "False").lower() == "true",
)
if mode:
    bench     = syn_cfg.get("bench") or os.getenv("BENCH", "DTLZ2")
    noise_std = float(syn_cfg.get("noise_std", os.getenv("NOISE_STD", "0.0")))
    run_synthetic_simulation(sim, mongo, bench=bench, noise_std=noise_std)
    continue
```

| Variable                | Meaning / effect                                | Default   |
| ----------------------- | ----------------------------------------------- | --------- |
| `ENABLE_DATA_SYNTHETIC` | Enable synthetic mode globally (`True`/`False`) | `"False"` |
| `BENCH`                 | Benchmark function: `DTLZ2`, `ZDT1`, `SCH1`     | `"DTLZ2"` |
| `NOISE_STD`             | Std-dev of additive Gaussian noise per objective| `"0.0"`   |

`docker-compose.yaml` (master-node service):

```yaml
services:
  masternode:
    environment:
      - ENABLE_DATA_SYNTHETIC=True
      - BENCH=DTLZ2
      - NOISE_STD=0.05
      - MONGO_URI=mongodb://mongodb:27017/?replicaSet=rs0
      - DB_NAME=simlab
      - IS_DOCKER=True
```

> Note: per-experiment config always wins. The env vars only apply to experiments
> that were created without a `synthetic` block (e.g. legacy experiments, or the
> Launch Wizard with the toggle off).

---

## 3. Benchmark functions

All benchmarks are **minimization** problems. Decision variables are the genome
coordinates normalized to `[0,1]^n` via the region bounding box.

| ID    | Objectives `M` | Pareto front                                   | Reference |
| ----- | -------------- | ---------------------------------------------- | --------- |
| DTLZ2 | `M ≥ 2` (needs `n ≥ M-1`) | Unit hypersphere segment in the first orthant (`‖f‖₂ = 1`) | Deb, Thiele, Laumanns & Zitzler (2005) |
| ZDT1  | exactly `2`    | Convex curve `f₂ = 1 − √f₁`                     | Zitzler, Deb & Thiele (2000) |
| SCH1  | exactly `2`    | `f₁ = x²`, `f₂ = (x−2)²` (Schaffer)             | Schaffer (1985) |

Notes:

- **DTLZ2** — `g = Σ (xᵢ − 0.5)²` over the last `k = n − (M−1)` variables; the
  front is reached when those variables equal `0.5`.
- **ZDT1** — `f₁ = x₀`, `g = 1 + 9·Σ(x₁…xₙ₋₁)/(n−1)`, `f₂ = g·(1 − √(f₁/g))`.
- **SCH1** — uses the first decision variable mapped back to the region's x-axis
  scale: `raw_x = x_min + x01[0]·(x_max − x_min)`.
- Objective names/order come from `parameters.objectives[].metric_name`, so the
  stored `{name: value}` dict matches exactly what the mo-engine reads back.

---

## 4. REST API

`GET /synthetic/benchmarks` returns the catalogue of available benchmark
descriptors (id, label, min/max objectives, description, `n_min_formula`). The
GUI uses it to render the benchmark selector.

The experiment list endpoint (`GET /experiments/by-status/{status}`) now also
returns `is_synthetic` and `synthetic_bench` for each experiment, so the GUI can
render the synthetic badge without fetching the full document.

---

## 5. Flow comparison

* **Real mode**: worker downloads simulation files via GridFS → SCP to a Cooja
  container → runs Cooja over SSH → retrieves logs → converts to CSV → computes
  metrics/objectives → marks the simulation done.

* **Synthetic mode**: worker skips all of the above and calls
  `run_synthetic_simulation(sim, mongo, bench, noise_std)`, which:

  1. Marks the simulation **running**.
  2. Extracts the genome (relay coordinates) from `simulationElements.fixedMotes`.
  3. Normalizes to `[0,1]^n` using `parameters.problem.region`.
  4. Evaluates the benchmark (+ optional Gaussian noise).
  5. Writes objectives keyed by `metric_name` and marks the simulation **done**.
  6. Closes the generation when all its simulations are done.

---

## 6. Convergence metrics against the true front

Because the synthetic benchmarks have a Pareto front known in closed form,
`pareto-analysis/compute_hv_gd.py` can measure **Generational Distance (GD)**
against the analytical (true) front instead of the experiment's own final front
(which would drive GD trivially to zero on the last generation):

```bash
python compute_hv_gd.py --expid <id> \
  --objectives f1 f2 f3 --minimize true true true \
  --true-front-bench DTLZ2 --true-front-m 3
```

Without `--true-front-bench` the behavior is unchanged (self-reference front).
The analytical fronts live in `pareto-analysis/lib/true_fronts.py` (DTLZ2 =
unit hypersphere segment; ZDT1 = `1 − √f₁`; SCH1 = `x²`/`(x−2)²`, `x∈[0,2]`).

> GD here is the **RMS variant** `sqrt((1/N)·Σ dᵢ²)` (Schütze et al., 2012),
> not Van Veldhuizen's classic `(1/N)·(Σ dᵢᵖ)^(1/p)`. HV uses the `moocore`
> library with a reference point set to the worst feasible objective + margin.

---

## 7. Verifying it works

1. **Logs** — `docker compose logs masternode` shows
   `Starting benchmark simulation <oid> (bench=… noise_std=…)` and *no* SSH/Cooja
   activity.
2. **Database** — simulations reach `done` with a populated `objectives` map,
   with `csc_file_id`/`pos_file_id`/`source_repository_id` set to `null`.
3. **GUI** — the experiment shows the amber `⬡ Synthetic — <BENCH>` badge; charts
   and Pareto analysis behave exactly as for real experiments.
4. **Noise sanity** — increasing `noise_std` scatters points around the
   theoretical front.

---

> ⚠️ Synthetic mode bypasses simulation entirely — it produces no CSV logs or
> time-series traces. It is meant for fast validation of the algorithmic
> pipeline, not for full simulation trace analysis.
