# Synthetic Benchmark Mode

SimLab can evaluate individuals against **classical multi-objective benchmark
functions** (DTLZ2, ZDT1, SCH1) instead of running real Cooja simulations,
skipping all container orchestration, SSH, log parsing and CSV conversion. It is
intended for **fast validation of the optimization pipeline** (NSGA-II/III,
Random Search, operators, convergence metrics) without the cost of network
simulation.

## Two evaluation paths — read this first

Synthetic evaluation is implemented in **two different places**, and knowing
which path your experiment takes matters both for performance and for reading
the code:

| Path | Problem encoding | Who evaluates | Simulation documents? |
| ---- | ---------------- | ------------- | --------------------- |
| **1. In-process analytical (P0)** — current, preferred | `problem0` — decision vector `x ∈ [0,1]ⁿ` | **mo-engine**, inside the strategy loop ([`lib/strategy/analytical.py`](../../mo-engine/lib/strategy/analytical.py)) | **None** — the master-node is never involved |
| **2. Master-node synthetic** — legacy / fallback | any (see triggers below) | **master-node** ([`lib/synthetic_data.py`](../../master-node/lib/synthetic_data.py)) | Yes — enqueued as usual, but without CSC/firmware artifacts |

**Path 1** is what you get today when you launch an experiment from the
**Synthetic Instances** GUI page with any of the evolutionary or random-search
strategies. The benchmark is a closed-form function of the decision vector, so
the mo-engine evaluates it directly in its own process: no `Simulation`
document is written, no Change-Stream round-trip happens, and the master-node
never sees the experiment.

**Path 2** still exists because three scenarios cannot use the fast path:

1. **`batch` strategy** — `BatchStrategy` has no analytical fast-path; it
   always enqueues `Simulation` documents (skipping only the CSC/firmware
   artifacts), and the master-node evaluates them synthetically.
2. **Legacy P1-encoded synthetic experiments** — older experiments where the
   benchmark variables were encoded as relay coordinates of a `problem1`
   instance. The master-node reconstructs the genome from
   `simulationElements.fixedMotes` and normalizes it to `[0,1]ⁿ` via the
   region bounding box.
3. **Global env-var override** — setting `ENABLE_DATA_SYNTHETIC=true` on the
   master-node turns *any* incoming simulation into a synthetic one,
   regardless of how the experiment was created (used e.g. for pipeline-wide
   validation runs).

> ⚠️ **Do not remove the master-node synthetic module** while any of the three
> scenarios above is still supported. Conversely, if you are adding a new
> strategy and want synthetic experiments to stay in-process, implement the
> analytical fast-path (see §2) rather than relying on the master-node.

---

## 1. Enabling synthetic mode

There are two configuration scopes; the per-experiment block always wins:

| Scope | Where it is configured | Precedence |
| ----- | ---------------------- | ---------- |
| **Per-experiment** (recommended) | `parameters.simulation.synthetic` in the experiment document — set through the **Synthetic Instances** GUI editor | Highest |
| **Global fallback** | `ENABLE_DATA_SYNTHETIC` / `BENCH` / `NOISE_STD` environment variables on the master-node | Used only when the experiment has no `synthetic` block |

### Per-experiment configuration (GUI)

The **Synthetic Instances** page (sidebar → *Synthetic*) lets you define a
benchmark instance visually and launch an experiment from it — analogous to the
Problem Editor + Launch Wizard flow.

The editor produces an experiment whose `parameters` contain:

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
  },
  "problem": {
    "name": "problem0",       // pure analytical benchmark problem
    "n": 12                   // decision-vector length
  }
}
```

The instance is encoded as a **P0 problem** (`Problem0SyntheticAdapter`): the
chromosome is a flat real-valued decision vector `x ∈ [0,1]ⁿ` — no sink, no
relays, no region round-trip. Genetic operators are the textbook real-coded
pair (Simulated Binary Crossover + Polynomial Mutation), applied per variable
over the unit interval.

> Historical note: earlier versions encoded synthetic instances as a **P1
> problem** (relay placement), mapping each pair of decision variables to a
> relay `(x, y)` position and normalizing through the region bounding box.
> Experiments created that way are still evaluated correctly — by the
> master-node legacy path (§3) — but new instances always use P0.

When an experiment has `synthetic.enabled = true`:

- The **mo-engine** evaluates P0 individuals **in-process** (§2) and, for the
  strategies that do enqueue simulations, skips CSC/firmware file generation
  and the source-repository lookup (no Cooja artifacts are produced).
- The **master-node** routes any synthetic simulation it *does* receive to
  `run_synthetic_simulation()` instead of the Cooja worker (§3).
- The **GUI** shows an amber `⬡ Synthetic — <BENCH>` badge on the experiment
  card and detail header. For DTLZ2 with `M = 3` the editor renders an
  interactive 3D preview of the theoretical Pareto front (unit quarter-sphere).

---

## 2. Path 1 — In-process analytical evaluation (P0)

Implemented in [`mo-engine/lib/strategy/analytical.py`](../../mo-engine/lib/strategy/analytical.py)
and gated by the adapter property `ProblemAdapter.is_analytical` (only
`Problem0SyntheticAdapter` returns `True`).

During `_generation_enqueue`, each strategy checks the adapter **before**
creating any `Simulation` document:

```python
# nsga2.py / nsga3.py / random_search.py (simplified)
if self._problem_adapter.is_analytical:
    _, obj_min = analytical_objectives(
        self._problem_adapter, genome, genome_hash,
        bench=self._bench, noise_std=self._noise_std,
        seeds=self._sim_rand_seeds, ...
    )
    # insert the Individual with objectives already filled,
    # update the genome cache — and enqueue NO simulation.
    continue
```

Key properties of this path:

- **No `Simulation` documents** are created; the generation completes through
  the existing "no simulations pending → mark generation DONE" logic.
- The benchmark is evaluated **once per simulation seed and averaged (mean)**,
  matching the multi-seed aggregation used for simulation-based problems.
- **Noise is reproducible**: the Gaussian-noise RNG is seeded with
  `f"{seed}:{genome_hash}"`, so the same genome under the same seed always
  yields the same objectives and the genome cache stays consistent.
- The **genome cache** works exactly as in real mode: repeated chromosomes
  reuse persisted objectives.

Strategy support matrix:

| Strategy | Analytical fast-path? | Notes |
| -------- | --------------------- | ----- |
| `nsga2`, `nsga3`, `random_search` | ✅ | Implemented natively |
| `nsga2_deap`, `nsga2_pymoo`, `nsga3_deap`, `nsga3_pymoo` | ✅ | Inherit it from the native classes (they override only environmental selection) |
| `batch` | ❌ | Always enqueues simulations → falls back to the master-node path |

---

## 3. Path 2 — Master-node synthetic evaluation (legacy / fallback)

Implemented in [`master-node/lib/synthetic_data.py`](../../master-node/lib/synthetic_data.py).
For every dequeued simulation, the worker resolves the synthetic mode
(per-experiment config first, env vars second) and, when enabled, calls
`run_synthetic_simulation()` instead of the Cooja pipeline:

```python
# master-node.py (simplified)
mode, bench, noise_std = resolve_synthetic_settings(exp_doc)
if mode:
    run_synthetic_simulation(sim, mongo, bench=bench, noise_std=noise_std)
    continue
```

`run_synthetic_simulation()` supports **both encodings**:

- **P0** — the decision vector is read verbatim from
  `parameters.simulationElements.decisionVector` (no scaling). This is the
  case for `batch` experiments over `problem0`.
- **P1-encoded (legacy)** — the genome is reconstructed from the relay
  coordinates in `simulationElements.fixedMotes` (the sink is excluded — it is
  fixed infrastructure, not a decision variable) and normalized to `[0,1]ⁿ`
  via the `parameters.problem.region` bounding box.

### Global fallback (environment variables)

When an experiment does **not** carry a `synthetic` block, the master-node
falls back to environment variables inside its worker loop:

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

> Note: per-experiment config always wins. The env vars only apply to
> experiments that were created without a `synthetic` block (e.g. legacy
> experiments). Also note that the env vars can only affect simulations that
> **reach** the master-node — P0 experiments evaluated in-process by the
> mo-engine never do.

---

## 4. Benchmark functions

All benchmarks are **minimization** problems, implemented once in the shared
[`pylib/benchmarks.py`](../../pylib/benchmarks.py) module (single source of
truth for the mo-engine, the master-node and the rest-api). Decision variables
are either the P0 vector itself or, on the legacy path, the genome coordinates
normalized to `[0,1]ⁿ` via the region bounding box.

| ID    | Objectives `M` | Pareto front                                   | Reference |
| ----- | -------------- | ---------------------------------------------- | --------- |
| DTLZ2 | `M ≥ 2` (needs `n ≥ M-1`) | Unit hypersphere segment in the first orthant (`‖f‖₂ = 1`) | Deb, Thiele, Laumanns & Zitzler (2005) |
| ZDT1  | exactly `2`    | Convex curve `f₂ = 1 − √f₁`                     | Zitzler, Deb & Thiele (2000) |
| SCH1  | exactly `2`    | `f₁ = x²`, `f₂ = (x−2)²` (Schaffer)             | Schaffer (1985) |

Notes:

- **DTLZ2** — `g = Σ (xᵢ − 0.5)²` over the last `k = n − (M−1)` variables; the
  front is reached when those variables equal `0.5`.
- **ZDT1** — `f₁ = x₀`, `g = 1 + 9·Σ(x₁…xₙ₋₁)/(n−1)`, `f₂ = g·(1 − √(f₁/g))`.
- **SCH1** — uses the first decision variable mapped to the benchmark's
  decision domain (configurable via `synthetic.sch1_domain`).
- Objective names/order come from `parameters.objectives[].metric_name`, so the
  stored `{name: value}` dict matches exactly what the mo-engine reads back.

---

## 5. REST API

`GET /synthetic/benchmarks` returns the catalogue of available benchmark
descriptors (id, label, min/max objectives, description, `n_min_formula`). The
GUI uses it to render the benchmark selector.

The experiment list endpoint (`GET /experiments/by-status/{status}`) also
returns `is_synthetic` and `synthetic_bench` for each experiment, so the GUI
can render the synthetic badge without fetching the full document.

---

## 6. Flow comparison

* **Real mode**: the mo-engine enqueues `Simulation` documents → a master-node
  worker downloads simulation files via GridFS → SCP to a Cooja container →
  runs Cooja over SSH → retrieves logs → converts to CSV → computes
  metrics/objectives → marks the simulation done.

* **Synthetic, in-process (P0 + evolutionary/random-search strategies)**: the
  mo-engine evaluates the benchmark directly on the decision vector inside
  `_generation_enqueue`, inserts the `Individual` with objectives already
  filled, and **never creates a `Simulation` document**. The generation is
  closed by the "no simulations pending" path. The master-node is idle.

* **Synthetic, master-node (batch / legacy P1 / env override)**: the mo-engine
  enqueues `Simulation` documents without CSC/firmware artifacts; the worker
  calls `run_synthetic_simulation(sim, mongo, bench, noise_std)`, which:

  1. Marks the simulation **running**.
  2. Obtains the decision vector — verbatim (`decisionVector`, P0) or by
     extracting relay coordinates and normalizing via the region (P1 legacy).
  3. Evaluates the benchmark (+ optional reproducible Gaussian noise).
  4. Writes objectives keyed by `metric_name` and marks the simulation **done**.
  5. Closes the generation when all its simulations are done.

---

## 7. Convergence metrics against the true front

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

## 8. Verifying it works

For the **in-process path (P0)**:

1. **Logs** — `docker compose logs moengine` shows
   `Synthetic mode enabled — skipping CSC/source-repo for simulations.` and the
   generations complete without any master-node activity.
2. **Database** — `individuals` carry populated `objectives`, while the
   `simulations` collection has **no documents** for the experiment.

For the **master-node path** (batch / legacy / env override):

1. **Logs** — `docker compose logs masternode` shows
   `Starting benchmark simulation <oid> (bench=… noise_std=…)` and *no*
   SSH/Cooja activity.
2. **Database** — simulations reach `done` with a populated `objectives` map,
   with `csc_file_id`/`pos_file_id`/`source_repository_id` set to `null`.

In both cases:

3. **GUI** — the experiment shows the amber `⬡ Synthetic — <BENCH>` badge;
   charts and Pareto analysis behave exactly as for real experiments.
4. **Noise sanity** — increasing `noise_std` scatters points around the
   theoretical front.

---

> ⚠️ Synthetic mode bypasses simulation entirely — it produces no CSV logs or
> time-series traces. It is meant for fast validation of the algorithmic
> pipeline, not for full simulation trace analysis.
