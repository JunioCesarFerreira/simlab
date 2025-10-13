## Synthetic Data Mode (master-node)

You can run the **master-node** component in *synthetic data mode*, so that it generates benchmark results internally instead of executing real Cooja simulations via SSH. This is especially useful for testing and validating the optimization pipeline without the overhead of real simulation.

### 1. Enable synthetic mode

The master-node checks the environment variable `ENABLE_DATA_SYNTHETIC` inside its worker loop:

```python
mode = bool(os.getenv("ENABLE_DATA_SYNTHETIC", "False"))
if mode:
    from lib.synthetic_data import run_synthetic_simulation
    run_synthetic_simulation(sim, mongo)
    continue
````

To activate synthetic mode, set:

```bash
export ENABLE_DATA_SYNTHETIC=true
```

Or, when using Docker / `docker-compose.yml`:

```yaml
services:
  master-node:
    environment:
      - ENABLE_DATA_SYNTHETIC=True
      # (additional synthetic settings below)
```

### 2. Configure benchmark and noise parameters

The module `synthetic_data/run_synthetic_simulation` uses environment variables to control how synthetic data is generated. Key variables:

| Variable    | Meaning / effect                              | Default   |
| ----------- | --------------------------------------------- | --------- |
| `BENCH`     | Benchmark function to use (DTLZ2, ZDT1, SCH1) | `"DTLZ2"` |
| `NOISE_STD` | Standard deviation of additive Gaussian noise | `"0.0"`   |

You can set these in your environment or via Docker:

```bash
export BENCH=ZDT1
export NOISE_STD=0.05
```

Or in `docker-compose.yml`:

```yaml
services:
  master-node:
    environment:
      - ENABLE_DATA_SYNTHETIC=True
      - BENCH=ZDTL2
      - NOISE_STD=0.1
      # … other master-node env vars …
```

### 3. How the synthetic mode flow differs

* **Default (real mode)**: the worker downloads simulation files via GridFS, sends them via SCP, starts Cooja via SSH, retrieves simulation logs, converts logs to CSV, computes objectives/metrics, and marks simulation done.

* **Synthetic mode**: when `ENABLE_DATA_SYNTHETIC=True`, the worker *skips* all file transfer, SSH, and simulation execution steps. Instead it directly calls `run_synthetic_simulation(sim, mongo)`, which:

  1. Marks the simulation as **running** in the database
  2. Computes synthetic objective values using benchmark functions (e.g. DTLZ2, ZDT1) and noise
  3. Marks the simulation as **done** with those objective values
  4. Checks whether all simulations in the generation are done and marks generation completion

### 4. Example `docker-compose.yml` snippet

Here is how you might set the synthetic mode in your Docker composition for master-node:

```yaml
services:
  master-node:
    environment:
      - ENABLE_DATA_SYNTHETIC=True
      - BENCH=DTLZ2
      - NOISE_STD=0.05
      - MONGO_URI=mongodb://mongo:27017/?replicaSet=rs0
      - DB_NAME=simlab
      - IS_DOCKER=True
      # ... other env vars ...
```

Then launch:

```bash
docker-compose up --build -d
```

### 5. Verify that synthetic mode is working

1. **Check logs**

   ```bash
   docker-compose logs master-node
   ```

   You should see log entries like:

   ```
   Starting benchmark simulation <sim_oid>
   ```

   and *no* logs related to SSH, Cooja execution or log transfer.

2. **Database inspection**
   In MongoDB, simulations should appear with state `done`, with objective values stored, even though no real simulation was run.

3. **Parameter variation test**
   Change `BENCH` and `NOISE_STD`, relaunch, and confirm output values change accordingly.

4. **Behavior of downstream modules**
   Verify that your MO-engine or REST API modules ingest the synthetic results exactly as they would ingest real simulation results.

---

> ⚠️ Note: Synthetic mode bypasses simulation entirely — it does not generate CSV logs or detailed time-series output. It is intended for faster testing / validation of the algorithmic pipeline rather than full simulation trace analysis.

