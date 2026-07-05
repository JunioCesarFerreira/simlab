# Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — Phase 1 of the implementation plan

### Added — synthetic benchmark instances (GUI + per-experiment config)

- **Synthetic Instances GUI editor** (`gui/simlab/src/components/synthetic-editor/`,
  route `/synthetic`, sidebar entry): visual definition of DTLZ2 / ZDT1 / SCH1
  benchmark instances (objectives `M`, decision variables `n`, noise σ, region Ω)
  with a live theoretical Pareto-front preview — 2D SVG for `M = 2` and an
  interactive **3D quarter-sphere** (echarts-gl) for DTLZ2 `M = 3`. A dedicated
  4-step launch wizard creates the experiment directly.
- **Per-experiment synthetic config**: `parameters.simulation.synthetic =
  { enabled, bench, noise_std }` (`SyntheticConfig` in `pylib`), which takes
  precedence over the `ENABLE_DATA_SYNTHETIC` / `BENCH` / `NOISE_STD` env vars.
  Also exposed as an optional toggle in the existing Launch Wizard (step 3).
- **REST endpoint** `GET /synthetic/benchmarks` (benchmark catalogue) and
  `is_synthetic` / `synthetic_bench` fields on the experiment list response,
  driving an amber *Synthetic* badge in the experiment card and detail views.
- **mo-engine**: when `synthetic.enabled = true`, all strategies (NSGA-II/III,
  Random Search, Batch) skip CSC/firmware generation and source-repository
  lookups (no Cooja artifacts are produced).

### Fixed — synthetic evaluation correctness

- Objectives are now derived from `parameters.objectives[].metric_name` (in
  order) instead of the empty `data_conversion_config.objectives`, so results
  are stored under the exact keys the mo-engine reads back (previously the
  objectives map was written empty).
- Region normalization now reads `parameters.problem.region` (was reading a
  non-existent top-level `parameters.region`, always falling back to the default
  Ω and ignoring custom regions).
- The fixed **sink** is excluded from the genome, keeping the decision-variable
  count consistent with `n_relays` (genome length = `2 · n_relays`).
- SCH1 now maps its decision variable back to the region's x-axis scale
  consistently with DTLZ2/ZDT1; ZDT1 guards against `sqrt` of a negative value.

### Added — alignment with the SimLab paper (article reproducibility)

- **Random Search strategy** (`mo-engine/lib/strategy/random_search.py`):
  baseline algorithm-agnostic generator referenced in §3.4 / Table 2 of the
  paper, registered in `engine.py` under `parameters.strategy = "random_search"`.
  Reuses the same genome cache and penalty-objectives mechanisms as NSGA-III.
- **Example experiment payload** for the new strategy:
  `debug/requests/post-random-search-experiment-p2.json`.
- **Script-based baseline orchestrator** (`baseline/`): minimal shell + Docker
  Compose pipeline that mirrors what §5.6 / Table 6 of the paper compares
  against. Includes `run_baseline.sh` (SSH/SCP orchestration without state
  machine, persistence, or resume), `docker-compose.baseline.yaml`
  (10 standalone Cooja workers), and `prepare_workload.py` (generates the
  30 × 50-mote workload from §5.2 using the same Cooja template the mo-engine
  uses).
- **External NSGA-III benchmark** (`experiments/external-nsga3-benchmark/`):
  reproduction infrastructure for Table 3 of the paper, comparing the native
  SimLab NSGA-III against DEAP and pymoo implementations on DTLZ2. Scripts
  share a common DTLZ2 definition and HV / GD / IGD / Coverage metrics. DEAP
  and pymoo remain **off-path dependencies** (in `experiments/.../requirements.txt`)
  and are not pulled into the production Docker image.

---

## [v1.0.0] – Initial Release (2025-10-12)

### Overview
This is the **first public and functional release** of the **SimLab** project — a distributed framework for managing and executing large-scale multi-objective simulations using Dockerized environments and a REST API interface.

The current version provides a **fully operational system** capable of running and monitoring experiments end-to-end through container orchestration and MongoDB integration.

### Features
- ✅ **Functional base architecture**
  - Master-node orchestrator for simulation execution (via SSH/SCP)
  - MO-engine (multi-objective optimization loop)
  - REST API for experiment management
  - MongoDB integration for experiment and generation tracking
- 🐳 **Dockerized environment**
  - Ready-to-run Docker Compose configurations for local and distributed setups  
  - Debug environments under `debug/` for simple testing or small experiments
- ⚙️ **Synthetic data mode**
  - Built-in synthetic benchmark evaluation (`DTLZ2`, `ZDT1`, `SCH1`) for validation and algorithm testing without running Cooja simulations
- 📡 **Asynchronous orchestration**
  - Multi-threaded simulation queue, automatic enqueue of waiting experiments
- 📁 **GridFS-based file management**
  - Storage and retrieval of simulation inputs, outputs, logs, and CSV results

### Documentation and Improvements (Planned)
This initial release is functional but not yet fully documented.  
The following enhancements are planned for upcoming versions:

- Complete documentation of setup and deployment workflows
- Additional testing and CI automation  
- Extended examples of experiment submission and monitoring  
- Benchmark dataset publication and performance validation  
- Development of a graphical user interface (GUI) in Vue.js to simplify experiment configuration, execution monitoring, and result visualization  
- English and Portuguese documentation parity  
- More algorithms and analyzers

### Notes
This version establishes the operational baseline of SimLab.  
Subsequent versions will focus on documentation, reproducibility, and academic publication preparation.

---

**Author:**  
Junio Cesar Ferreira<br/>
Institute of Mathematical and Computer Sciences (ICMC), University of São Paulo (USP)
