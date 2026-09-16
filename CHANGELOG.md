# Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased] — NSGA metrics: review corrections

### Fixed

- Restore benchmark axis order before analytical GD, so a ZDT1 request with
  permuted objectives has the same distance as the canonical request.
- Resolve missing survivor sets per generation. `population_sources` reports
  the set used for each generation; `population_source="mixed"` identifies a
  heterogeneous series. Charts label the fallback and each compared run.
- Serialize DEAP's global NumPy RNG context across experiments, including
  restoration on exceptions.
- Persist pymoo NSGA-III's hyperplane normalization in a versioned BSON-safe
  `selection_state` checkpoint. Legacy records remain readable with a warning
  when exact continuation cannot be guaranteed; malformed snapshots fail
  explicitly.

### Tests

- Regressions cover interior/on-front and off-front ZDT1 permutations, partial
  survivor histories with and without cumulative HV, empty survivor sets,
  concurrent RNG contexts and exceptions, and exact resume trajectories for
  all six backends across 72 seed/checkpoint/status combinations.
- Frontend tests cover mixed-population labels and legacy API responses.
- Validation and limitations are documented in
  [NSGA regression validation](mo-engine/tests/regression/README.md#validation-of-integration-fixes).

---

## [Unreleased] — NSGA metrics: endpoint robustness and launch defaults

### Fixed

- **`/hv-gd` read objectives by position while building the reference by name.**
  Individuals store their objectives as a positional list in the order the
  experiment declared them, so any request that reordered or subsetted the axes
  compared one objective against another's reference — asking for `[f2, f1]`
  turned a GD of 0 into 11.31. Requested names are now resolved against
  `parameters.objectives[].metric_name`, and an unknown name returns 422.
- **A subset request used the analytical front of a smaller benchmark.** DTLZ2
  with M=3 read on two axes is not DTLZ2 M=2; a subset now falls back to the
  empirical reference. A *permutation* keeps the analytical front, reordered to
  match — ZDT1 is not symmetric in its objectives.
- **All individuals penalised raised a 500.** `max()` over the empty set; it now
  returns the empty response shape.
- **A synthetic run with no stored `pareto_front` returned empty series** even
  though its analytical front needed none. The early return moved into the
  empirical branch, the only one that needs a stored front.
- **The launch wizard's mutation defaults left the search crossover-driven.**
  `prob_mt = 0.1` and `per_gene_prob = 0.05` compose: over 10 variables that is
  one variable mutated every twenty children. The defaults are now the textbook
  convention — every child mutated, each variable with probability `1/n`.
- **A fixed `divisions = 10` fitted no objective count.** It gives 11 reference
  directions in M=2 — a population of 50 competing for a tenth of the niches it
  could use — and 3003 in M=6. It is now derived from M and the population.

### Added

- `suggestedDivisions` and `expectedMutatedVariables` in `gui/src/lib/nsga3.ts`.
  The wizard shows the expected number of mutated variables per child, warning
  below 0.1, and both launch wizards now warn when the reference lattice is far
  *below* the population — the mirror of the existing H > population warning.
- `docs/markdown/SYNTHETIC_MODE.md`: a protocol-conventions table (HV reference,
  measured population, generation 0, evaluation budget, mutation composition,
  divisions) and a note that SimLab's default SCH1 domain is `[-5, 5]` against
  the `[-10, 10]` common in the literature — a different problem, not a harder
  or easier one in any simple sense.

### Notes

- The mutation default was chosen on measurement, not convention. Over 15 seeds
  and 40 generations the textbook rate is neutral on DTLZ2 (M=3) and large on
  ZDT1: HV 0.619 → 0.802 for NSGA-II and 0.633 → 0.806 for NSGA-III, with the
  seed-to-seed spread cut by a factor of three. **This corrects the phase-2
  reading**, which at 5 seeds and 20 generations suggested the opposite; that
  difference was noise.
- The two mutation probabilities are not redundant at equal products.
  Concentrating mutations in few children is a different search from spreading
  them thinly across all of them.
- The regression baseline is re-frozen at `phase-6`, its defaults tracking what
  the wizard sends. `nsga3-sch1` rises from 16.110 to 16.522 and its worst
  per-step HV drop among survivors falls from 0.229 to 0.041 — the case phase 0
  opened as the one where survivors also dropped sharply is now closed. It was
  the divisions.
- The audit reported `rest-api/tests/test_experiment.py` hanging on
  `TestClient`/AnyIO startup. That did not reproduce here; the suite runs.

---

## [Unreleased] — NSGA metrics: exact generational distance

### Fixed

- **GD reported discretisation error as lack of convergence.** Measured against
  a *sampled* reference front, GD cannot go below that sample's fill distance.
  Points lying exactly on the DTLZ2 front — true GD zero — scored 0.0016 at
  M=2, 0.030 at M=3 and 0.194 at M=6. Raising the sample to 200 000 points
  still left 0.051 at M=6, because the front is an (M−1)-dimensional manifold
  and fill distance shrinks only as `N^(-1/(M-1))`.
  GD now uses the **closed-form distance to the true front**
  (`pylib.benchmarks.front_distance`): the same points score 5·10⁻¹⁷ at every M.
  For DTLZ2 the front is the unit sphere, so the nearest point to any
  positive-orthant `f` is `f/‖f‖`; ZDT1 and SCH1 are plane curves solved to
  machine precision.
- **`compute_hv_gd.py` used a different hypervolume reference from everything
  else.** With `--true-front-bench` it still derived the reference point from
  the observed worst value, while `/hv-gd` and `plot_pareto_results.py` used the
  benchmark's fixed `1.1 × nadir` — so the same experiment reported one HV on
  the CLI and another in the GUI, and no two runs were comparable. It now uses
  the fixed nadir like the others.
- **Indicators were normalised by the sampled reference's extremes**, which fall
  short of the real corners and depend on how the sample was drawn. When the
  benchmark has a known range, normalisation now uses the theoretical
  ideal-nadir pair.

### Added

- `pylib.benchmarks.front_distance` and `pylib.benchmarks.ideal`;
  `moo_metrics.gd_analytical` and an explicit `bounds` argument on `gd`, `igd`
  and `igd_plus`. Mirrored into `pareto-analysis/lib/`, with the parity tests
  extended to the new surface — the CLIs must keep running with no `pylib` on
  the path, so the mirror stays a mirror.
- `/hv-gd` returns `gd_method` (`analytical` / `reference_front`), `gd_formula`
  and `normalization`; the chart caption says when GD escaped the reference
  front's discretisation error.
- `docs/markdown/SYNTHETIC_MODE.md` §7 documents the definitions, the measured
  discretisation table, and why IGD cannot take the same route.

### Notes

- **IGD and IGD+ keep the sampled reference and keep the floor.** They average
  over the reference set itself — that is what makes them measure coverage — so
  there is nothing to substitute. Read them as comparative numbers between runs
  sharing a reference, not as absolute distances.
- The regression baseline is re-frozen at `phase-4`. Hypervolume is unchanged;
  only GD moved, and the size of the move is the discretisation error that was
  being misread. It is 25× on `nsga2-sch1`, whose population gets close enough
  to the front that nearly all the reported GD came from the reference, and
  negligible on ZDT1, whose runs stop far enough away that it never mattered.
  The separate `radial_error` column added in phase 0 is gone: phase 4 promoted
  exactly that quantity to GD, so it became a duplicate.
- The plan's original exit criterion for this phase — "GD below 1e-3 for M=2, 3
  and 6, via a denser sample" — was unreachable and has been corrected in place.
  A Das–Dennis grid projected onto the sphere was also tried and rejected: at
  M=6 with 252 points it is *worse* than the random sample (0.272 vs 0.194),
  because the radial projection is not uniform on the sphere.
- `pareto-analysis` tests needed `requests`, missing from the local venv (CI
  already installed it). 76 tests now run where 24 did.

---

## [Unreleased] — NSGA metrics: determinism and resume

### Fixed

- **The DEAP and pymoo NSGA-III backends ignored the experiment seed.** Five
  identical calls produced five different selections. DEAP shuffles through the
  process-wide `numpy.random` module; pymoo builds
  `np.random.default_rng(None)` — OS entropy — whenever no `random_state` is
  passed, which `np.random.seed` cannot reach. Both are now driven from the
  experiment's `algorithm.random_seed`. The DEAP call runs inside a context that
  seeds the global numpy stream and **restores** it afterwards, so the engine
  does not disturb anything else sharing its interpreter.
  *Correction to the audit report: only the two NSGA-III adapters were affected.
  Rank-and-crowding survival has no random step, so the NSGA-II adapters were
  already deterministic; they are seeded anyway, against future library changes.*
- **Resuming a checkpoint restarted from the wrong population.**
  `_restore_population_state` loaded the previous generation's *offspring* as
  parents. Those documents are Q(t-1), not the survivors P(t-1), so every parent
  environmental selection had kept from an older generation was dropped: 4 to 7
  of 12 in the test scenario. Parents now come from that generation's
  `survivors`.
- **A resumed run drew from a fresh random stream.** Each generation document
  now carries an `rng_state` snapshot, taken at enqueue time — before any draw
  belonging to that generation. Because every library seed derives from the same
  generator, that one snapshot is the whole random state of a run.
- **Population order was lost on resume.** Individuals came back sorted by
  chromosome hash while the live population is in generation order, and the
  mating tournament draws by index — so a reordered population is a different
  search. Individuals now record their `index` within the generation, and the
  restore sorts by it.

### Added

- `mo-engine/lib/strategy/library_rng.py` — seed derivation, the numpy global
  seeding context, and BSON-safe snapshot/restore of the engine generator.
- `Generation.rng_state` and `Individual.index`.
- `IndividualRepository.find_by_experiment_and_ids` — survivor hashes cannot be
  resolved against one generation's documents.
- `mo-engine/tests/test_resume_reproducibility.py` — 4 continuous generations
  against 2 + resume + 2 must produce identical populations throughout, driving
  the real `_generation_enqueue` and `_evolution`.

### Notes

- Experiments started before this change still resume: missing `rng_state` or
  `survivors` logs a warning and falls back to the previous behaviour. A
  survivor with no matching individual document aborts the resume rather than
  continuing on a truncated population.
- `Generation.survivors` is now stored verbatim — in selection order, repeats
  included — because a resume rebuilds the population from it. De-duplicating
  would be harmless for the metrics but would shrink the restored population.

---

## [Unreleased] — NSGA metrics: genetic operators

### Fixed

- **Bounded SBX gave both children the same spread factor**, computed from the
  distance to the *lower* bound. The two children are pushed in opposite
  directions, so each needs the distance to the bound it is heading towards.
  Each child now derives its own factor from the same draw, matching DEAP's
  `cxSimulatedBinaryBounded` exactly. This affects every strategy, the `_deap`
  and `_pymoo` variants included — they replace only the survival step.
- **Native NSGA-III niching was not the canonical algorithm.** Four defects,
  all corrected: normalisation now uses ideal point, extreme points and
  hyperplane intercepts over `St = accepted ∪ truncated front` instead of the
  front's own min/range; association is by perpendicular distance to the
  reference *direction* rather than Euclidean distance to the reference
  *point*; niche occupancy is seeded from the already-accepted individuals
  instead of starting at zero; and niches with no remaining candidate leave the
  minimum-occupancy race, which removes a fallback that picked uniformly at
  random from everything left without even updating the occupancy.
  `associate_to_niches` was dead code in this module and is now the association
  the selection uses.
- **The NSGA-II mating tournament broke rank ties at random**, dropping the
  crowded-comparison operator. Ties now go to the larger crowding distance.
  NSGA-III deliberately keeps the random tie-break: Deb & Jain enforce diversity
  through reference-point niching, not through a crowded tournament.

### Notes

- **DTLZ2 results get worse with the corrected operators, and the cause is
  understood.** The old SBX clipped ~0.54% of children onto the *upper* bound
  (0% onto the lower one — the defect was asymmetric). DTLZ2's position
  variables must reach 0 or 1 to produce the front's corner solutions, which
  dominate hypervolume, so the defect manufactured half of those corners for
  free. The regression persists at 60 and 150 generations and under textbook
  mutation, so it is not a short-budget artifact. The fix stands — the old
  distribution is provably wrong — but **DTLZ2 numbers produced before this
  change depend on that artifact and need recomputing.**
- The crowded mating tournament's mean effect is within the 5-seed noise; what
  it measurably does is raise run-to-run variance on DTLZ2 (M=3), consistent
  with crowding distance being a weak diversity signal beyond two objectives.
- Intercepts follow **pymoo**, not DEAP. DEAP solves the hyperplane in the
  translated space and then divides by `intercepts - ideal`, subtracting the
  ideal point a second time; pymoo returns `ideal + 1/x`, which is what Deb &
  Jain's Eq. (4) describes. A test pins the choice.
- The selected population carries ~4% duplicate slots — a child that reproduces
  a surviving parent exactly enters the union twice. Pre-existing in both
  algorithms and untouched here.

### Added

- `mo-engine/tests/test_sbx_bounded.py`, `test_niching_selection.py` and
  `test_tournament_crowding.py` — parity against DEAP and pymoo, plus one test
  per audit defect.
- `compute_crowding_distances` in `lib.genetic_operators.selection`.
- The regression baseline is re-frozen at stage `phase-2`
  (`mo-engine/tests/regression/baseline.json`, renamed from
  `baseline_pre_fix.json`). Its
  [README](mo-engine/tests/regression/README.md) carries the per-operator
  ablation, the bound-clipping measurements and the pre-fix table.

---

## [Unreleased] — NSGA metrics: the measured population

Corrections derived from the [NSGA metrics audit](experiments/nsga-metrics-audit/README.md).

### Fixed

- **Quality indicators measured the wrong set.** HV, GD, IGD and IGD+ were
  computed over the *offspring* `Q_t` — the children evaluated in a generation —
  because those are the individuals a generation document stores. Environmental
  selection can keep an excellent parent that never reappears among the
  children, so the curves showed a regression while the search was in fact
  holding that parent. `mo-engine` now persists the surviving population `P_t`
  on each generation and `GET /experiments/{id}/hv-gd` measures it by default.
- **The last generation never got an environmental selection.** The stop
  condition in `_evolution` was tested *before* the selection, so the reported
  Pareto front was `ND(P_{t-1} ∪ Q_t)` — a union of up to `2·pop_size`
  candidates no selection had run on — instead of `ND(P_final)` of `pop_size`
  individuals. The selection now runs first for both NSGA-II and NSGA-III. The
  evaluation budget is unchanged: neither path enqueues another generation.
- **"Cumulative" only ever applied to HV.** GD and IGD stayed on the offspring
  whichever view was selected, so the two halves of the panel described
  different sets. All indicators now follow one selector.

### Added

- **`GET /experiments/{id}/hv-gd?population=`** — `survivors` (default),
  `offspring` or `archive`. The response reports `population_source`: runs
  recorded before survivor sets existed degrade to `offspring` and say so,
  rather than returning empty series. `hv_cumulative` is unchanged and stays
  independent of the selection.
- **`Generation.survivors`** — the chromosome hashes environmental selection
  kept, written by `GenerationRepository.set_survivors`. Stored on the
  generation rather than as a flag per individual: one write per generation, no
  change to the unique `(generation_id, individual_id)` index, and a survivor
  carried over from an older generation has no document in the current one to
  flag. Writes are best-effort — survivors are analysis metadata and a failed
  write must not abort a running experiment.
- **Web GUI** — the HV-only *Per generation / Cumulative* toggle is replaced, on
  both the experiment detail chart and the comparison page, by a **Measured
  set** selector (*Survivors / Offspring / Archive*) that drives HV, GD and IGD
  together. The caption states which set was measured and warns when a run fell
  back to the offspring.
- **`mo-engine/tests/regression/`** — a frozen pre-correction baseline (30 runs:
  NSGA-II and NSGA-III over DTLZ2, ZDT1 and SCH1, seeds 1/2/3/5/7) with a
  synchronous kernel harness that needs no MongoDB, so each later phase of the
  fix plan produces an inspectable, attributable diff. See its
  [README](mo-engine/tests/regression/README.md).

### Notes

- Reading survivors requires resolving chromosome hashes across the whole
  experiment, not within one generation: a survivor may have been evaluated
  several generations earlier.
- Experiments finished before this change keep their stored `pareto_front`,
  which was built from the unselected union. Re-running is the only way to get
  the `ND(P_final)` front for them.

---

## [Unreleased] — Coverage level α as an explicit problem parameter

### Added

- **Web GUI — problem editor**: new **Coverage Level α (%)** field for Problems
  P1 and P2, next to the communication radii. It edits the problem's
  `min_coverage_percentage` — the minimum share of the sampled mobile-node
  trajectory that must stay connected to the sink — and is validated to the
  closed range `[0, 100]`. The field is hidden for P3/P4, whose adapters have
  no trajectory coverage constraint.
- **Web GUI — launch wizard**: step 1 (*Problem*) summarizes the value as
  `Coverage level α  ≥ N%`, so the level in force is visible before an
  experiment is created; the value is exported in `parameters.problem` for P1/P2
  only.
- **Web GUI — visualization**: the problem topology modal on the experiment
  detail page shows an `α ≥ N%` pill, and the *Problem* parameter table lists
  `min_coverage_percentage` like any other problem field. Experiments launched
  before this change simply have no pill.
- **`pylib/config/problems`**: `parse_min_coverage_percentage()` centralizes the
  parsing for `ProblemP1`/`ProblemP2` and now **rejects** non-numeric values and
  values outside `[0, 100]` at the cast boundary, so a mis-configured experiment
  fails before any simulation is spent instead of silently penalizing the whole
  population. The field stays optional and still defaults to `100.0`.
- **mo-engine**: the P1 and P2 adapters log the α in force when the coverage
  structures are built, so a run's log records the level it was executed with.

### Notes

- α remains a **problem** parameter, not a GA knob: it defines what *feasible*
  means, while `apply_coverage_repair` / `repair_coverage_budget` control how
  hard the operators try to reach it. The serialized key is unchanged
  (`min_coverage_percentage`), so existing saved problems, stored experiments
  and request payloads keep working — α is presentation and validation on top
  of the field the engine has always read.

---

## [Unreleased] — Inverted Generational Distance (IGD / IGD+)

### Added

- **`pylib/moo_metrics`**: new canonical module for the convergence quality
  indicators — GD, IGD and IGD+ (Ishibuchi et al., 2015) — plus reference-front
  sanitation and ideal-nadir normalisation. Single source of truth shared by the
  REST API; `pareto-analysis/lib/metrics.py` mirrors it for the standalone CLIs
  (which must run with no `pylib` on the path) under a new parity test.
- **REST API**: `GET /experiments/{id}/hv-gd` now returns `igd_plus` alongside
  `igd`, plus `reference_size` and `normalized`, and accepts `?normalize=`.
- **Web GUI**: the experiment detail page shows **IGD and IGD+ side by side with
  GD** in a third chart panel, each with its own PNG export. A caption names the
  reference front and warns when it is the run's own final front, in which case
  GD and IGD are self-referential — progress towards this run's own result
  rather than convergence to the true optimum, and not comparable across runs.
- **`pareto-analysis`**: both CLIs report IGD/IGD+; the uploaded figure gained a
  third panel; `compute_hv_gd.py` now reuses `compute_convergence_metrics`
  instead of its own duplicate loop (and so gains the cumulative-HV curve).
  New `--raw-distances` flag on both.

### Fixed

- **IGD was unusable on non-synthetic experiments**: the stored Pareto front was
  used as the GD/IGD reference with no filtering, so a penalised individual
  (the engine writes 1e9+ for infeasible solutions) contaminated it. IGD
  averages *over* the reference, so one such row dragged the indicator to ~1e9.
  GD hid the same contamination because it minimises over the reference
  instead. Penalised, duplicate and dominated rows are now dropped, and a
  reference row missing an objective is skipped rather than defaulted to `0.0`
  (which read as optimal on a minimised axis and pulled the whole front towards
  the origin).
- **`docs/markdown/SYNTHETIC_MODE.md`** claimed GD was the RMS variant
  `sqrt((1/N)·Σ dᵢ²)`; the code has always used the `p = 1` arithmetic mean.
  The doc now matches the code, and a test locks the definition down.

### Changed

- **GD values move**: distances are now normalised by the reference front's
  ideal-nadir range by default, so objectives of different magnitudes weigh
  comparably. Pass `?normalize=false` (API) or `--raw-distances` (CLIs) for the
  previous raw-unit values. Hypervolume is unaffected — it keeps its own
  reference point in raw units.
- `compute_convergence_metrics` returns a named `ConvergenceMetrics` tuple
  (still positionally unpackable) rather than a bare 4-tuple.

---

## [Unreleased] — Runtime (computational) telemetry per experiment

### Added — runtime metrics collection, persistence and visualization

- **`pylib/telemetry`**: new module that, when an experiment finishes
  (Done/Error), queries Prometheus/cAdvisor over the exact
  `[start_time, end_time]` window, normalizes every time series
  (timestamp, metric, value, unit, scope, labels) and preserves them
  integrally as an **immutable GridFS artifact** — Parquet/snappy preferred,
  CSV.gz fallback when pyarrow is unavailable. The experiment document gains
  only a small `runtime_metrics` block: collection status, artifact reference
  (`file_id`, sha256, size, schema version) and a summary (duration, CPU
  average/peak %, memory average/peak bytes) computed over all aggregate
  samples of the window. Collection is idempotent (atomic claim) and new
  metric summaries can be added without breaking changes.
- **mo-engine**: starts a runtime-metrics watcher (Change Stream on finished
  experiments + startup backfill sweep bounded by `TELEMETRY_BACKFILL_HOURS`).
  Tunables: `PROMETHEUS_URL`, `TELEMETRY_ENABLED`, `TELEMETRY_QUERY_STEP`,
  `TELEMETRY_COLLECTION_DELAY_SECONDS`, `TELEMETRY_CONTAINER_FILTER`.
- **REST API**: `GET /experiments/{id}` now embeds the `runtime_metrics`
  summary; new `GET /experiments/{id}/runtime-metrics?max_points=N`
  reconstructs the full series from the GridFS artifact on demand, with
  bucket-average downsampling.
- **GUI**: new **Runtime Metrics** section on the experiment detail page —
  summary tiles (duration, CPU avg/peak, memory avg/peak) shown immediately,
  full CPU/memory time-series charts (aggregate + per-container) loaded only
  when the user clicks *Show charts*.
- **docker-compose**: `moengine` joins `monitoring-net` and receives
  `PROMETHEUS_URL` so it can reach Prometheus.
- Experiment cascade delete now also removes the telemetry artifact from
  GridFS.

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
