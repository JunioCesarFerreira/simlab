# Campaign v2 — the experiments behind the thesis and the papers

Status: **plan, nothing launched** (2026-11-18). Files and settings it names that
come from this plan or from [PROVENANCE_GUIDE.md](PROVENANCE_GUIDE.md) do not exist
yet (`experiments/campaign-v2/`, `scripts/build.sh`, `SIMLAB_REQUIRE_CLEAN_BUILD`,
`provenance.*`, `runtime.*`), and neither does the `seed_study` strategy on `main`.
C0 and CC create their [`batch`](../../mo-engine/lib/strategy/README.md#batch)
experiments through the API; the GUI does not offer that strategy.

**Decision:** every result obtained with firmware v1 (`rpl-udp-csma`, `rpl-udp-tsch`)
is discarded. In v1, `latency` averaged the zeros of lost pings, `energy` was mA·s
labelled mJ, and `throughput` counted the probe replies. Together these made the
energy × throughput conflict of the paper-1 fronts partly a counting artefact (corr
0.50–0.58 in P1, 0.77 in P2). See [FIRMWARE_METRICS.md](FIRMWARE_METRICS.md) §7. The v1 experiments stay in
the database for audit and are listed as superseded (§8). No v1 number is used in any
text.

This document fixes, **before** anything runs:

- what is run;
- in which order;
- with which parameters;
- how it is analysed.

Changing any of these after looking at results has to be reported as such.

---

## 1. Overview

| Code | Purpose | Feeds | Strategy | Size (sims) | Pool time* |
|---|---|---|---|---|---|
| **C0** | Acceptance: build, provenance, determinism, pool rate | all | batch, nsga3 (smoke) | ~400 | ~1 h + checks |
| **CA** | Seed study: how many seeds per individual (`k`) | paper 3, sets `k` for CB/CC | seed_study | 4,000 | ~10 h |
| **CB** | NSGA-III vs random search, P1 vs P2 | paper 1, thesis | nsga3, random_search | 38,600–64,400 | 4–7 days |
| **CC** | CSMA vs TSCH on the fronts of CB | thesis (paper 1 optional) | batch | ~600 TSCH | ~15 h |
| **CD** | USL scalability of the pool, v2 workload | paper 2 | usl_benchmark | ~300 per machine | ~1 day per machine, pool idle |
| **CE** | NSGA-III implementation check vs DEAP/pymoo | paper 1 (Table 3) | synthetic (no Cooja) | — | minutes |

\*At about 400 CSMA-v2 simulations per hour. The basis is the measured pool rate (~320
sims/h of CSMA v1 at 180 min, local + 4 VMs, 2026-11-07) scaled by the v2/v1 cost per
simulation measured in the harness (85 s vs 112 s). C0 measures the real rate. One TSCH-v2
simulation costs about 10 CSMA-v2 simulations. These costs predate the 64-entry
tables of 2026-11-21, with which a TSCH-v2 run cost 5–7 CSMA-v2 runs in the
replays; C0 redoes the table.

Order: **C0 → CA → (choose `k`) → CB → CC**. CE can run at any time. CD runs in a window
with nothing else on the pool.

---

## 2. Rules for every campaign

1. **One frozen release.** All campaigns run on one tagged commit, built and pinned as in
   `PROVENANCE_GUIDE.md`. That release includes:
   - the validity guards of 2026-11-18 (NaN, invalid fronts, MAC gene, evidence-based filter);
   - the seed-study feature, which today exists only on `feature/dist-seed` (23 commits
     ahead of `main`) and must be merged first.

   A code change during a campaign means a new tag. Runs on the old tag are either
   repeated or reported apart.
2. **Provenance recorded and verified.** `SIMLAB_REQUIRE_CLEAN_BUILD=true` on the engine.
   Every experiment carries `provenance.engine.git_sha` equal to the tag, and
   `firmware_snapshot.status = captured`. One Cooja image ID per campaign
   (`PROVENANCE_GUIDE.md` §7).
3. **Firmware v2 only.** Use `rpl-udp-csma-v2`, plus `rpl-udp-tsch-v2` in CC only. Upload
   from `firmware/` with `python upload-firmware.py rpl-udp-csma-v2 rpl-udp-tsch-v2`. The default v2
   data conversion is used unchanged (16 metrics).
4. **Common random numbers.** One list of `k` simulation seeds is drawn once (§3, D2) and
   used by every individual of every run in CB and CC. Differences between algorithms are
   then not differences of seeds.
5. **Replica policy.** The default applies to CB/CC: repeat a failed replica once, then
   give it the worst objectives. CA keeps `incomplete_objective: accept`, as the seed study
   requires. Every refused, repeated or timed-out evaluation is counted and reported (§7).
6. **Timeout.** `SIM_TIMEOUT_SEC = 10800`, the compose default. A TSCH-v2 run of 180 min
   took up to ~3,600 s under load before the 64-entry tables of 2026-11-21, and
   816–1,167 s with them ([firmware/README.md](../../firmware/README.md#simulation-speed)). After each campaign, the number of timeouts must be 0,
   or the affected individuals are re-run before analysis.
7. **Pool discipline.**
   - Nothing else runs on the pool during CD.
   - During CB, all cells run interleaved. Running cell after cell would put a host
     change or a slow night on one algorithm only.
8. **Naming and manifest.**
   - Experiment name: `v2-<campaign>-<problem>-<strategy>-r<NN>`, e.g. `v2-CB-P1-nsga3-r07`.
   - Every launched id goes into `experiments/campaign-v2/manifest.csv` when it is
     launched, not afterwards (§8).

---

## 3. Decisions before launch

Each decision is marked **[needs your choice]** or **[informative]**. The recommendation
is the default if nothing is said.

**D1 — Objectives [needs your choice].**

- **Recommended:** the same set for P1 and P2, so the two formulations can be compared:
  - `latency` (median one-way delay, min);
  - `energy` (total network energy, min);
  - `delivery_ratio` (max).
- **Why:**
  - Under CSMA, `throughput` and `energy` both grow with the number of nodes. P2 varies
    the relay count (12–30): more relays generate more reports and more idle-listening
    energy. That is structure, not a trade-off between routes.
  - `delivery_ratio` does not grow with node count.
  - Under CSMA, `energy` is 91.5% idle listening. So in P1 (fixed node count) it is
    nearly constant (CV 5–6% in v1), and in P2 it essentially measures the relay count.
    That is a real cost, but it has to be said in that form.
- **Not optimised but reported for every individual:** `throughput`, `power_mw`,
  `node_coverage`, `latency_p95`, `hops`.
- **Alternative:** keep the paper-1 set {latency, energy, throughput} for continuity,
  and declare the node-count effect.

**D2 — Seeds per individual `k` [informative; set by CA].**

- `k = max` over the chosen objectives of `m*`, the seeds needed for ranking reliability
  Eρ² ≥ 0.90. Round it up, with a minimum of 3.
- In v1 that gave 6–8. Paper-1 runs used 2.
- The seed list is drawn once with `random.Random(20261118).sample(range(1, 2**31), k)` and
  frozen in the manifest.

**D3 — P1 crossover [needs your choice].**

- **The defect:** `sbx_with_radial_translate` pairs the relays of the two parents by
  list index, whose order is arbitrary. It applies SBX to x and y in separate calls, each
  with its own random child swap. So half of the child relays combine the x of one relay
  with the y of an unrelated one. No radial translation happens.
- **Recommended:**
  - Before CB, pair the relays by nearest neighbour (Hungarian assignment on distance).
  - Apply SBX to each pair with a single swap decision.
  - Rename the operator.
  - Run a small ablation (P1, NSGA-III, old vs new operator, 5 runs each, inside CB's
    budget), so the effect is reported, not assumed.
- **Alternative:** keep the operator and declare it as a limitation.
- **Note:** "NSGA-III ≈ random" in paper 1 may come partly from this defect.

**D4 — GA parameters [needs your choice: confirm].** Recommended standard settings
(Deb & Jain 2014; Deb & Agrawal 1995), the same for every cell:

| Parameter | Value | Note |
|---|---|---|
| `population_size` | 32 | smallest multiple of 4 above H = 28 reference points (3 objectives, `divisions = 6`) |
| `divisions` | 6 | |
| `number_of_generations` | 10 | G0…G10, 11 populations, 352 slots per run |
| `prob_cx`, `eta_cx` | 0.9, 20 | |
| `prob_mt` | 1.0 | every child goes through mutation… |
| `per_gene_prob` | 1/n | …each gene with probability 1/n: P1 n = 40 coordinates → 0.025; P2 n = 32 candidates → 0.031 |
| `eta_mt` | 20 | |

- Paper 1 used `prob_mt 0.15 × per_gene_prob 0.1`: a 1.5% chance per coordinate, applied
  to 15% of the children only.
- Random search uses `population_size` and `number_of_generations` only.

**D5 — Runs per cell `R` [needs your choice].**

- 10 is the minimum for a non-parametric comparison with an effect size; 20 is
  comfortable. Paper 1 had 2.
- Cost of CB, with distinct evaluations ≈ 352 per random-search run and ≈ 292 per
  NSGA-III run (~17% of children are parent clones):

| `R` | `k` = 3 | `k` = 5 | `k` = 8 |
|---|---|---|---|
| 10 | 38,600 sims, ~4 days | 64,400, ~6.7 days | 103,000, ~10.7 days |
| 20 | 77,300, ~8 days | 128,800, ~13.4 days | — |

- Upper bounds: infeasible P2 individuals are not simulated.
- Recommended: `R = 10`, with `k` from CA; raise to 20 if C0 shows a faster pool.

**D6 — Duration [informative].**

- Keep 180 min simulated: the scenario is defined by it, and the totals scale with it.
- The truncation check of 2026-11-15 (12 simulations) is too small to justify a shorter
  window.

**D7 — Mobility model [needs your choice].**

- Cooja's Mobility plugin repeats every trajectory with the period of the **longest** one
  (`WRAP_MOVES`). In the shared scenario that is ~111 s, so the 43-s mobile node stands
  still ~70 s per cycle and moves slower than its nominal 5 m/s on average.
  `is_closed` is ignored.
- Recommended: declare it as a limitation (it applies equally to every cell).
- Alternative: write one trajectory per node padded to the common period before C0.

---

## 4. The campaigns

### C0 — Acceptance (before anything counts)

1. **Release.**
   - Build from the tag with `scripts/build.sh`.
   - Check the provenance query of `PROVENANCE_GUIDE.md` §7 on a throw-away experiment.
   - Check that every host reports the same Cooja image ID.
2. **Firmware.**
   - Upload the v2 sources.
   - The `firmware_snapshot` sha256 must equal the files in `firmware/rpl-udp-csma-v2`
     (the test `pylib/tests/test_firmware_v2.py` pins the v1 hashes; check v2 by hand).
3. **Determinism.**
   - Batch: 3 chromosomes × 2 seeds, launched twice.
   - The 16 metrics must be identical between the two launches.
4. **Smoke runs.**
   - `nsga3`, P1 and P2: `population_size 8`, 2 generations, 2 seeds.
   - Check every experiment has:
     - `pareto_front_validity.valid == items`;
     - no individual with `completeness.refused`;
     - every stored `mac_protocol` equal to 0 (CSMA);
     - every metric finite;
     - zero timeouts.
5. **Pool rate.**
   - From the smoke runs and a 200-simulation batch, measure sims/h.
   - Redo the cost table of D5.
6. **Optional, informative:** replay 30 individuals of the paper-1 runs under v2 and
   report Kendall τ per objective. It tells the thesis how much of the old ranking
   survives. It is not used for any conclusion.

### CA — Seed study (paper 3; sets `k`)

- **Strategy:** `seed_study` with random panels, drawn by the strategy from each
  problem's decision space, on the shared scenario
  (`region [-150,150]²`, sink (0,0), radii 60/120, 180 min).
- **Panels:**
  - P1: 30 individuals × 100 seeds = 3,000 sims;
  - P2: 20 individuals × 50 seeds = 1,000 sims (paper 3 had 30 seeds; 50 tightens the
    interval on `m*`).
- **Firmware:** CSMA v2 only, one firmware per study. Mixing protocols in one panel
  underestimates `m*`: in v1 the protocol explained 82–95% of the variance.
- **Output:**
  - `m*` per objective, for Eρ² ≥ 0.90;
  - `n*` for ±5% precision of the mean, for paper 3;
  - `k` per D2.

### CB — Optimisation comparison (paper 1, thesis)

- **Cells:** {P1, P2} × {NSGA-III, random search}, `R` runs each. GA seeds `1001…1000+R`,
  the same list for every cell.
  - These are independent runs. Sharing the seed numbers is bookkeeping, **not**
    pairing.
- **Engine restarts:** every strategy resumes exactly since 2026-11-22. Do not change the code
  of the engine in the middle of a run: a resume reproduces what the same code would do.
- **Scenario:** the shared `ind1` scenario of paper 1:
  - same mobile nodes, sink, radii, region and duration;
  - P1 with 20 relays;
  - P2 with the 32-candidate set;
  - `min_coverage_percentage 100`.
- **Parameters:** objectives D1, `k` seeds D2, GA D4, firmware `rpl-udp-csma-v2` only
  (the MAC gene is pinned to CSMA by the 2026-11-18 fix).
- **Operator ablation (if D3 = fix):** P1 NSGA-III with the old operator, 5 runs, same
  seeds. It counts in the budget and is not part of the main comparison.

### CC — CSMA vs TSCH on the CB fronts (thesis)

The MAC as a gene inside the optimisation is dropped. TSCH-v2 runs cost ~10× more and
time out under load, and a timeout gives the worst objective: the gene would be judged
by the clock. Instead:

- **Input:** for each problem, the non-dominated valid individuals of the union of all
  CB archives, deduplicated and capped at 60 by HV contribution.
- **Run:** `batch` with `rpl-udp-tsch-v2` and the same `k` seeds. The CSMA values are
  the CB values: same chromosome and seeds, and the simulator is deterministic.
- **Output:**
  - Paired differences per individual and objective: Wilcoxon signed-rank (the pairing
    is real here: same topology, same seeds), Holm-corrected.
  - Kendall τ of the rankings.
  - Fraction of the CSMA front still non-dominated under TSCH.
  - `node_coverage` reported next to every TSCH number.

### CD — Scalability (paper 2)

- **What changes from paper 2:** only the workload firmware moves to `rpl-udp-csma-v2`.
  The benchmark chromosome and the duration stay as before. The network metrics are not
  used: the benchmark measures time.
- **Machines:** one machine at a time, with nothing else on the pool. Record the
  following on the experiment (`name`/notes):
  - CPU model;
  - P/E cores;
  - governor;
  - RAM;
  - JVM heap.
- **Levels and repetitions:**
  - Levels 1, 2, 4, 6, 8, 10, 12, 16 (cut at 1.5× the logical cores);
  - `repetitions ≥ 5`;
  - ~300 sims per machine.
- **Analysis:** USL fit with bootstrap intervals on σ and κ, using
  `paper2-usl-distributed-pool/scripts/usl_benchmarks.py`. Hardware heterogeneity
  (laptop P/E cores) is stated, not averaged away.

### CE — Implementation check (paper 1 Table 3; synthetic)

- **Setup:** DTLZ2 (and the other benchmarks of the table), 30 seeds per implementation
  (SimLab, DEAP, pymoo), `divisions` stated explicitly (12 in the benchmark; the engine's
  default is 10).
- **Tests:**
  - Unpaired Mann–Whitney with Â12. With 5 seeds and a paired Wilcoxon, the smallest
    attainable p was 0.0625.
  - Equivalence claimed only through TOST with a margin declared beforehand.
- **README:** corrected on 2026-11-22: it no longer says "same algorithmic code that runs in
  production". Only sorting, niching and the operators are shared; the variation loop is
  reimplemented.
- **Before running:** `regenerate_table3.py` does not import with the pinned DEAP, and the
  three runners do not use the same parameters or return the same set
  ([known issues](../../experiments/external-nsga3-benchmark/README.md#known-issues)).

---

## 5. Analysis plan (fixed before CB starts)

**Unit of analysis.**

- The run.
- A run's approximation set is the non-dominated set of **all valid evaluations of its
  archive**: the evidence filter of `pylib/evaluation_validity.py`, the same rule for
  every strategy.
- The stored `pareto_front` is not used. In NSGA it is the front of the final
  population, in random search the front of the archive: comparing the stored fields
  favours random search.

**Normalisation.**

- Per problem: ideal and nadir of the union of all runs (all algorithms), mapped to
  [0, 1].
- For P1 vs P2: one normalisation over the union of both problems.
- HV with reference point 1.1 in normalised space (moocore).
- IGD+ against ND(union of all runs of that problem).

**Budget.**

- Indicators as a function of **distinct evaluations** (anytime curves).
- The headline comparison is at `B* = min` over runs of the distinct evaluations
  (NSGA-III spends ~17% of its slots on parent clones).
- The full-budget values are reported too.

**Tests.** Per problem × indicator:

- two-sided Mann–Whitney U;
- Vargha–Delaney Â12 with a bootstrap 95% interval;
- Holm correction over the family (2 problems × 2 indicators);
- medians and IQRs reported.

Intervals are bootstrap (BCa) or t-based with n − 1 degrees of freedom, never ±1.96·s/√n
at these sample sizes. "No difference" is claimed only with TOST on ΔHV, with margin
**0.02** (normalised HV) declared here.

**Always reported:**

- Per run:
  - valid, infeasible and refused evaluations;
  - repeated replicas;
  - timeouts;
  - distinct evaluations;
  - wall time.
- Per experiment: `pareto_front_validity`.

---

## 6. Limitations every text must state

- **Radio:** UDGM with success 1.0, no loss with distance; seed variance is internal to
  the stack.
- **Energy and sleep:** CSMA energy is ~90% idle listening; Cooja has no LPM (`lpm = 0`,
  CPU always on).
- **Repair and search space:**
  - Lamarckian repair (connectivity and coverage) runs after every operator.
  - Random search samples the *repaired* space, not the decision space uniformly.
- **Selection:** NSGA-III mating uses a rank tournament (Deb & Jain use random
  selection).
- **Coverage:** checked on trajectory samples every R/2.
- **Mobility:** shared period of the mobility plugin (D7), unless fixed.
- **Delivery-conditional metrics:** latency, power and hops describe only delivered
  reports; `node_coverage` is shown next to them.
- **P1 crossover:** if D3 is not fixed, the defect is stated.

---

## 7. Checks after each campaign (before analysis)

- [ ] Every manifest id is `Done`, `pareto_front_validity.valid > 0` (CB), and
      `provenance.engine.git_sha` equals the tag.
- [ ] Zero timeouts; zero `refused`. Otherwise, re-run or report the count per cell.
- [ ] One Cooja image ID across the campaign's simulations.
- [ ] Every stored `mac_protocol` matches the firmware that ran.
- [ ] Export with `util/mongo_backup.py`. Archive the manifest and the export together.

---

## 8. Data management

- `experiments/campaign-v2/manifest.csv` has these columns:
  `campaign, cell, run, experiment_id, ga_seed, sim_seeds, launched_at, tag, git_sha, status, notes`.
  Each row is written at launch.
- `experiments/campaign-v2/superseded_v1.csv` lists every v1 experiment the papers
  used, with id and role, from the `RUNS` tables of the paper scripts:
  - `6ae37a63…`, `6aefaeef…647/648`, `6aefaef0…649–64d` (controlled comparison);
  - `6ab859f4…`, `6ab917ff…`, `6ad608b3…`;
  - the panels `6ae5c8c9…`, `6aefb065…`;
  - paper 2: `6acbca2a…`, `6acc689f…`, `6acce14b…`, `6acd5f87…`, `6acd89d7…`, `6acda525…`,
    `6acec243…`, `6acf822d…`, `6acf97e4…`.

  They are not deleted.
- Paper scripts (`SIMLAB-Overleaf/*/scripts/`) switch to reading the manifest instead
  of hard-coded ids. They use the evidence filter, not the 1e8 threshold.
- After the last campaign: one Zenodo dataset record (export, manifest, analysis
  scripts), linked to the software DOI (`PROVENANCE_GUIDE.md` §6).

---

## 9. Mapping to the texts

| Text | Campaigns |
|---|---|
| Paper 1 (P1 vs P2, NSGA-III vs random) | CB (+ CE for Table 3, + CC if the MAC discussion stays) |
| Paper 2 (USL, distributed pool) | CD |
| Paper 3 (seed estimation) | CA (+ the CB individuals as a large panel, like `6ad608b3` before) |
| Thesis | all |
