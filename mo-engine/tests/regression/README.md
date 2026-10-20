# NSGA / HV / GD regression baseline

Corrections to NSGA operators and metrics change the numbers **deliberately**.
Without a frozen baseline, intended corrections cannot be distinguished from
accidental regressions. That is the purpose of this directory.

## What is measured

[`kernel.py`](kernel.py) runs SimLab's actual reproduction and environmental
selection methods on an analytical benchmark, synchronously: no MongoDB, change
streams, or workers. A run is a pure function of `(config, seed)`.

Frozen grid: native NSGA-II and NSGA-III × DTLZ2 (M=3, n=10), ZDT1 (M=2, n=10),
and SCH1 (M=2, n=1) × seeds 1, 2, 3, 5, 7 — population 50, 20 generations,
zero noise, **launch wizard defaults**. The defaults track what the interface
actually sends, so the baseline continues to measure the platform as shipped;
Phase 6 changed them to `prob_mt=1.0`, `per_gene_prob=1/n`, and `divisions`
derived from M and the population size.

Three sets are recorded per generation because the audit found that the platform
and the notebooks plotted different populations:

| Set | Definition | Current use |
| --- | --- | --- |
| `offspring` | `ND(Q_t)`, the newly generated children | explicit option and per-generation fallback in the `/hv-gd` endpoint |
| `survivors` | `ND(P_t)`, the population after selection | default in the `/hv-gd` endpoint and the NSGA-Studies notebooks |
| `archive` | `ND` of everything evaluated so far | Archive option for HV, GD, IGD, and IGD+ |

HV, GD, IGD, and IGD+ are computed **exactly as in the endpoint**
(HV with `1.1 × nadir`, using only points that strictly dominate the reference;
GD/IGD normalized by the benchmark's theoretical ideal–nadir range), so a
baseline value can be compared with an actual experiment's series.

**GD is the exact distance** to the analytical front
(`benchmarks.front_distance`), not the mean nearest-neighbor distance to a
sampled reference. Phase 0 recorded this in a separate column, `radial_error`;
Phase 4 promoted that same quantity to GD, making the extra column redundant,
so it was removed. IGD and IGD+ still use the sampled reference — which is what
allows them to measure coverage — and therefore retain its discretization floor.

Both the kernel and production apply environmental selection to the final batch
of children. The final front corresponds to the selected survivors.

## Usage

```bash
cd mo-engine
../.venv/bin/python -m tests.regression.baseline --summary   # baseline table
../.venv/bin/python -m tests.regression.baseline --check     # compare (exit 1 on mismatch)
../.venv/bin/python -m tests.regression.baseline --write     # freeze a new baseline
../.venv/bin/python -m pytest tests/regression -q
```

When changing operators or metrics, run `--check`, **read the diff and confirm
that the change is intended**, then run `--write` and advance `BASELINE_STAGE`
in [`baseline.py`](baseline.py). Freezing a new baseline without reviewing the
diff defeats the purpose of this directory.

## `phase-6` baseline (30 runs)

`drop` is the largest single-step decrease in HV across all seeds.

```
config                     HV off    HV surv   drop off  drop surv    GD surv
-----------------------------------------------------------------------------
nsga2-dtlz2-m3-n10       0.415902   0.475401   0.095484   0.025915   0.170337
nsga3-dtlz2-m3-n10       0.546410   0.568859   0.048615   0.007782   0.078157
nsga2-zdt1-m2-n10        0.560509   0.570737   0.093423   0.000000   0.239160
nsga3-zdt1-m2-n10        0.528164   0.544913   0.016820   0.000000   0.244904
nsga2-sch1-m2-n1        16.395027  16.550427   1.252112   0.047228   0.000109
nsga3-sch1-m2-n1        16.400878  16.521825   1.252112   0.040700   0.000063
```

### What Phase 6 changed (defaults)

| config | HV surv Phase 4 | HV surv Phase 6 | Δ |
| --- | ---: | ---: | ---: |
| nsga2-dtlz2-m3-n10 | 0.495511 | 0.475401 | −0.020 |
| nsga3-dtlz2-m3-n10 | 0.569152 | 0.568859 | −0.000 |
| nsga2-zdt1-m2-n10 | 0.509439 | **0.570737** | +0.061 |
| nsga3-zdt1-m2-n10 | 0.447513 | **0.544913** | +0.097 |
| nsga2-sch1-m2-n1 | 16.518083 | 16.550427 | +0.032 |
| **nsga3-sch1-m2-n1** | 16.110428 | **16.521825** | **+0.411** |

The improvement in `nsga3-sch1` comes from `divisions`: at M=2, the fixed value
10 produced 11 reference directions for 50 individuals. The derived value gives
49 divisions (50 directions, one per slot). This was precisely the case that
Phase 0 identified as having the largest HV decrease among survivors (0.569),
which Phase 2 reduced to 0.229 — it is now 0.041.

The −0.020 for `nsga2-dtlz2` is within the variation across seeds (a standard
deviation of 0.039 measured over 15 seeds). In a comparison using 15 seeds and
40 generations, textbook mutation is neutral on DTLZ2 and clearly better on ZDT1:

| problem | algorithm | 0.1 × 0.05 | 1.0 × 1/n |
| --- | --- | ---: | ---: |
| DTLZ2 M=3 | NSGA-II | 0.545873 ± 0.039 | 0.550428 ± 0.034 |
| DTLZ2 M=3 | NSGA-III | 0.633169 ± 0.023 | 0.635412 ± 0.021 |
| ZDT1 M=2 | NSGA-II | 0.619231 ± 0.089 | **0.801769 ± 0.026** |
| ZDT1 M=2 | NSGA-III | 0.632825 ± 0.082 | **0.805805 ± 0.017** |

Besides the change in the mean, the standard deviation falls by a factor of
three on ZDT1 — the old rate left the search highly dependent on the initial
population.

Phase 4 did not change HV; only GD changed, by replacing the sampled reference
with the exact distance. The decrease in the `GD surv` column is exactly the
discretization error previously interpreted as a lack of convergence:

| config | GD before (Phase 2) | GD after (Phase 4) | error removed |
| --- | ---: | ---: | ---: |
| nsga2-dtlz2-m3-n10 | 0.149428 | 0.141762 | 0.0077 |
| nsga3-dtlz2-m3-n10 | 0.085026 | 0.075072 | 0.0100 |
| nsga2-zdt1-m2-n10 | 0.265797 | 0.265773 | 0.00002 |
| nsga3-zdt1-m2-n10 | 0.315556 | 0.315554 | 0.00000 |
| **nsga2-sch1-m2-n1** | 0.000835 | **0.000034** | 25× |
| **nsga3-sch1-m2-n1** | 0.001048 | **0.000270** | 4× |

ZDT1 barely changes because those runs stop far from the front: discretization
error is negligible compared with the actual distance. SCH1 is the opposite —
the population gets so close that almost all reported GD came from the reference.

### `pre-fix` (Phase 0), for comparison

```
nsga2-dtlz2-m3-n10       0.498542   0.533516   0.039925   0.017707   0.099148
nsga3-dtlz2-m3-n10       0.581202   0.620339   0.042602   0.015225   0.054280
nsga2-zdt1-m2-n10        0.456881   0.464750   0.095775   0.000000   0.253981
nsga3-zdt1-m2-n10        0.417986   0.432678   0.055778   0.000000   0.295333
nsga2-sch1-m2-n1        16.339305  16.449792   0.443291   0.023870   0.000827
nsga3-sch1-m2-n1        15.653757  15.741825   1.316102   0.569466   0.000964
```

The `pre-fix` baseline reproduced the audit exactly (NSGA-III/DTLZ2:
HV 0.581202 → 0.620339; decrease 0.042602 → 0.015225), validating the harness
as a faithful port of the audit script.

### What Phase 2 changed, by operator

Ablation with each old behavior restored independently, NSGA-III, survivor HV,
mean over 5 seeds:

| SBX | niching | DTLZ2 M=3 | SCH1 M=2 |
| --- | --- | ---: | ---: |
| old | old | 0.620339 | 15.741825 |
| old | new | 0.580534 | **16.148035** |
| new | old | 0.575054 | 15.741825 |
| new | new | 0.569152 | 16.110428 |

- **Phase 2.2 exit criterion met**: `nsga3-sch1` was the case where survivors
  also showed large decreases. The maximum single-step decrease falls from
  0.569 to 0.229, and the improvement is attributable to niching (SBX barely
  affects SCH1 with n=1).
- **DTLZ2 gets worse, and the effect is real** — it persists at 60 and 150
  generations and under 1/n mutation. See the next section.

### Why DTLZ2 gets worse with correct operators

The old SBX used the distance to the **lower** bound for both children. For
parents near the upper bound, this spreads the upper child too far, after which
it is clamped exactly to the bound:

| Parent samples | children clamped to the bound (before) | (after) |
| --- | ---: | ---: |
| uniform in [0, 1] | 0.543% | 0.000% |
| in [0.85, 1.0] | 0.597% | 0.000% |
| in [0.0, 0.15] | 0.000% | 0.000% |

The defect was asymmetric: it produced solutions exactly on the **upper** bound
and never on the lower bound. In DTLZ2, the position variables must reach 0 or
1 to produce the front's corner solutions, which contribute the most to HV.
The bug supplied half of those corners for free. The corrected SBX matches
DEAP — children never leave the box — and reaches the corners only
asymptotically.

This is no reason to revert: the old operator demonstrably used the wrong
distribution (exact parity with DEAP is tested in `test_sbx_bounded.py`), and
its advantage is an artifact of a single benchmark. However, DTLZ2 numbers
published before Phase 2 depended on that artifact.

### Tournament selection with crowding (Phase 2.3)

The effect is within the noise over 5 seeds; the clear effect is increased
variance on DTLZ2 (M=3), consistent with the known weakness of crowding distance
in three or more objectives — the premise that motivated NSGA-III:

| tie-breaker | mean HV | standard deviation | per seed |
| --- | ---: | ---: | --- |
| random | 0.522031 | 0.017574 | 0.508 0.531 0.549 0.509 0.514 |
| crowding | 0.495511 | 0.082897 | 0.484 **0.361** 0.518 0.537 0.579 |

A conclusive comparison requires more seeds and a significance analysis;
this table alone does not establish one.

## Versions

The baseline records `numpy`, `pymoo`, `deap`, and `moocore`. `--check` warns
when they differ: a numerical difference under other versions is not
necessarily a SimLab regression.

## Validation of integration fixes

During the 2026-09-15 review, nine API cases and two engine cases reproduced
failures before the fixes. All six suites passed, with **857 tests passed and
one skipped**; frontend type checking, lint, and build also passed.

Coverage includes:

- **Objective permutation:** the interior ZDT1 point `(0.25, 0.5)` must have
  approximately zero GD in both orders; before the fix, the reversed order
  returned `0.035137`. An off-front point is also checked, together with the
  invariance of HV, IGD, and IGD+, in both raw and normalized units.
- **Partial history:** fallback to offspring only for generations without
  `survivors`, explicit provenance for each point, preservation of empty sets,
  and independence of the cumulative archive.
- **DEAP concurrency:** global RNG contexts do not overlap; draws match isolated
  executions; state and lock are restored after exceptions. New uses of the
  global RNG must honor the same lock or use an independent `Generator`.
- **Resume:** eight evolution steps, six backends, seeds 1 and 42, checkpoints
  after 0, 1, and 4 evolution steps, and pending or completed generations —
  **72 combinations**. The test uses DTLZ2, three objectives, six variables,
  and a population of 12, requiring exact equality of individuals, survivors,
  and RNG state.
- **pymoo persistence:** BSON round-trip, snapshots without shared memory,
  readable legacy records, and explicit rejection of malformed states.
- **Frontend:** labels for mixed series and compatibility with older responses.

Resume tests use actual enqueue, selection, restoration, and evolution code,
with MongoDB, asynchronous triggers, and uploads replaced by test doubles.
The results are neither integration tests with a real database and external
workers nor a general proof of convergence or equivalence across library
versions. The [execution protocol](../../../docs/markdown/SYNTHETIC_MODE.md)
documents the limitations of legacy checkpoints and the interpretation of
mixed series.
