# Seed Study

How many simulation seeds an instance needs, measured instead of assumed. A
study runs one simulation per seed, with everything else fixed, and reports how
every metric is distributed and how many seeds each objective needs.

Open it from the sidebar (**Seed Study**). The analysis lives in
[`pylib/seed_study.py`](../../pylib/seed_study.py), the run in
[`mo-engine/lib/strategy/seed_study.py`](../../mo-engine/lib/strategy/seed_study.py)
and the routes in
[`rest-api/api/endpoints/seed_study.py`](../../rest-api/api/endpoints/seed_study.py).

## Two questions, two numbers

"How many seeds" has two answers, and they differ by orders of magnitude.
Measured on P1 with 156 individuals and 43 seeds:

| Question | Symbol | `latency` | `energy` | `throughput` |
|---|---|---:|---:|---:|
| To **report** the value of one solution within ±5% | `n*` | 244 | 3 | 11 |
| To **rank** solutions with reliability 0.90 | `m*` | 6 | 8 | 4 |

- `n*` depends on the noise of one solution. Use it to publish a value.
- `m*` depends on the noise relative to the difference between solutions. Use it
  to configure an optimisation. It needs a panel of solutions and has no answer
  with a single one.

Neither is a property of the system. Both hold for the instance, duration,
image and aggregator of the study, and the page says so next to the number.

## Running a study

| Field | Default | Notes |
|---|---|---|
| Problem instance | | A saved problem |
| MAC protocol, firmware | | **One per study.** A panel that mixes protocols counts the protocol as a difference between solutions: on a csma+tsch experiment it explained 82% to 95% of the variation of the objectives, which would make `m*` far too small. To optimise the MAC gene, run one study per protocol and use the larger `m*` |
| Duration | 180 min | Every conclusion holds for this duration |
| Solutions | Panel of 20 | Random feasible solutions, distinct once their firmware is resolved. Below 10, `m*` swings too much between panels. One solution answers how it varies, not how many seeds an optimisation needs |
| Seeds | 30 generated | Or a file, `{ "seeds": [...] }` or a bare list, the format the launch wizard exports |
| Objectives | latency, energy, throughput | Use the aggregators the optimisation will use: the study evaluates that estimator |
| Targets | 95%, ±5%, 0.90 | Can change later without simulating again |
| Control repeats | 2 | Repeats of the first seed on every solution |

The form shows the exact number of simulations before starting. The time is an
estimate from recent runs of the same duration.

A random panel stands for the start of a search. Solutions from a converged
front are more alike, which can ask for more seeds. To study them, send them as
`panel.source = "provided"` through the API.

### What the seed reaches

Read before interpreting a low variation. In
`juniocesarferreira/simulation-cooja:v1.1`:

| Source | Reach of the seed |
|---|---|
| Boot delay, up to 1 s per mote | The whole seed |
| Firmware PRNG, `srand((unsigned short)(seed + moteID))` | **Only 16 bits** |
| Radio, UDGM with success ratio 1.0 | **None**: no random loss |

So the distribution measured is internal variability, not channel variability.
A low CV means the result is stable *in this radio model*. Changing the image
can change what a seed means, so a study only holds for the image it ran on.

The 16 bits do not make seeds that share them equivalent. Cooja saves only the
`data`, `bss` and `common` sections of the mote library per mote, so the state
of the libc `rand()` is outside them and shared between motes, and the boot order
comes from the whole seed. Measured on P1 with one solution and 20 triples
`s, s+1, s+65536`, under the Spearman correlation between the pairs, 21 of 22
tests gave no sign of resemblance; `total_sent` had ρ ≈ 0.45 for both pairs, which
does not survive a correction for 22 tests. That sample detects a strong
resemblance, not a weak one. So generated lists still keep their low 16 bits at
least 64 apart, which costs nothing, and an imported list that does not is
accepted with a caution.

## Reading the result

**Above everything**, the page states what would undermine the study: a control
that diverged, results that are missing, seeds that are not independent, or more
than one worker image.

**The control.** Determinism is the premise: if repeating a seed changes a
number, the distribution is not the distribution of the seed. The repeats run in
a generation of their own, tagged `seed_study: {role: "control"}`, and stay out
of the distributions.

**Metrics** are classified before any statistic is taken:

| Status | Meaning |
|---|---|
| `ok` | Usable |
| `extrapolated` | `n*` is past the seeds that ran |
| `indicative` | `n*` is more than three times the seeds that ran, or fewer than 10 ran |
| `constant` | Does not change with the seed. "Always zero" says it carries nothing |
| `measurement defect` | Values past 2⁵⁰: an unsigned subtraction wrapped around, even if diluted in a run's mean. No recommendation |
| `near zero` | The interval of the mean contains zero, so a relative error has no meaning. The page asks for an absolute one |
| `no precision target` | A minimum or maximum, whose expected value moves with the number of seeds. The Precision chart shows how it moves |
| `beyond the sample` | Resampling did not reach the target within the seeds that ran |

Metrics with identical numbers are grouped under one name, since `latency` and
`rtt_latency` read one column.

**What each number of seeds allows:**

| n | Output |
|---:|---|
| < 10 | Values, mean, S. `n*` is indicative |
| ≥ 10 | Intervals |
| ≥ 30 | The shape of the distribution, and `n*` as an estimate |
| ≥ 29, 59, 299 | Distribution-free upper limit of p90, p95, p99 |

## Method

Each seed is an independent replication of a terminating simulation, so nothing
is removed as warm-up. The analysis is a pure function of the per-seed values:
same data, same `analysis_version`, same numbers. Resampling draws from a
generator derived from the seed list.

**To report a value.** For a half-width `h = γ'|ȳ|` with `γ' = γ/(1+γ)`:

```text
n* = min { i ≥ 2 : t(i−1, 1−α/2) · S / √i ≤ h }
```

The adjustment is what bounds the error relative to the *true* value: if
`|ȳ−μ| ≤ γ'|ȳ|`, then `|ȳ−μ|/|μ| ≤ γ'/(1−γ') = γ`. Next to `n*` the page shows
the range from the BCa interval of S, Stein's two-stage total, and a
non-parametric second opinion: the probability that `k` resampled seeds land
within the same half-width, by `k`. On a panel the recommendation is the median
across solutions, with p90 and the maximum, since the noise varies widely
between solutions (the CV changed 47-fold on the data above). For `J` objectives
holding together, confidence is `1 − α/J` (Bonferroni).

| Aggregator | Criterion |
|---|---|
| `mean` | `n*` |
| `median`, `trimmed_mean` | Resampling only, with the real estimator |
| `min`, `max` | None: the target moves with the number of seeds |

**To rank solutions.** The crossed random-effects model
`y = μ + aᵢ + bₛ + eᵢₛ`, one per metric, gives by generalizability theory:

```text
Eρ²(m) = σ²ₐ / (σ²ₐ + σ²ₑ/m)          ranking with shared seeds
Φ(m)   = σ²ₐ / (σ²ₐ + (σ²_b + σ²ₑ)/m)  against an absolute threshold
m*(ρ)  = ⌈ ρ/(1−ρ) · σ²ₑ/σ²ₐ ⌉
```

Intervals use the pigeonhole bootstrap, resampling solutions and seeds. Since
the model assumes a variance the data does not have, the page also shows the
Spearman correlation between rankings from two disjoint sets of seeds, which
assumes nothing; it is the answer for aggregators other than the mean. The
similarity of Pareto fronts between such sets is shown as a diagnostic only:
with many near-ties, membership keeps changing after the ranking has settled.

## What the system refuses to do

- **Choose seeds by their results.** The page never offers "best seeds". Using
  fewer seeds means the first `m` of a list whose order was fixed before any
  result. The main effect of the seed was at most 0.5% of the variation, so
  there is nothing to find, and a selection would bias the evaluation.
- **Remove outliers.** They are marked and stay in every calculation.
- **Extend a study by looking at it.** Adding seeds until an interval is narrow
  enough lowers its real coverage.
- **Treat a study as universal.** Every recommendation carries its scope.

## API

| Route | |
|---|---|
| `POST /seed-studies/` | Queue a study. Answers with the cost and the diagnosis of the list |
| `GET /seed-studies/` | Every study, with progress |
| `GET /seed-studies/{id}?confidence=&relative_error=&reliability=&absolute=` | The study and its analysis, as of now. `absolute` is `metric:value,...` |
| `GET /seed-studies/{id}/metrics/{metric}?chromosome=` | One metric on one solution, with what its charts need |
| `GET /seed-studies/{id}/values.csv` | Every value, as `chromosome, seed, order, metric, value, status, simulation_id` |
| `GET /seed-studies/estimate?duration=` | The mean time of recent runs of that duration |

```json
{
  "name": "p1 · 30 seeds",
  "problem": { "name": "problem1", "...": "as the wizard exports it" },
  "source_repository_options": { "csma": "<id>" },
  "duration": 180,
  "seeds": { "mode": "generated", "count": 30, "master_seed": 20261030 },
  "panel": { "source": "generated", "size": 20, "random_seed": 7741 },
  "objectives": [{ "metric_name": "latency", "goal": "min", "aggregator": "mean" }],
  "targets": { "confidence": 0.95, "relative_error": 0.05, "reliability": 0.9 },
  "control_replicates": 2
}
```

Imported seeds: `"seeds": { "mode": "imported", "values": [...], "file_name": "..." }`.
A given panel: `"panel": { "source": "provided" }` with `"chromosomes": [...]`.
A `synthetic` block, as in an experiment, runs the study without Cooja. Limits:
3 to 500 seeds, up to 100 solutions, up to 5,000 simulations. Unknown fields are
refused with a 422.

A failed seed is repeated once with the same seed, which the determinism of the
simulator makes harmless. If it fails again the study goes on with what it has
and reports what is missing, separating failures of the infrastructure from
failures of the run: the second kind may depend on the seed, and the distribution
is then conditional on the seeds that finished.

## Validation

- **Coverage** of the intervals and the recovery of known variance components,
  by Monte Carlo, in `pylib/tests/test_seed_study.py`.
- **Parity** of the vectorised aggregators with `AGGREGATOR_DISPATCH`, so the
  study evaluates what the engine computes.
- **Reproduction** of `m*` of 6, 8 and 4 and of the variance shares above with
  the 43-seed data.
- **End to end without Cooja.** A synthetic study with `noise_std = 0.02`, 20
  solutions and 30 seeds returned S ≈ 0.02, σ_e ≈ 0.02 and σ_b ≈ 0 through the
  API, the engine and the master-node.
- **Determinism beyond P1.** Studies on P2 and P3 reproduced every metric on the
  repeated seed. P4 has no saved instance yet, and the control of its first study
  will answer for it.
