# Adaptive Simulation Budget — cost/quality comparison

Standalone reproduction of the cost and quality comparison between

| Arm | Strategy | Problem |
|---|---|---|
| classic  | `nsga3` | `problem2` |
| baseline | `nsga3` | `problem2_topology_aware` |
| adaptive | `nsga3_adaptive_simulation` | `problem2_topology_aware` |

The **classic** arm is the historical reference. The **baseline** arm shares the
structural repair operator with the adaptive arm, so the difference between the
last two isolates the *strategy* — not the problem variant.

Like [`external-nsga3-benchmark/`](../external-nsga3-benchmark/), this script is
independent of the SimLab runtime: no MongoDB, no master-node, no Cooja. It
drives the real strategy classes through the real generation lifecycle using the
in-process fakes from [`mo-engine/tests/adaptive_fakes.py`](../../mo-engine/tests/adaptive_fakes.py),
with a deterministic closed-form evaluator standing in for the simulator.

## Two harnesses

* `run_comparison.py` — does the adaptive strategy spend less simulation
  budget than plain NSGA-III, and at what cost in front quality?
* `run_encoding_comparison.py` — does dropping the repair pass in favour of a
  feasibility-closed tree encoding cost anything? Same strategy, same seed,
  the three P2 encodings.

## Quick start

```bash
python experiments/adaptive-simulation/run_comparison.py
python experiments/adaptive-simulation/run_comparison.py \
    --population 24 --generations 10 --seed 7 \
    --kappa 1.96 1.0 0.5 --json report.json
```

```bash
python experiments/adaptive-simulation/run_encoding_comparison.py --repeats 3
```

Only `numpy` and `pymongo` (for `bson.ObjectId`) are needed — both already in
[`mo-engine/requirements.txt`](../../mo-engine/requirements.txt).

## Encoding comparison

`run_encoding_comparison.py` runs NSGA-III unchanged over `problem2`,
`problem2_topology_aware` and `problem2_tree_encoded`, so the only variable is
how feasibility is obtained — global BFS repair, structural repair, or an
encoding whose operators cannot produce an infeasible individual. It also
asserts the invariant directly, by rebuilding the sink-rooted tree of every
persisted chromosome and counting the disconnected ones (must be zero for all
three).

## What it reports

**Cost** — real evaluations per arm, exact-cache reuses, estimate-only
individuals, promotions, audit simulations, and

$$\text{simulation reduction ratio} = 1 - \frac{N_\text{sim adaptive}}{N_\text{sim baseline}}$$

**Quality** — hypervolume ratio against the baseline front (Monte-Carlo, fixed
seed, shared reference box), Pareto recall and Pareto precision.

**Estimator** — MAE / RMSE of the predictions that were later measured for real
(promotions and audit samples), and the *false skip rate*: the fraction of
audited skips whose real objectives turned out to be non-dominated.

## Reading the κ sweep

κ scales the confidence band: an individual is only skipped when its optimistic
estimate $L(x) = \hat f(x) - \kappa\hat\sigma(x)$ is *clearly* dominated by a
really-evaluated solution. Larger κ ⇒ more conservative ⇒ fewer skips.

The sweep exists because the useful operating point is problem-dependent. On a
three-objective landscape the 95% band (κ = 1.96, the default) is conservative
enough that little gets skipped once the population converges onto the front —
most offspring are then genuinely near-front and *deserve* a simulation. The
savings come from the regions where the estimator is confident and the candidate
is clearly bad, and how large those regions are depends on how orderable the
objective landscape is.

Run the sweep on your own scenario before fixing κ for a campaign.

## The stand-in evaluator

`TopologyEvaluator` computes the three objectives as smooth functions of the
deployed topology — relay count `n` and mean distance to the sink:

```
energy     = 5 + 2.0·n + 0.01·n·spread/10                  (min)
latency    = 8 + 120/(1+n) + 0.20·n + 0.06·spread          (min)
throughput = 100·n/(n+8) − 0.60·n − 0.15·spread            (max)
```

Latency and throughput are deliberately **not monotone** in `n`: extra relays
shorten routes up to a point, past which forwarding overhead dominates. That
inflection is what produces a bounded Pareto front with genuinely dominated
regions — the regime in which "is this individual worth simulating?" is a
meaningful question at all. A landscape where every relay count is trivially
non-dominated would show zero savings for any heuristic, and a strictly ordered
one would show implausibly large savings.

**These numbers characterise the mechanism, not Cooja.** The absolute reduction
on a real campaign depends on the noise floor of the metrics, the number of
seeds per individual and how quickly the population converges. What transfers
is the shape of the trade-off, and the guarantee that quality is bounded by the
promotion step.
