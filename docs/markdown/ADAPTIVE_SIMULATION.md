# Adaptive Simulation Budget

> **Hypothesis under test.** Topological knowledge + simulation history +
> uncertainty ⟶ a better allocation of the simulation budget.
>
> Solutions whose structure resembles historically poor configurations, and
> whose *optimistic* estimate is already dominated, can skip a full Cooja
> evaluation — while novel, uncertain or potentially non-dominated solutions
> keep receiving a real one.

The goal is **not** to replace the simulator. It is to concentrate the
computational cost on the regions of the search space with the highest
potential to change the population and the Pareto-front approximation.

---

## 1. What is new

Three additions, all side-by-side with the existing implementations, which are
untouched:

| | Existing | New |
|---|---|---|
| Problem | `problem2` → `Problem2DiscreteMobilityAdapter` | `problem2_topology_aware` → `Problem2TopologyAwareAdapter`<br>`problem2_tree_encoded` → `Problem2TreeEncodedAdapter` |
| Strategy | `nsga3` → `NSGA3LoopStrategy` | `nsga3_adaptive_simulation` → `NSGA3AdaptiveSimulationStrategy` |

The **phenotype** is unchanged in every variant: the binary mask over the
candidate set `Q`, hashed the same way, encoded to Cooja the same way. That is
what keeps results comparable one-to-one with a classic P2 run and lets the
three variants share `genome_cache` entries.

`problem2_tree_encoded` additionally stores the sink-rooted tree *inside* the
chromosome (`ChromosomeP2Tree.tree_parents`), which is genotype only — ignored
by equality and hashing — and exists so the operators can be closed over
feasible solutions. See section 4b.

---

## 2. Component map

```
                    ┌──────────────────────────────────────────┐
                    │  NSGA3AdaptiveSimulationStrategy         │
                    │  (orchestration only)                    │
                    └───┬──────────────────┬───────────────┬───┘
                        │                  │               │
       ┌────────────────▼───────┐  ┌───────▼──────────┐  ┌─▼───────────────────┐
       │ Problem2TopologyAware  │  │ AdaptiveEvalua-  │  │ AdaptiveMetrics     │
       │ Adapter                │  │ tionPolicy       │  │ Tracker             │
       │  └ Problem2TreeEncoded │  │                  │  │                     │
       │     (repair-free ops)  │  │                  │  │                     │
       └───┬────────┬────────┬──┘  └───┬──────────┬───┘  └─────────────────────┘
           │        │        │         │          │
   ┌───────▼──┐ ┌───▼─────┐ ┌▼───────┐ │   ┌──────▼──────────────┐
   │ Scenario │ │Topology │ │Routing │ │   │ ObjectiveEstimator  │
   │ Topology │ │Repair   │ │Knowl.  │ │   │ (WeightedKNN)       │
   └────┬─────┘ └────┬────┘ └────────┘ │   └─────────────────────┘
        │            │                 │
   ┌────▼────────────▼──┐    ┌─────────▼────────────────┐
   │ RootedTreeBackend  │    │ EvaluationKnowledgeBase  │
   │ ├ ParentArrayTree  │    │ (ground truth only)      │
   │ └ TwoLevelTree     │    └──────────────────────────┘
   └────────────────────┘
        │  ▲
        │  └── TreeOperators (PAO/CAO/grow/prune/transplant)
        │
   ┌────▼──────────────────────┐
   │ TopologyDescriptorExtractor│  →  φ(x)
   └───────────────────────────┘
```

Nothing under `lib/adaptive/` knows about P2, Cooja or MongoDB: it consumes
descriptor vectors and objective vectors in minimization space.

---

## 3. The sink-rooted tree

`RootedTreeBackend` is a Protocol (`root`, `parent`, `children`, `depth`,
`path_to_root`, `subtree_nodes`, `cut_subtree`, `link`, `reroot_component`,
`is_connected_to_root`). Two implementations satisfy it and are
interchangeable everywhere: `ParentArrayTree` (parent map + children lists) and
`TwoLevelTree` (segmented preorder sequence, `O(sqrt(n))` splices — section
4b).

**This is not the RPL DODAG.** It is the structural connectivity skeleton of a
chromosome, derived from geometry alone. The DODAG that Cooja actually forms is
observed *after* a simulation and enters through `RoutingKnowledge`.

Construction (`build_sink_rooted_tree`) is a Dijkstra shortest-path tree rooted
at the sink over the *active* relays, with edge cost

```
c(i, j) = w_d · d̂(i, j)²  +  w_r · (1 − r_ij)
```

`w_r` defaults to 0, so with no history the cost is exactly the squared
normalised distance. Relays that cannot reach the sink through other active
relays stay in the tree as **detached** nodes — the exact signal the repair
consumes. Ties break on the candidate index, so `(scenario, mask)` always
yields the same tree.

---

## 4. Structural repair

Replaces the global BFS repair (`repair_connectivity_to_sink`) *in this variant
only*:

1. build the tree, read off the detached relays;
2. run a Dijkstra from the **whole sink component** over the full candidate
   graph — traversing an already-active relay is nearly free, activating an
   inactive one costs `span − H(u,v) + ε > 0` with

   ```
   H(u, v) = w₁·I_v + w₂·(1 − d̂(u,v)) + w₃·Q_v − w₄·C_v
   ```

   `I_v` is historical link importance (0 without history), `Q_v` blends
   connectivity degree and trajectory reach, `C_v` is a uniform per-relay
   activation cost — which is what makes the search minimise the *number* of
   added relays;
3. activate the bridge, repeat until nothing is detached or the budget
   (`repair_budget`, default `max(16, |Q|)`) runs out.

A component with no admissible bridge is **deactivated** rather than left
dangling, so the returned mask always satisfies "every active relay reaches the
sink". Each activated candidate is adjacent to its predecessor by construction,
so the repair can never activate an unreachable relay. If the budget is
exhausted with relays still detached, `RepairResult.feasible` is `False` and
`penalty_objectives` returns a structural penalty (`5e9`) — above the coverage
penalty (`1e9·(1+deficit)`), below the "no metrics" sentinel (`1e12`).

---

## 4b. Tree encoding — feasibility without repair

`problem2_tree_encoded` is the same problem again, with the repair pass
*removed rather than improved*. It subclasses the topology-aware adapter, so it
keeps the scenario caches, the descriptors, the fingerprint and the routing
knowledge; only the variation pipeline changes.

### The structure

`TwoLevelTree` stores the sink-rooted **forest** as its preorder walk with
depths — the node-depth linearization — cut into `~sqrt(n)` contiguous
**segments**:

```
[(sigma,0) (a,1) (b,2) (c,2) (d,1) | (e,0) (f,1)]
 |<---- tree rooted at sigma ---->| |<-fragment->|
```

A subtree is always a *contiguous range*, so moving one is a splice. The
two-level index is what keeps that splice cheap:

| operation | cost |
|---|---|
| `precedes(a,b)`, `sequence_index(v)` | `O(1)` |
| `cut_subtree` / `link` of an `m`-node block | `O(m + sqrt(n))` |
| `rebalance` | `O(n)`, amortised `O(sqrt(n))` |

The TSP form of this structure carries a per-segment **reversal bit** so 2-opt
can flip a sub-path in `O(1)`. Rooted trees have a fixed orientation, so there
is nothing to reverse and no reversal bit is kept; re-rooting a fragment — the
one operation that does invert a path — is explicit, in `O(depth)`. That is the
only deliberate deviation from the published structure.

`TwoLevelTree` and `ParentArrayTree` both satisfy `RootedTreeBackend` and are
selected by `problem.tree_encoding.backend`; a differential test drives an
identical random operation sequence through both and asserts they stay
indistinguishable.

### Why no repair is needed

Deactivated relays are not deleted — they stay in the forest as **detached
fragments**, so "activate" and "deactivate" are the same cut/link primitives.
Every operator then preserves feasibility by construction:

| operator | move | why it stays feasible |
|---|---|---|
| **PAO** | re-hang a subtree under a new parent | new parent is attached and within `R_com` |
| **CAO** | re-root a fragment, then re-hang it | every inverted edge already existed |
| **grow** | link an inactive candidate as a leaf | an admissible parent is required |
| **prune** | cut a subtree | cutting a subtree from a tree leaves a tree |
| **transplant** (crossover) | graft a donor subtree | the block keeps its shape; the graft point must be admissible |

Coverage is handled the same way: `grow_to_coverage` runs the same greedy
set-cover as the mask variant, but restricted to the *admissible frontier*, so
unlike `greedy_coverage_repair_mask` it can never require a follow-up
connectivity fix. `_structural_repair_mask` is overridden to raise, so a future
edit that reintroduces a repair call fails immediately instead of silently
re-enabling the behaviour this encoding exists to remove.

### Genotype vs phenotype

`ChromosomeP2Tree` extends `ChromosomeP2` with `tree_parents[i]` — the parent
of candidate `i`, `-1` for a child of the sink, `-2` for an inactive one.

The tree is **genotype only**: `__eq__`, `__hash__` and `get_hash` ignore it and
stay identical to `ChromosomeP2`, because two trees over the same relay set
deploy the same motes and would run the same simulation. Consequences, all
intended:

* a tree-encoded experiment **shares `genome_cache` entries** with a
  mask-encoded one — results transfer both ways;
* `Individual.individual_id` is unchanged, so nothing downstream notices;
* `to_dict` still emits the array, so the exact tree survives a restart, and
  `chromosome_from_dict` falls back to the canonical tree when it is absent;
* descriptors phi(x) keep using the **canonical** shortest-path tree of the
  mask, not the genotype tree, so they stay a deterministic function of
  `scenario + chromosome` as the estimator requires.

A plain `ChromosomeP2` is accepted wherever the operators expect a tree — the
canonical tree is rebuilt — so a tree-encoded run can be seeded from a
mask-encoded population.

### Measured effect

`experiments/adaptive-simulation/run_encoding_comparison.py`, NSGA-III on all
three encodings, population 20, 8 generations, averaged over 3 seeds:

| problem | evals | mean relays | front | HV | disconnected |
|---|---|---|---|---|---|
| `problem2` | 163.7 | 23.6 | 13.0 | 18577 | 0 |
| `problem2_topology_aware` | 162.3 | 23.5 | 12.7 | 18146 | 0 |
| `problem2_tree_encoded` | **132.0** | **17.6** | **19.7** | **22368** | 0 |

The tree encoding reaches a better front with fewer relays and fewer real
evaluations. The plausible reason is that repair has an **activation bias** —
it only ever *adds* relays to restore connectivity — which drags the population
towards denser, more expensive networks, while the tree operators prune as
freely as they grow. Treat that as a hypothesis the harness now lets you test,
not a settled result: the numbers come from the closed-form stand-in evaluator,
not from Cooja.

---

## 5. Descriptors φ(x)

20 **structural** descriptors, deterministic for a fixed `scenario + chromosome`
and computed entirely from the pre-computed caches:

```
active_relays  relay_ratio  number_of_edges  mean/min/max_degree
connected_components  sink_reachability_ratio
mean/max_tree_depth  tree_leaves  tree_branching_factor
mean/max_distance_to_sink  mean/max_hop_count
trajectory_coverage_ratio
minimum/mean_temporal_connectivity  critical_time_slices
```

plus a **historical** block that is computed, persisted and used for novelty,
but deliberately **kept out of the regression input**:

```
routing_importance_sum / _mean / _max
nearest_evaluated_hamming_distance  nearest_evaluated_descriptor_distance
```

Reason: their value for a fixed chromosome drifts as the knowledge base grows,
which would silently invalidate every descriptor stored in earlier generations.
Normalisation is min-max, fitted by the estimator on its own training set.

---

## 6. Knowledge base and exact cache

`EvaluationKnowledgeBase` holds `D = {(φ(xᵢ), xᵢ, f(xᵢ))}` for **one**
`scenario_fingerprint`. Records from another fingerprint are rejected outright.

It introduces **no new persistence**: it is rebuilt at start-up from the
`genome_cache` collection the strategies already maintain (chromosome +
objectives in minimization space), recomputing descriptors deterministically.
That is also what makes restart/resume work without a second source of truth.

The fingerprint covers the candidate set, sink, mobility samples, `R_com`,
`R_inter`, region, coverage threshold, the objective list *and their senses*,
the metric transform config, the aggregator, duration, seeds, and the source
repositories.

**Exact cache** (`REUSE`) is the genome hash: an identical chromosome already
fully evaluated in the same fingerprint reuses its objectives and runs no
simulation. Only identity qualifies — structural similarity feeds the
estimator, never the cache.

Objectives whose magnitude exceeds `1e8` (infeasibility penalties, "no metrics"
sentinels) are kept as records but excluded from the training set, where they
would dwarf every real measurement.

---

## 7. Estimator

`ObjectiveEstimator` is a Protocol (`fit` / `predict`). The initial
implementation is distance-weighted k-NN over normalised descriptors:

```
w_i    = 1 / (d_i + ε)
f̂_j(x) = Σ w_i f_j(x_i) / Σ w_i
σ̂_j(x) = sqrt( Σ w_i (f_j(x_i) − f̂_j(x))² / Σ w_i )
```

An exact structural match (distance 0) short-circuits to the stored objectives
rather than blending them with distant neighbours. Fewer than `k` samples uses
what exists; an untrained estimator returns `None`, which the policy reads as
"simulate".

No neural network, no Gaussian process, no extra dependency — deliberately, so
the first version is auditable.

---

## 8. Decision policy

All comparisons happen in minimization space, derived from
`objectives[].goal`; nothing assumes latency is minimised or throughput
maximised.

```
exact cache hit                                 → REUSE
|D| < min_training_samples                      → SIMULATE  (warm-up)
no prediction available                         → SIMULATE
N(x) > novelty_threshold                        → SIMULATE  (exploration)
mean(σ̂/range) > uncertainty_threshold           → SIMULATE  (exploration)
∃y ∈ front : f(y) ≺ᵐ L(x)   and audit draw hits → SIMULATE  (audit sample)
∃y ∈ front : f(y) ≺ᵐ L(x)                       → ESTIMATE_ONLY
otherwise                                       → SIMULATE  (near-front)
```

with `L(x) = f̂(x) − κσ̂(x)` (optimistic), `U(x) = f̂(x) + κσ̂(x)`
(pessimistic), `≺ᵐ` clear dominance by at least `dominance_margin × range` in
every objective *and* strict improvement somewhere, and

```
N(x) = λ·d_φ(x) + (1 − λ)·d_H(x)
```

Audit draws use a **separate RNG stream** derived from the experiment seed, so
enabling the heuristic does not shift the GA's crossover/mutation sequence.

`SIMULATE_REDUCED` exists in the enum as the seam for future
adaptive-seed-count evaluation; it is never emitted today.

---

## 9. The two-phase generation

**Phase A — screening.** Offspring are triaged and the SIMULATE ones are
queued. ESTIMATE_ONLY individuals enter the generation with their
**conservative** value `U(x)` and `evaluation_source="estimated"`.

**Phase B — promotion.** When the screening simulations finish, a *provisional*
NSGA-III selection runs over `P_{t−1} ∪ P_t`. Any ESTIMATE_ONLY individual in
the first front or in the survivor set is `PROMOTE_TO_SIMULATION`: its
provisional value is dropped, its Individual document is reset to
`evaluation_source="simulated"`, and its simulations are queued **into the same
generation**, whose status returns to `Running`. master-node closes the
generation again when they finish, and that second terminal event drives the
definitive selection.

**The invariant.** With `require_simulated_survivors: true` (the default), any
individual still only estimated is excluded from the environmental selection.
Combined with promotion, every individual entering `P_{t+1}` carries simulated
or exact-cache objectives, so the evolutionary state is never contaminated. The
filter is skipped, with a warning, only if it would starve the selection.

An estimated individual additionally:

* never enters `genome_cache` (ground truth only);
* never becomes an estimator training sample;
* never appears in the published `pareto_front`.

---

## 10. Reproducibility and metrics

Every decision is upserted into `adaptive_evaluations`, keyed
`(experiment, generation, individual)` so a replay overwrites instead of
appending:

```json
{
  "decision": "ESTIMATE_ONLY",
  "decision_reason": "optimistic_bound_dominated",
  "predicted_objectives": [...], "uncertainty": [...],
  "optimistic_objectives": [...], "conservative_objectives": [...],
  "novelty": 0.14, "nearest_neighbor_distance": 0.12,
  "dominance_result": true, "audit_selected": false,
  "promotion_selected": false, "actual_objectives": null,
  "evaluation_source": "estimated", "descriptors": {...}
}
```

Per-generation and per-experiment cost accounting goes to `adaptive_metrics`:

```
generated_individuals  exact_cache_hits  estimated_only  initial_simulations
promotions  audit_simulations  total_actual_simulations  baseline_simulations
avoided_simulations  simulation_reduction_ratio
prediction_mae  prediction_rmse  prediction_error_per_objective
false_skip_rate  mean_uncertainty  mean_novelty
```

with

$$\text{simulation reduction ratio} = 1 - \frac{N_\text{sim adaptive}}{N_\text{sim baseline}}$$

`baseline_simulations` counts what a plain NSGA-III would have simulated in the
same generation: every new genome that was neither an exact cache hit nor
rejected by the hard-constraint penalty.

Structured logs are one line per decision:

```
[adaptive-eval] generation=4 individual=abc123def456 decision=ESTIMATE_ONLY \
    reason=optimistic_bound_dominated nearest_distance=0.1200 novelty=0.0800
```

---

## 11. Restart / resume

On `start()`:

1. `genome_cache` restores `_inserted_genomes` and `_genome_objectives_cache`
   (base behaviour, unchanged);
2. the knowledge base is rebuilt from the same entries, recomputing descriptors
   and re-feeding `RoutingKnowledge`;
3. the decision log restores which individuals are still only estimated;
4. the base resume path restores the population and the generation index.

Ground truth is written to `genome_cache` **as soon as it is measured**, not
only at the end of a generation, so an interruption between the screening and
the promotion rounds still leaves the base fully restorable. Simulations
already `Done` are never re-queued, and decisions are upserted rather than
duplicated.

---

## 12. Configuration

Two complete, launchable payloads:
[`post-nsga3-adaptive-experiment-p2.json`](../../debug/requests/post-nsga3-adaptive-experiment-p2.json) (repair-based) and
[`post-nsga3-adaptive-experiment-p2-tree.json`](../../debug/requests/post-nsga3-adaptive-experiment-p2-tree.json) (tree-encoded).

```jsonc
"algorithm": {
  "population_size": 50, "number_of_generations": 20,
  "random_seed": 42, "prob_cx": 0.8, "prob_mt": 0.2, "divisions": 6,
  "per_gene_prob": 0.1,
  "adaptive_evaluation": {
    "enabled": true,
    "min_training_samples": 50,
    "estimator": { "type": "weighted_knn", "k": 7, "epsilon": 1e-9 },
    "confidence": { "kappa": 1.96 },
    "novelty": { "descriptor_weight": 0.7, "hamming_weight": 0.3, "threshold": 0.40 },
    "uncertainty_threshold": 0.25,
    "dominance_margin": 0.02,
    "audit_probability": 0.05,
    "simulation_budget": {
      "enabled": false, "min_per_generation": 5,
      "max_per_generation": 20, "promotion_reserve": 5
    },
    "require_simulated_survivors": true
  }
},
"problem": {
  "name": "problem2_topology_aware",
  "topology_heuristic": {
    "enabled": true,
    "distance_weight": 1.0, "routing_importance_weight": 1.0,
    "structural_quality_weight": 1.0, "relay_cost_weight": 1.0,
    "tree_distance_weight": 1.0, "tree_routing_importance_weight": 0.0
  }
}
```

For the repair-free variant, swap the problem block — `topology_heuristic`
becomes irrelevant because nothing repairs:

```jsonc
"problem": {
  "name": "problem2_tree_encoded",
  "tree_encoding": {
    "backend": "two_level",   // or "parent_array"
    "mutation_moves": 2,      // PAO/CAO/grow/prune moves per mutation
    "max_relays": null        // cap on the size of the initial random tree
  }
}
```

**Choosing kappa.** Larger κ ⇒ more conservative ⇒ fewer skips. On a
three-objective landscape the 95% band (1.96) is conservative enough that
little is skipped once the population converges onto the front. Sweep it on
your own scenario with
[`experiments/adaptive-simulation/run_comparison.py`](../../experiments/adaptive-simulation/run_comparison.py)
before fixing it for a campaign.

---

## 13. Deliberately not in this version

Neural networks, Gaussian processes, reinforcement learning, changes to the
standard NSGA-III, reduced-seed evaluation, and any reuse of knowledge across
differing fingerprints. The abstractions leave room for all of them.

The two-level tree **is** implemented (section 4b) and is the default backend of
`problem2_tree_encoded`. What is not implemented is the TSP reversal bit, which
a rooted tree has no use for, and the degree-constrained NDDR extension, which
P2 does not need.
