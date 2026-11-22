# Overview of the Problems

The five problems considered in this current version are:

0. **Pure analytical benchmark (algorithm validation)** — no WSN semantics.
1. **Continuous placement of fixed motes to cover mobile motes, with a fixed sink.**
2. **Selection of candidate positions for fixed motes to cover mobile motes, with a fixed sink.**
3. **Selection of candidate positions for fixed motes to cover sensing targets, with a fixed sink.**
4. **Route and stop planning for a mobile sink to collect data in an initially disconnected sensor network.**

Each problem is formally defined below, followed by the operators that
implement it. The definitions state the intended formulation. Where the code
does not yet enforce part of it, [Known Limitations](#known-limitations) says so.

| Problem | Adapter | Status |
|---|---|---|
| P0 | `p0_synthetic.py` | Runs in-process |
| P1 | `p1_continuous_mobility.py` | Runs |
| P2 | `p2_discrete_mobility.py` | Runs |
| P3 | `p3_target_coverage.py` | Runs |
| P4 | `p4_mobile_sink_collection.py` | Runs |

---

## Problem 0 — Pure Analytical Benchmark

**Description.**
A synthetic problem used to validate the *algorithmic* behaviour of the
multi-objective optimizers (NSGA-II / NSGA-III and variants, Random Search)
against classical test problems — **DTLZ2**, **ZDT1** and **SCH1** — fully
decoupled from the WSN topology machinery of Problems 1–4.

**Given:**

* A number of decision variables $n \ge 1$;
* A benchmark function $f : [0,1]^n \rightarrow \mathbb{R}^M$ with a Pareto
  front known in closed form (see `pylib/benchmarks.py`).

**Representation.**
The chromosome is the decision vector itself:

$$
x = (x_0, \ldots, x_{n-1}), \qquad x_i \in [0, 1].
$$

There is no MAC gene, no sink, no relays and no connectivity/coverage repair.
The genetic operators are the textbook real-coded pair — Simulated Binary
Crossover (SBX) and Polynomial Mutation — applied independently per variable
over the unit interval.

**Evaluation.**
Because the objectives are a closed-form function of $x$, the adapter exposes
`is_analytical = True` and the strategies evaluate individuals **in-process**
(`lib/strategy/analytical.py`): no `Simulation` document is created and the
master-node is never involved. The exception among the optimization strategies is `batch`,
which lacks the analytical fast-path and delegates evaluation to the
master-node's synthetic module. See
[docs/markdown/SYNTHETIC_MODE.md](../../../docs/markdown/SYNTHETIC_MODE.md)
for the full description of the two evaluation paths.

**Purpose.**
Fast, reproducible validation of selection/variation operators and convergence
metrics (HV, GD, IGD) against known Pareto fronts, without the cost of network
simulation.

---

## Problem 1 — Continuous Coverage with Mobility

**Description.**
A continuous deployment problem where fixed sensor motes must be placed freely in a region so as to maintain connectivity between mobile motes and a fixed sink at all times.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$ (in the implementation, an axis-aligned rectangle, as in every problem below);
* A fixed sink located at a known position $\sigma \in \Omega$;
* A fixed number $N$ of sensor motes to be placed arbitrarily in $\Omega$;
* A set of $M$ mobile motes following continuous, regular, and recurrent trajectories (closed loops or bounded oscillatory paths);
* A homogeneous communication model with:

  * communication radius $R_{\text{com}} > 0$;
  * interference radius $R_{\text{inter}} \ge 2 R_{\text{com}}$ (a modelling assumption; the engine does not validate it).
* A required **coverage level** $\alpha \in [0, 100]$, in percent (JSON key
  `min_coverage_percentage`, optional, default $100$).

**Goal.**
Determine a set of positions $P \subset \Omega$, with $|P| = N$, for the fixed motes such that, for all times $t \in \mathbb{R}_+$, every mobile mote can communicate with the sink through a multi-hop path whose individual hops do not exceed $R_{\text{com}}$.

**Coverage level $\alpha$.**
The trajectories are discretized by arc length, with a step of at most
$R_{\text{com}}/2$, and the requirement above is enforced on the sampled points:
at least $\alpha$ percent of them must lie within $R_{\text{com}}$ of the sink or
of some deployed mote (which the repair keeps connected to the sink). With
$\alpha = 100$ the constraint is the strict formulation; lower values relax it,
which is useful when full coverage is infeasible for the given $N$ and
$R_{\text{com}}$. Individuals below $\alpha$ are rejected by
`penalty_objectives` and never reach simulation.

**Trajectories** (P1, P2, and the sink of P4). `is_closed` means the mote goes
back from the end of its path to the start: when the segments do not already
end where they begin, that return is a straight segment, sampled for the
coverage and travelled in the simulation (`pylib/trajectory.py`). A round trip
goes back along the path. An open path that is not a round trip starts over at
its beginning. Each mote repeats its own path at its own period for the whole
simulation: the positions file spans the least common multiple of the periods
when that is shorter than the simulation, and the whole simulation otherwise.

---

## Problem 2 — Discrete Coverage with Mobility

**Description.**
A discrete variant of Problem 1, where fixed motes can only be placed at predefined candidate locations.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$;
* A fixed sink at position $\sigma \in \Omega$;
* A finite set of candidate positions $Q = \{q_1, \dots, q_J\} \subset \Omega$;
* A set of $M$ mobile motes with continuous and recurrent trajectories;
* Homogeneous communication and interference radii $R_{\text{com}}$ and $R_{\text{inter}}$;
* A required **coverage level** $\alpha \in [0, 100]$, in percent (JSON key
  `min_coverage_percentage`, optional, default $100$), with the same
  discretization as in Problem 1: a sample is covered when it lies within
  $R_{\text{com}}$ of the sink or of an active candidate.

**Goal.**
Determine the **smallest subset** $P \subset Q$ of fixed mote positions such that, for all times $t \in \mathbb{R}_+$, each mobile mote can communicate with the sink via a multi-hop path with hop lengths bounded by $R_{\text{com}}$.

The cardinality $|P|$ is not an objective by itself: the engine optimizes only
the objectives the experiment declares. Sparsity comes from the operators and
from whatever those objectives make of the node count.

---

## Problem 3 — Sensing Coverage Problem

**Description.**
A static sensing coverage problem with explicit sensing targets and connectivity constraints.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$;
* A fixed sink at position $\sigma \in \Omega$;
* A finite set of candidate sensor positions $Q = \{q_1, \dots, q_J\}$;
* A finite set of sensing targets $\Xi = \{\xi_1, \dots, \xi_H\}$;
* Radii:

  * sensing radius $R_{\text{cov}} > 0$;
  * communication radius $R_{\text{com}} > 0$;
  * interference radius $R_{\text{inter}} \ge 2 R_{\text{com}}$;
* Integer parameters:

  * $k$: required coverage degree for each target (JSON key `k_required`);
  * $g$: required minimum communication degree for each deployed mote (JSON
    key `g_required`, optional, default 1; the sink counts as a neighbour).

**Goal.**
Determine the **smallest subset** $P \subset Q$ such that:

1. Each target $\xi_h$ is covered by at least $k$ fixed motes;
2. Each deployed mote has at least $g$ neighbors within communication range;
3. Every deployed mote reaches the sink over the communication graph.

With $g = 1$ the second condition follows from the third.

**Penalty.** An individual that violates a condition is not simulated. With
the three deficits in $[0, 1]$ (share of motes cut off from the sink, missing
covers over $k \cdot |\Xi|$, missing neighbours over $g \cdot |P|$) and $d$
their mean, every objective is $10^9 (1 + d)$, as in P1 and P2.

---

## Optimization Objectives (Problems 1–3)

The formulation evaluates Problems 1–3 according to the following criteria:

* Minimize average energy consumption;
* Minimize average end-to-end latency;
* Maximize network throughput.

These objectives define a **multi-objective optimization setting**. They are not
built into the adapters: each experiment lists its objectives in
`parameters.objectives`, by metric name. The wizard's defaults (median latency,
mean power, reports delivered) and what each metric measures are in
[FIRMWARE_METRICS.md](../../../docs/markdown/FIRMWARE_METRICS.md).

---

## Problem 4 — Data Collection with a Mobile Sink

**Description.**
A data collection and mobility planning problem where a mobile sink gathers data from a set of fixed motes in an initially disconnected network.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$;
* A base location $b \in \Omega$, where the sink can recharge and offload data;
* A set of fixed motes with known positions and continuous data generation rates;
* Finite energy budgets and finite buffer capacities at each mote;
* A finite set of candidate sink stopping locations $L$;
* A mobility graph $(L, A)$ defining feasible sink movements;
* Communication and interference radii $R_{\text{com}}$ and $R_{\text{inter}}$.

**Goal.**
Determine:

* The subset of stopping locations actually visited by the sink;
* The visiting order (route) over the mobility graph;
* The dwell time at each stop,

such that all motes can successfully deliver their data to the sink through multi-hop communication paths, without exceeding their energy constraints.

**In the implementation.** The sink's route repeats (its path is closed). A
stop collects from every mote in a connected part of the network (multi-hop,
within $R_{\text{com}}$) that holds a mote in its `visibleNodes`, and only
while the sink dwells there. An individual is infeasible, and penalised as in
P1–P3, when a mote is never collected or when a buffer overflows: with
$\delta$ the data rate, $W$ the buffer and $g_i$ the longest time between two
collections of mote $i$, over the round and into the next,
$\delta \, g_i > W$. With $u$ the share of motes never collected and $o$ the
mean over the collected motes of $\min(1, (\delta g_i - W)/W)$, every
objective is $10^9 (1 + (u + o)/2)$.

Energy is not a constraint of the implementation: it is measured by the
simulation (the firmware's energy metrics), and `initial_energy` is validated
and stored with the problem. The sink is Cooja's mote 1, the RPL root, although
it is mobile.

Positions are `[x, y]`; sojourn `i` must have id `i` (the route holds indices,
the adjacency ids), and sojourn 0 is the base.

---

## Scope

These problem definitions serve as the conceptual and mathematical foundation for:

* algorithmic design;
* simulation-based evaluation;
* multi-objective optimization of WSN architectures.

The definitions are intentionally formulation-centric and independent of specific protocols, hardware platforms, or solvers. The sections below are about this implementation.

---

# Search Operators

Strategies (NSGA-II/III, random search) are in
[lib/strategy/README.md](../strategy/README.md). When an experiment has a
single source repository, the MAC gene described below is pinned to its
protocol after every operator ([mac_gene.py](mac_gene.py)).

## P1. Genetic Operators for Problem 1

This section describes the genetic operators used to solve **Problem P1 (Continuous Coverage with Mobility)**.

The chromosome representation consists of:

* a **continuous set of relay positions**  
  $$
  P = \{p_1, \dots, p_N\} \subset \Omega \subset \mathbb{R}^2,
  $$
  where each relay position is a pair $(x_i, y_i)$;
* a **MAC protocol gene**, encoded as a binary value (e.g., CSMA / TSCH).

All operators are explicitly designed to **preserve connectivity to the sink** of the induced geometric graph under the communication radius constraint.


### P1.1. Random Individual Generation

Initial individuals are generated using a [**connectivity-constrained continuous network generator**](./docs/Continuous%20Network%20Generation%20via%20Connectivity-Constrained%20Spatial%20Expansion.md), which incrementally places relay nodes inside the deployment region while preserving global connectivity.

Instead of sampling relay positions independently, the algorithm constructs a **graph connected to the sink by design**. The result then goes through the same connectivity and coverage repair as the variation operators; it can still fall below $\alpha$ and be penalized.

#### Operator description

For each individual:

1. The deployment region $\Omega$ and communication radius $R$ define a geometric graph model.
2. A set of $N$ relay positions is generated using a **spatial growth process** seeded at the sink that:
   * preserves connectivity at every insertion step;
   * promotes spatial dispersion of relays.
3. Connectivity and, if `apply_coverage_repair`, coverage repair are applied.
4. The MAC protocol gene is initialized uniformly at random.

This initialization strategy provides **connected and well-distributed relay topologies**, suitable as starting points for multi-objective optimization.



### P1.2. Crossover Operator

Crossover operates on the continuous relay positions and on the MAC protocol gene independently.

Two crossover strategies are supported (`crossover_method`): `sbx_with_radial_translate` and `rand_network` (alias `rand`).



#### (a) SBX with Radial Connectivity Enforcement

This is the default crossover method.

##### Operator description

Given two parent chromosomes with relay sets $P^{(1)}$ and $P^{(2)}$:

1. Relays of the two parents are **paired by list index**, and **Simulated Binary Crossover (SBX)** is applied independently to each coordinate $(x, y)$ of a pair, each with its own child swap:
   * crossover is bounded by the region $\Omega$;
   * distribution index $\eta_{\text{cx}}$ controls offspring spread.
2. The resulting offspring relay sets may violate connectivity.
3. A **radial translation operator** is applied to each offspring:
   * each component disconnected from the sink is moved rigidly toward it, in steps, until it connects;
   * positions are clipped to $\Omega$; if connectivity still fails, the relay network is regenerated.
4. If `apply_coverage_repair`, a greedy coverage repair moves up to `repair_coverage_budget` relays (default 2).
5. The MAC protocol gene is inherited by uniform random selection between parents.

This operator allows **continuous variation of relay positions** while restoring connectivity.

Relay order has no meaning, so pairing by index combines unrelated relays, and
the independent swaps can give a child the $x$ of one relay and the $y$ of
another. See decision D3 of
[CAMPAIGN_V2.md](../../../docs/markdown/CAMPAIGN_V2.md#3-decisions-before-launch).



#### (b) Random Network Crossover

As an alternative strategy, crossover may ignore parental geometry entirely:

1. Two new connected relay sets are generated using the continuous network generator and repaired as above.
2. MAC genes are inherited uniformly from the parents.

This strategy introduces **strong exploration pressure** and is useful to prevent premature convergence.



### P1.3. Mutation Operator

Mutation is applied independently to relay coordinates and to the MAC protocol gene.



#### Relay position mutation

For each relay position $(x_i, y_i)$:

1. **Polynomial mutation** is applied independently to $x_i$ and $y_i$ with per-gene probability $p_{\text{gene}}$;
2. Mutated coordinates are clipped to remain inside $\Omega$;
3. The **repair pipeline of the crossover** (radial connectivity, then optional coverage) is applied.

The mutation strength is controlled by the distribution index $\eta_{\text{mt}}$.



#### MAC protocol mutation

* With probability $p_{\text{gene}}$, the MAC protocol gene is flipped (0 ↔ 1).



### P1.4. Design Rationale

The design of the genetic operators for Problem P1 follows the principle of **feasibility preservation in continuous space**:

* **Connectivity by construction:** random generation grows a graph connected to the sink;
* **Coverage by repair and penalty:** what the repair cannot cover is penalized and not simulated;
* **Continuous variation:** SBX and polynomial mutation allow smooth exploration of $\Omega$;
* **Explicit connectivity enforcement:** crossover and mutation are followed by geometric translation instead of combinatorial repair;
* **Exploration–exploitation balance:** random-network crossover provides controlled diversity injection.

These operators are particularly suited for continuous deployment problems with strict connectivity requirements and mobile coverage objectives.



### P1.5. Summary of Operator Flow

```

Random generation:
continuous_network_gen → connected relay set
→ connectivity repair → coverage repair (if apply_coverage_repair)

Crossover:
SBX on relay coordinates
→ radial connectivity enforcement
→ coverage repair (if apply_coverage_repair)
uniform MAC inheritance
(or full random network regeneration)

Mutation:
polynomial mutation on relay coordinates
→ radial connectivity enforcement
→ coverage repair (if apply_coverage_repair)
bit-flip MAC mutation

```

---


## P2. Genetic Operators for Problem 2

This section describes the genetic operators used to solve **Problem P2 (Discrete Coverage with Mobility)**.
The chromosome representation consists of:

* a **binary mask** over candidate positions $Q$, indicating which fixed motes are activated;
* a **MAC protocol gene**, encoded as a binary value (e.g., CSMA / TSCH).

The operators are designed to **preserve connectivity by construction**, so that individuals passed to evaluation are **connected to the sink**. Coverage is handled by repair and penalty (P2.4).



### P2.1. Random Individual Generation

Initial individuals are generated using a [**stochastic rooted growth process**](./docs/Rooted%20stochastic%20expansion%20over%20the%20reachability%20graph.md) over the reachability graph.

Instead of sampling a random binary mask, the algorithm constructs a **connected and sparse subgraph rooted at the sink**, which directly satisfies the connectivity constraint of Problem P2.

#### Operator description

For each individual:

1. The reachability graph is implicitly defined over the candidate positions $Q$ using the communication radius $R$.
2. A stochastic expansion process rooted at the sink $S$ generates a connected subset $P \subseteq Q$.
3. The resulting binary mask is guaranteed to be connected to the sink.
4. Coverage repair is applied (P2.4) and the MAC protocol gene is initialized uniformly at random.
5. A genome already in the population is drawn again, up to 20 times.

This strategy biases the initial population toward **small, connected solutions**, which is consistent with the objective of minimizing $|P|$.



### P2.2. Crossover Operator

Crossover is performed independently on each gene type:

* the binary mask is recombined using **uniform crossover**;
* the MAC protocol gene is inherited uniformly from either parent.

Since uniform crossover may destroy connectivity, a **connectivity repair step** is always applied.

#### Operator description

Given two parent chromosomes:

1. Apply uniform crossover to the binary masks, producing two offspring masks.
2. Apply a [**Connectivity Repair via Rooted Growth**](./docs/Connectivity%20Repair%20via%20Rooted%20Growth.md) to each offspring mask:

   * each disconnected component is joined by a shortest path from the active nodes within reach of the sink;
   * only the nodes of those paths are activated (a heuristic, not a minimum).
3. If repair fails (e.g., unreachable component), the offspring mask is reverted to the corresponding parent mask.
4. The MAC protocol gene is inherited by uniform random selection between parents.

This keeps **offspring connected**, while still allowing significant structural variation.



### P2.3. Mutation Operator

Mutation is applied separately to the mask and to the MAC protocol gene.

#### Mask mutation

1. The binary mask undergoes **bit-flip mutation** with per-gene probability $p_{\text{bit}}$.
2. The mutated mask is then passed through the [**Connectivity Repair via Rooted Growth**](./docs/Connectivity%20Repair%20via%20Rooted%20Growth.md).
3. If repair fails, the original mask is restored.

Mutation therefore does not break connectivity. Coverage can still fall below $\alpha$ and be penalized.

#### MAC protocol mutation

* With probability $p_{\text{bit}}$, the MAC protocol gene is flipped (0 ↔ 1). 



### P2.4. Coverage Repair (optional)

In addition to connectivity, Problem P2 carries a **trajectory coverage constraint**: the sink and the active candidates must cover at least $\alpha$ (`min_coverage_percentage`, a *problem* parameter — not a GA knob) of the sampled trajectory points, otherwise the individual is penalized by `penalty_objectives` and skips simulation.

When the algorithm parameter `apply_coverage_repair` is enabled (default `true`), a **greedy set-cover repair** is applied after every connectivity repair (random generation, crossover, and mutation):

1. The coverage bitsets of the candidates and of the sink are precomputed once per problem instance (`build_candidate_cover_bits`), transposing the bitset coverage matrix. The sink is always active and is not a gene, so what it covers needs no relay.
2. While the coverage is below the threshold and at most `repair_coverage_budget` activations (default 8) have been spent, the inactive candidate covering the most still-uncovered points is activated (ties break on the lowest index — fully deterministic, activation-only, never deactivates genes).
3. If any candidate was activated, [**Connectivity Repair via Rooted Growth**](./docs/Connectivity%20Repair%20via%20Rooted%20Growth.md) reconnects the mask to the sink; if that fails, the pre-repair (connected) mask is kept.

Individuals the repair cannot fix — or any individual when `apply_coverage_repair` is `false` — remain subject to the coverage penalty, which acts as the safety net. The same `apply_coverage_repair` flag also gates the greedy coverage repair stage of Problem P1 (`_greedy_coverage_repair`); P1's connectivity repair always runs regardless.



### P2.5. Design Rationale

The combination of **growth-based initialization** and **repair-based variation** yields several advantages:

* **Connectivity preservation:** individuals are connected to the sink at evaluation time;
* **Search space reduction:** disconnected masks are never evaluated;
* **Controlled sparsity:** both initialization and repair favor minimal activations;
* **High diversity:** uniform crossover and stochastic repair induce multiple valid topologies;
* **Low computational overhead:** repairs are graph-based and avoid MILP formulations.

Together, these operators define a robust evolutionary pipeline for Problem P2, well suited for multi-objective optimization under strict connectivity constraints.



### P2.6. Summary of Operator Flow

```
Random generation:
    stochastic_reachability_mask → feasible connected individual
        → coverage repair (if apply_coverage_repair)

Crossover:
    uniform mask crossover
        → connectivity repair
        → coverage repair (if apply_coverage_repair)
    uniform MAC inheritance

Mutation:
    bit-flip mask mutation
        → connectivity repair
        → coverage repair (if apply_coverage_repair)
    bit-flip MAC mutation
```

---

## P3. Genetic Operators for Problem 3

The chromosome is the same as in P2: a binary mask over $Q$ and a MAC gene.

| Operator | Pipeline |
|---|---|
| Random generation | The rooted growth of P2.1, redrawn up to 20 times until the mask is feasible (the three conditions, the sink included) and new in the population. After that the last draw is accepted, and penalised if infeasible |
| Crossover | Uniform mask crossover → $k$-coverage repair → connectivity repair; uniform MAC inheritance |
| Mutation | Bit-flip mask mutation → $k$-coverage repair → connectivity repair; bit-flip MAC |

The $k$-coverage repair activates the candidates nearest to each deficient
target; the connectivity repair then activates paths to the sink. Both only
activate, so the second keeps what the first achieved. No operator repairs a
degree below $g$: the penalty does.

---

## P4. Genetic Operators for Problem 4

The chromosome holds a route (indices into `sojourns`, where index 0 is the
base), one dwell time per route position within `tau_bounds`, and a MAC gene.
Sojourn ids have to equal their indices.

| Operator | Pipeline |
|---|---|
| Random generation | Random walk over the adjacency, up to `max_route_length`, closed at the base → route repair; uniform dwell times |
| Crossover | Single-cut route splice → route repair; blend crossover of the dwell times; uniform MAC inheritance |
| Mutation | With probability `per_gene_prob`, one internal stop is replaced → route repair; with probability `pm_tau`, Gaussian noise (`sigma_tau`) on the whole dwell vector; bit-flip MAC |

The route repair keeps the longest valid prefix from the base and returns to
the base by the shortest walk over the adjacency, dropping the last stops when
the route would exceed `max_route_length` stops. A route that cannot leave the
base is `[0, 0]`.

---

# GA Parameter Consumption per Problem

Each adapter declares the problem-specific GA keys it reads in its
`CONSUMED_GA_KEYS` class attribute; `resolve.build_adapter` logs a warning for
any key sent by the launcher that the problem ignores. Strategy-level keys
(`population_size`, `number_of_generations`, `random_seed`, `divisions`,
`prob_cx`, `prob_mt`, `selection_method`) are handled by `lib/strategy/*`
for every problem: random search reads only the first three, `divisions` is
read by NSGA-III only, and `selection_method` is accepted but fixed, since
NSGA-II and NSGA-III always use tournament selection.

| Key | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| `per_gene_prob` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `crossover_method` | — | ✓ (`sbx_with_radial_translate` default, `rand_network`) | — | — | — |
| `eta_cx`, `eta_mt` | ✓ | ✓ | — | — | — |
| `apply_coverage_repair`, `repair_coverage_budget` | — | ✓ | ✓ | — | — |
| `pm_tau`, `sigma_tau` | — | — | — | — | ✓ |
| `repair_coverage_candidates`, `repair_coverage_relay_candidates`, `connectivity_repair_step_ratio`, `connectivity_repair_retries` | — | ✓ | — | — | — |

Defaults when a key is absent from a non-empty `algorithm` block:

| Key | P0 | P1 | P2 | P3 | P4 |
|---|---|---|---|---|---|
| `per_gene_prob` | $1/n$ | $1/(2N)$ | 0.1 | 0.1 | 0.1 |
| `eta_cx` | 20 | 20 | — | — | — |
| `eta_mt` | 20 | 25 | — | — | — |
| `apply_coverage_repair` | — | `true` | `true` | — | — |
| `repair_coverage_budget` | — | 2 | 8 | — | — |
| `pm_tau`, `sigma_tau` | — | — | — | — | 0.5, 5 |

P1's other four repair keys default to 4, 3, 0.1 and 2, in the order of the
table. The GUI does not send them, and P1 also accepts the older spellings
`coverage_repair_*` of the first two and of the budget.

Crossover/mutation operators are **fixed by design** everywhere except P1's
crossover: feasibility-preserving pipelines (connectivity/coverage repair)
are part of each problem's formulation, not tunable knobs. The GUI mirrors
this table in `gui/simlab/src/lib/problemCapabilities.ts` — keep both in
sync when an adapter starts consuming a new key.

Note that $\alpha$ (`min_coverage_percentage`) is **not** in this table: it
belongs to the problem instance, not to the algorithm, and therefore travels
in `parameters.problem` — edited in the GUI's problem editor, summarized in the
launch wizard, and shown on the problem visualization. `apply_coverage_repair`
and `repair_coverage_budget` are the GA-side knobs that decide *how hard* the
operators try to reach $\alpha$; $\alpha$ itself decides *what* feasible means.

---

# Known Limitations

Where the code departs from the formulations above.

**Problem 3**

- A degree below $g$ is penalised, not repaired.


---