# Overview of the Problems

The four problems considered in this current version are:

1. **Continuous placement of fixed motes to cover mobile motes, with a fixed sink.**
2. **Selection of candidate positions for fixed motes to cover mobile motes, with a fixed sink.**
3. **Selection of candidate positions for fixed motes to cover sensing targets, with a fixed sink.**
4. **Route and stop planning for a mobile sink to collect data in an initially disconnected sensor network.**

Each problem is formally defined below.

---

## Problem 1 — Continuous Coverage with Mobility

**Description.**
A continuous deployment problem where fixed sensor motes must be placed freely in a region so as to maintain connectivity between mobile motes and a fixed sink at all times.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$;
* A fixed sink located at a known position $\sigma \in \Omega$;
* A fixed number $N$ of sensor motes to be placed arbitrarily in $\Omega$;
* A set of $M$ mobile motes following continuous, regular, and recurrent trajectories (closed loops or bounded oscillatory paths);
* A homogeneous communication model with:

  * communication radius $R_{\text{com}} > 0$;
  * interference radius $R_{\text{inter}} \ge 2 R_{\text{com}}$.

**Goal.**
Determine a set of positions $P \subset \Omega$, with $|P| = N$, for the fixed motes such that, for all times $t \in \mathbb{R}*+$, every mobile mote can communicate with the sink through a multi-hop path whose individual hops do not exceed $R*{\text{com}}$.

---

## Problem 2 — Discrete Coverage with Mobility

**Description.**
A discrete variant of Problem 1, where fixed motes can only be placed at predefined candidate locations.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$;
* A fixed sink at position $\sigma \in \Omega$;
* A finite set of candidate positions $Q = {q_1, \dots, q_J} \subset \Omega$;
* A set of $M$ mobile motes with continuous and recurrent trajectories;
* Homogeneous communication and interference radii $R_{\text{com}}) and (R_{\text{inter}}$.

**Goal.**
Determine the **smallest subset** $P \subset Q$ of fixed mote positions such that, for all times $t \in \mathbb{R}*+$, each mobile mote can communicate with the sink via a multi-hop path with hop lengths bounded by $R*{\text{com}}$.

---

## Problem 3 — Sensing Coverage Problem

**Description.**
A static sensing coverage problem with explicit sensing targets and connectivity constraints.

**Given:**

* A compact, connected region $\Omega \subset \mathbb{R}^2$;
* A fixed sink at position $\sigma \in \Omega$;
* A finite set of candidate sensor positions $Q = {q_1, \dots, q_J}$;
* A finite set of sensing targets $\Xi = {\xi_1, \dots, \xi_H}$;
* Radii:

  * sensing radius $R_{\text{cov}} > 0$;
  * communication radius $R_{\text{com}} > 0$;
  * interference radius $R_{\text{inter}} \ge 2 R_{\text{com}}$;
* Integer parameters:

  * $k$: required coverage degree for each target;
  * $g$: required minimum communication degree for each deployed mote.

**Goal.**
Determine the **smallest subset** $P \subset Q$ such that:

1. Each target $\xi_h$ is covered by at least $k$ fixed motes;
2. Each deployed mote has at least $g$ neighbors within communication range.

---

## Optimization Objectives (Problems 1–3)

In addition to ensuring feasibility and connectivity, Problems 1–3 are evaluated and optimized according to the following criteria:

* Minimize average energy consumption;
* Minimize average end-to-end latency;
* Maximize network throughput.

These objectives define a **multi-objective optimization setting**.

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

---

## Scope

These problem definitions serve as the conceptual and mathematical foundation for:

* algorithmic design;
* simulation-based evaluation;
* multi-objective optimization of WSN architectures.

They are intentionally formulation-centric and independent of specific protocols, hardware platforms, or solvers.

---

---

# Search Strategies

## P1. Genetic Operators for Problem 1

This section describes the genetic operators used to solve **Problem P1 (Continuous Coverage with Mobility)**.

The chromosome representation consists of:

* a **continuous set of relay positions**  
  $$
  P = \{p_1, \dots, p_N\} \subset \Omega \subset \mathbb{R}^2,
  $$
  where each relay position is a pair $(x_i, y_i)$;
* a **MAC protocol gene**, encoded as a binary value (e.g., CSMA / TSCH).

All operators are explicitly designed to **preserve global connectivity** of the induced geometric graph under the communication radius constraint.


### P1.1. Random Individual Generation

Initial individuals are generated using a [**connectivity-constrained continuous network generator**](./docs/Continuous%20Network%20Generation%20via%20Connectivity-Constrained%20Spatial%20Expansion.md), which incrementally places relay nodes inside the deployment region while preserving global connectivity.

Instead of sampling relay positions independently, the algorithm constructs a **connected geometric graph by design**, producing feasible individuals without the need for repair.

#### Operator description

For each individual:

1. The deployment region $\Omega$ and communication radius $R$ define a geometric graph model.
2. A connected set of $N$ relay positions is generated using a **spatial growth process** that:
   * preserves connectivity at every insertion step;
   * promotes spatial dispersion of relays.
3. The MAC protocol gene is initialized uniformly at random.

This initialization strategy provides **feasible and well-distributed relay topologies**, suitable as starting points for multi-objective optimization.



### P1.2. Crossover Operator

Crossover operates on the continuous relay positions and on the MAC protocol gene independently.

Two crossover strategies are supported.



#### (a) SBX with Radial Connectivity Enforcement

This is the default crossover method.

##### Operator description

Given two parent chromosomes with relay sets $P^{(1)}$ and $P^{(2)}$:

1. **Simulated Binary Crossover (SBX)** is applied independently to each relay coordinate $(x, y)$:
   * crossover is bounded by the region $\Omega$;
   * distribution index $\eta_{\text{cx}}$ controls offspring spread.
2. The resulting offspring relay sets may violate connectivity.
3. A **radial contraction operator** is applied to each offspring:
   * relay positions are iteratively contracted toward the network core;
   * contraction continues until the induced graph is connected.
4. The MAC protocol gene is inherited by uniform random selection between parents.

This operator allows **continuous variation of relay positions** while explicitly restoring feasibility.



#### (b) Random Network Crossover

As an alternative strategy, crossover may ignore parental geometry entirely:

1. Two new connected relay sets are generated using the continuous network generator.
2. MAC genes are inherited uniformly from the parents.

This strategy introduces **strong exploration pressure** and is useful to prevent premature convergence.



### P1.3. Mutation Operator

Mutation is applied independently to relay coordinates and to the MAC protocol gene.



#### Relay position mutation

For each relay position $(x_i, y_i)$:

1. **Polynomial mutation** is applied independently to $x_i$ and $y_i$ with per-gene probability $p_{\text{gene}}$;
2. Mutated coordinates are clipped to remain inside $\Omega$;
3. A **radial connectivity enforcement step** is applied to ensure that the induced graph remains connected.

The mutation strength is controlled by the distribution index $\eta_{\text{mt}}$.



#### MAC protocol mutation

* With probability $p_{\text{gene}}$, the MAC protocol gene is flipped (0 ↔ 1).



### P1.4. Design Rationale

The design of the genetic operators for Problem P1 follows the principle of **feasibility preservation in continuous space**:

* **Connectivity by construction:** random generation always produces connected relay graphs;
* **Repair-free initialization:** no invalid individuals are generated at population creation;
* **Continuous variation:** SBX and polynomial mutation allow smooth exploration of $\Omega$;
* **Explicit connectivity enforcement:** crossover and mutation are followed by geometric contraction instead of combinatorial repair;
* **Exploration–exploitation balance:** random-network crossover provides controlled diversity injection.

These operators are particularly suited for continuous deployment problems with strict connectivity requirements and mobile coverage objectives.



### P1.5. Summary of Operator Flow

```

Random generation:
continuous_network_gen → connected relay set

Crossover:
SBX on relay coordinates
→ radial connectivity enforcement
uniform MAC inheritance
(or full random network regeneration)

Mutation:
polynomial mutation on relay coordinates
→ radial connectivity enforcement
bit-flip MAC mutation

```

---


## P2. Genetic Operators for Problem 2

This section describes the genetic operators used to solve **Problem P2 (Discrete Coverage with Mobility)**.
The chromosome representation consists of:

* a **binary mask** over candidate positions $Q$, indicating which fixed motes are activated;
* a **MAC protocol gene**, encoded as a binary value (e.g., CSMA / TSCH).

The operators are designed to **preserve feasibility by construction**, ensuring that every individual passed to evaluation remains **globally connected to the sink**.



### P2.1. Random Individual Generation

Initial individuals are generated using a [**stochastic rooted growth process**](./docs/Rooted%20stochastic%20expansion%20over%20the%20reachability%20graph.md) over the reachability graph.

Instead of sampling a random binary mask, the algorithm constructs a **connected and sparse subgraph rooted at the sink**, which directly satisfies the primary feasibility constraint of Problem P2.

#### Operator description

For each individual:

1. The reachability graph is implicitly defined over the candidate positions $Q$ using the communication radius $R$.
2. A stochastic expansion process rooted at the sink $S$ generates a connected subset $P \subseteq Q$.
3. The resulting binary mask is guaranteed to be connected to the sink.
4. The MAC protocol gene is initialized uniformly at random.

This strategy biases the initial population toward **small and feasible solutions**, which is consistent with the objective of minimizing $|P|$.



### P2.2. Crossover Operator

Crossover is performed independently on each gene type:

* the binary mask is recombined using **uniform crossover**;
* the MAC protocol gene is inherited uniformly from either parent.

Since uniform crossover may destroy connectivity, a **connectivity repair step** is always applied.

#### Operator description

Given two parent chromosomes:

1. Apply uniform crossover to the binary masks, producing two offspring masks.
2. Apply a [**Connectivity Repair via Rooted Growth**](./docs/Connectivity%20Repair%20via%20Rooted%20Growth.md) to each offspring mask:

   * all disconnected components are connected to the sink using shortest paths in the reachability graph;
   * only the strictly necessary nodes are activated.
3. If repair fails (e.g., unreachable component), the offspring mask is reverted to the corresponding parent mask.
4. The MAC protocol gene is inherited by uniform random selection between parents.

This ensures that **all offspring remain feasible**, while still allowing significant structural variation. 



### P2.3. Mutation Operator

Mutation is applied separately to the mask and to the MAC protocol gene.

#### Mask mutation

1. The binary mask undergoes **bit-flip mutation** with per-gene probability $p_{\text{bit}}$.
2. The mutated mask is then passed through the [**Connectivity Repair via Rooted Growth**](./docs/Connectivity%20Repair%20via%20Rooted%20Growth.md).
3. If repair fails, the original mask is restored.

This guarantees that mutation never produces infeasible individuals.

#### MAC protocol mutation

* With probability $p_{\text{bit}}$, the MAC protocol gene is flipped (0 ↔ 1). 



### P2.4. Design Rationale

The combination of **growth-based initialization** and **repair-based variation** yields several advantages:

* **Feasibility preservation:** all individuals are connected to the sink at evaluation time;
* **Search space reduction:** disconnected masks are never evaluated;
* **Controlled sparsity:** both initialization and repair favor minimal activations;
* **High diversity:** uniform crossover and stochastic repair induce multiple valid topologies;
* **Low computational overhead:** repairs are graph-based and avoid MILP formulations.

Together, these operators define a robust evolutionary pipeline for Problem P2, well suited for multi-objective optimization under strict connectivity constraints.



### P2.5. Summary of Operator Flow

```
Random generation:
    stochastic_reachability_mask → feasible connected individual

Crossover:
    uniform mask crossover
        → connectivity repair
    uniform MAC inheritance

Mutation:
    bit-flip mask mutation
        → connectivity repair
    bit-flip MAC mutation
```

---