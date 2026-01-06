# Stochastic Reachability Growth Process

We define a **stochastic reachability growth process**, rooted at the sink, with an **active frontier**, **explicit exclusion of discarded candidates**, and **controlled propagation**.
Below we formalize this process rigorously.

The implementation is available in `lib.util.random_network`, in the function `stochastic_reachability_mask`.
Here, we describe its theoretical foundation.

---

## 1. Formal Problem Modeling

Let:

* $Q = {q_0, q_1, \dots, q_{J-1}}$ be the set of candidate positions;
* $q_0$ be the **sink** (root);
* $r > 0$ be the reachability radius (`radius_of_reach`);
* The induced geometric graph:
$$
G = (Q, E), \quad (q_i, q_j) \in E \iff \lVert q_i - q_j \rVert \le r.
$$

**Algorithm objective:**

* Construct a subset $P \subseteq Q$;
* Such that the induced subgraph $G[P]$ is **connected**;
* With **stochastic and sparse** cardinality and spatial distribution.

---

## 2. Algorithm States

Each candidate $q \in Q$ is always in exactly one of the following states:

* **FREE** — not yet processed;
* **SELECTED** — included in $P$;
* **DISCARDED** — explicitly rejected.

Formally, we maintain:
$$
Q = F \cup S \cup D,
\quad\text{with}\quad
F \cap S = S \cap D = F \cap D = \emptyset.
$$

---

## 3. Auxiliary Structures

* **Frontier** $A \subseteq S$: active nodes that may still expand.

**Initialization:**

* $S = {q_0}$;
* $A = {q_0}$;
* $F = Q \setminus {q_0}$;
* $D = \emptyset$.

---

## 4. Formal Algorithm (Mathematical Description)

While $A \neq \emptyset$:

1. Select an active node $u \in A$;

2. Define the set of free reachable candidates:
$$
N(u) = {v \in F \mid \lVert u - v \rVert \le r}.
$$

3. If $N(u) = \emptyset$:

   * remove $u$ from $A$;
   * continue;

4. Sample:
$$
c \sim \mathcal{U}{1, |N(u)|};
$$

5. Randomly choose:
$$
C \subset N(u), \quad |C| = c;
$$

6. Update states:

   * $S \leftarrow S \cup C$;
   * $A \leftarrow A \cup C$;
   * $F \leftarrow F \setminus C$;

7. For the non-selected candidates:
$$
D_u = N(u) \setminus C;
$$
$$
D \leftarrow D \cup D_u;
$$
$$
F \leftarrow F \setminus D_u;
$$

8. Remove $u$ from $A$.

The algorithm terminates when:
$$
A = \emptyset
\quad\text{or}\quad
F = \emptyset.
$$

**Result:**
$$
P = S.
$$

---

## 5. Guaranteed Properties

### 5.1 Connectivity by Construction

Every node added to $S$:

* lies within the reachability radius of some node already in $S$;
* therefore, the induced subgraph is connected (a tree with possible cycles).

---

### 5.2 Avoidance of “All-Ones” Collapse

* Explicit discarding of $N(u) \setminus C$ prevents revisitation;
* Randomization of $c$ controls density;
* Nodes far from the root may never be reached.

---

### 5.3 Structural Diversity

* Two individuals with the same random seed rarely generate identical masks;
* Expansion order and sampled values of $c$ induce distinct topologies.

---

## 6. Algorithmic Interpretation (Computational View)

This algorithm can be interpreted as:

* A **stochastic BFS with variable branching**;
* A **controlled percolation process**;
* A **random connected subgraph generator for geometric graphs**.

---

## 7. Computational Complexity

Let $J = |Q|$.

* Each candidate changes state **at most once**;
* Neighborhood computation can be:

  * $O(J)$ in a naive implementation;
  * $O(\log J)$ using spatial data structures (e.g., KD-tree).

**Overall complexity:**
$$
O(J^2) \text{ (naive)} \quad\text{or}\quad O(J \log J).
$$

---

## 8. Important Critical Remark

This algorithm **does not guarantee minimum coverage** nor **any form of optimality** — and this is intentional:

* It generates **feasible initial seeds**;
* NSGA-based methods perform the multi-objective refinement;
* Heavy bias from MILP-based constructions is avoided.

---
