# Connectivity Repair via Rooted Growth

The idea behind this repair operator is **conceptually precise and well aligned with Problem P2**.

In practice, it defines a **rooted growth repair operator** that transforms **any invalid binary mask** (after crossover or mutation) into a **valid mask globally connected to the sink**, using a **heuristically minimal number of additional activations** (i.e., without relying on MILP).

This repair is deterministic with respect to the reachability graph and is designed to be lightweight, safe, and compatible with evolutionary pipelines.

The reference implementation is provided in the function `repair_connectivity_to_sink`.

---

## 1. Formal Repair Problem Definition

### Input

Let:

* $Q = {q_0, q_1, \dots, q_{J-1}}$ be the set of candidate positions;
* $q_s \in Q$ be the sink (root);
* $r > 0$ be the reachability radius;
* $x \in {0,1}^J$ be a (possibly invalid) binary mask.

Define the induced subgraph:
$$
G[x] = G\big({q_i \mid x_i = 1}\big).
$$

In general, $G[x]$ may consist of **multiple connected components**.

---

### Repair Objective

Find a (heuristically) minimal set of additional activations:
$$
A \subseteq { i \mid x_i = 0 }
$$
such that:
$$
G[x \cup A] \text{ is connected to the sink } q_s.
$$

---

## 2. Meaning of “Minimal Activations” (Without MILP)

Without an exact solver, *minimality* is interpreted heuristically as:

* **Minimal number of intermediate nodes added per connection**;
* **Shortest topological distance (in hops)** in the reachability graph;
* **Preference for short and reusable paths**.

Operationally:

> Each disconnected component is connected to the sink component through a **shortest path in the candidate reachability graph**, activating only the nodes on that path.

This corresponds to an **approximate Steiner Tree construction on a discrete geometric graph**, which is fully acceptable as a repair heuristic.

---

## 3. Repair Algorithm (Formal Description)

### Step 1 — Build the Full Reachability Graph

Construct the graph:
$$
G = (Q, E), \quad (i,j) \in E \iff \lVert q_i - q_j \rVert \le r.
$$

---

### Step 2 — Identify Components of the Current Mask

* Consider only vertices with $x_i = 1$;
* Compute the connected components:
  $$
  C_0, C_1, \dots, C_k;
  $$
* Identify the component containing the sink, denoted $C_{\text{sink}}$.

If $k = 0$, the mask is already valid and the algorithm terminates.

---

### Step 3 — Iterative Rooted Repair

While there exists a component $C \neq C_{\text{sink}}$:

1. Select a disconnected component $C$;
2. Compute the **shortest path in the full graph $G$** between:

   * any node $u \in C_{\text{sink}}$,
   * any node $v \in C$;
3. Activate all intermediate nodes along this path;
4. Update:
   $$
   C_{\text{sink}} \leftarrow C_{\text{sink}} \cup C \cup \text{path}.
   $$

Repeat until all components are merged into $C_{\text{sink}}$.

---
