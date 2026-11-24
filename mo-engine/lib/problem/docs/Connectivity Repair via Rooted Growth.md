# Connectivity Repair via Rooted Growth

The idea behind this repair operator is **conceptually precise and well aligned with Problem P2**.

In practice, it defines a **rooted growth repair operator** that transforms a disconnected binary mask (after crossover or mutation) into a **mask globally connected to the sink**, whenever the reachability graph allows it, using a **heuristically minimal number of additional activations** (i.e., without relying on MILP).

This repair is deterministic with respect to the reachability graph and is designed to be lightweight, safe, and compatible with evolutionary pipelines.

The reference implementation is the function `repair_connectivity_to_sink` in `lib.util.connectivity`. It returns an error flag with the mask; the callers keep the parent mask when it fails.

---

## 1. Formal Repair Problem Definition

### Input

Let:

* $Q = \{q_0, q_1, \dots, q_{J-1}\}$ be the set of candidate positions;
* $\sigma$ be the sink (root), a position of its own, not an element of $Q$;
* $r > 0$ be the reachability radius;
* $x \in \{0,1\}^J$ be a (possibly invalid) binary mask.

Define the induced subgraph:
$$
G[x] = G\big(\{\sigma\} \cup \{q_i \mid x_i = 1\}\big).
$$

In general, $G[x]$ may consist of **multiple connected components**.

---

### Repair Objective

Find a (heuristically) minimal set of additional activations:
$$
A \subseteq \{ i \mid x_i = 0 \}
$$
such that:
$$
G[x \cup A] \text{ is connected to the sink } \sigma.
$$

---

## 2. Meaning of “Minimal Activations” (Without MILP)

Without an exact solver, *minimality* is interpreted heuristically as:

* **Minimal number of intermediate nodes added per connection**;
* **Shortest topological distance (in hops)** in the reachability graph;
* **Preference for short and reusable paths**.

Operationally:

> Each disconnected component is connected through a **shortest path in the candidate reachability graph, starting at the active nodes within reach of the sink**, activating only the nodes on that path.

This is a heuristic in the spirit of an **approximate Steiner Tree construction on a discrete geometric graph**. It is not minimum: a path may activate a node that the final network does not need.

---

## 3. Repair Algorithm (Formal Description)

### Step 1 — Build the Full Reachability Graph

Construct the graph:
$$
G = (Q, E), \quad (i,j) \in E \iff \lVert q_i - q_j \rVert \le r.
$$

---

### Step 2 — Identify Components and the Sink's Gateways

* Consider only vertices with $x_i = 1$;
* Compute the connected components:
  $$
  C_0, C_1, \dots, C_k;
  $$
* Compute the **gateways**, the active candidates within reach of the sink:
  $$
  W = \{ i \mid x_i = 1, \ \lVert q_i - \sigma \rVert \le r \}.
  $$
* If $W = \emptyset$, activate the candidate nearest to the sink among those
  within $r$. If there is none, the repair fails.

If every component contains a gateway, or the mask is empty, the algorithm terminates.

---

### Step 3 — Iterative Rooted Repair

While there exists a component $C$ with $C \cap W = \emptyset$:

1. Select the first such component $C$;
2. Compute the **shortest path in the full graph $G$**, by breadth-first search, between:

   * any gateway $u \in W$,
   * any node $v \in C$;
3. If there is no path, the repair fails. Otherwise activate all nodes along this path;
4. Recompute the components and $W$.

Repeat until every component contains a gateway. The search starts at the
gateways and not at every node already connected to the sink, which is one
reason the result is not minimal.

---
