# Continuous Network Generation via Connectivity-Constrained Spatial Expansion

This method defines a **continuous network generation algorithm** that incrementally places points in a planar region while **maximizing spatial dispersion** and **preserving global connectivity** under a fixed communication radius.

It is designed to generate **connected geometric graphs** with good spatial coverage, suitable as initial feasible solutions for continuous deployment problems in wireless sensor networks.

---

## 1. Formal Problem Setting

Let:

* $\Omega = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \subset \mathbb{R}^2$ be a rectangular region;
* $N \in \mathbb{Z}_+$ be the desired number of points;
* $r > 0$ be a fixed communication radius.

For a finite set of points $P \subset \Omega$, define the induced geometric graph:
$$
G(P) = (P, E), \quad (p_i, p_j) \in E \iff \lVert p_i - p_j \rVert \le r.
$$

---

## 2. Objective

Construct a set of points
$$
P = {p_1, \dots, p_N} \subset \Omega
$$
such that:

1. The graph $G(P)$ is **connected**;
2. Points are **spatially well distributed**, i.e., new points tend to maximize their minimum distance to existing points;
3. Connectivity is preserved at every insertion step.

This is a **constructive heuristic** balancing coverage and connectivity in continuous space.

---

## 3. Initialization

The algorithm begins with a single root point placed at the geometric center of the region:
$$
p_1 = \left( \frac{x_{\min} + x_{\max}}{2}, \frac{y_{\min} + y_{\max}}{2} \right).
$$

At iteration $k$, the current point set is:
$$
P_k = {p_1, \dots, p_k}.
$$

---

## 4. Connectivity Predicate

A set of points $P$ is considered **connected** if the graph $G(P)$ is connected.

Connectivity is tested using a breadth-first search (BFS) over the adjacency relation induced by the radius $r$.

---

## 5. Incremental Point Generation Strategy

While $|P_k| < N$, the algorithm attempts to insert a new point according to the following prioritized strategy.

---

### 5.1 Candidate Generation Around Anchors

For each insertion step, the algorithm performs up to $T$ trials (where $T =$ `max_attempts`):

1. Randomly select an **anchor point** $a \in P_k$;
2. Sample:

   * an angle $\theta \sim \mathcal{U}(0, 2\pi)$,
   * a distance $d \sim \mathcal{U}(0.5r, r)$;
3. Propose a candidate point:
   $$
   p = a + d \cdot (\cos \theta, \sin \theta).
   $$

The candidate is accepted for evaluation only if $p \in \Omega$ and $G(P_k \cup {p})$ remains connected.

---

### 5.2 Coverage-Oriented Selection Criterion

Among all valid candidates generated in the current iteration, the algorithm selects the point that maximizes:
$$
\delta(p, P_k) = \min_{q \in P_k} \lVert p - q \rVert.
$$

That is, the point which is **farthest from the existing set**, subject to preserving connectivity, is chosen:
$$
p_{k+1} = \arg\max_p \delta(p, P_k).
$$

This criterion explicitly encourages **spatial dispersion**.

---

## 6. Fallback: Component Bridging

If no valid candidate satisfying the above constraints is found:

1. The algorithm computes the connected components of $G(P_k)$;
2. If multiple components exist, two components $C_1$ and $C_2$ are selected;
3. A new point is inserted near the midpoint between randomly chosen points from each component:
   $$
   m = \frac{p_1 + p_2}{2}, \quad p_1 \in C_1,; p_2 \in C_2;
   $$
4. A small perturbation orthogonal to the connecting direction is applied;
5. The resulting point is clipped to remain inside $\Omega$.

This step explicitly **repairs connectivity** by bridging components.

---

## 7. Final Fallback: Random Local Insertion

If component bridging is not applicable, the algorithm inserts a point uniformly at random within distance $r$ of a randomly chosen anchor:
$$
p = a + d \cdot (\cos \theta, \sin \theta),
\quad d \sim \mathcal{U}(0, r).
$$

This guarantees progress and termination.

---

## 8. Final Connectivity Enforcement

After all $N$ points are generated, a final connectivity check is performed.

If $G(P)$ is not connected:

* The algorithm iteratively inserts midpoint nodes between disconnected components until global connectivity is achieved.

This ensures the output always satisfies:
$$
G(P) \text{ is connected}.
$$

---

## 9. Algorithmic Interpretation

This method can be interpreted as:

* A **connectivity-preserving spatial growth process**;
* A **continuous analogue of graph expansion under distance constraints**;
* A **greedy max–min dispersion heuristic with topological constraints**.

It is particularly well suited for generating **feasible initial configurations** in continuous deployment problems.

---

## 10. Computational Complexity

Let $N$ be the number of generated points.

* Each insertion performs up to $T$ connectivity checks;
* Each connectivity check is $O(k^2)$ in the naive implementation at step $k$.

Overall worst-case complexity:
$$
O(N^3) \quad \text{(naive connectivity checks)}.
$$

In practice, $T$ is small and $N$ is moderate, making the method effective for initialization purposes.

---

## 11. Key Properties

* **Guaranteed connectivity** of the final point set;
* **Bias toward uniform spatial coverage**;
* **No optimality guarantees**, by design;
* **Well suited as a feasibility generator**, not as a final optimizer.

---

Below is a **simplified, method-section–ready pseudo-code**, intentionally abstracted from implementation details while preserving the **algorithmic logic and mathematical intent**.
It is suitable for inclusion in a **Methods** or **Algorithm Description** section of a thesis or journal paper.

---

## Algorithm: Connectivity-Constrained Continuous Network Generation

**Input:**

* Target number of points $N$
* Rectangular region $\Omega \subset \mathbb{R}^2$
* Communication radius $r$
* Maximum number of attempts per insertion $T$

**Output:**

* A connected set of points $P \subset \Omega$ with $|P| = N$

### Pseudo-code

```
Algorithm ContinuousNetworkGeneration(N, Ω, r, T)

1:  Initialize P ← { center(Ω) }

2:  while |P| < N do
3:      best_candidate ← null
4:      best_score ← 0

5:      for t = 1 to T do
6:          Select anchor a uniformly at random from P
7:          Sample angle θ ∈ [0, 2π)
8:          Sample distance d ∈ [0.5r, r]
9:          p ← a + d · (cos θ, sin θ)

10:         if p ∉ Ω then
11:             continue
12:         end if

13:         if G(P ∪ {p}) is connected then
14:             score ← min_{q ∈ P} ||p − q||
15:             if score > best_score then
16:                 best_score ← score
17:                 best_candidate ← p
18:             end if
19:         end if
20:     end for

21:     if best_candidate ≠ null then
22:         P ← P ∪ {best_candidate}
23:     else
24:         if G(P) has multiple connected components then
25:             Select two components C₁, C₂
26:             Select p₁ ∈ C₁, p₂ ∈ C₂
27:             p ← midpoint(p₁, p₂) with small random perturbation
28:         else
29:             Select anchor a ∈ P
30:             Sample p uniformly within distance r of a
31:         end if
32:         Clip p to Ω
33:         P ← P ∪ {p}
34:     end if
35: end while

36: if G(P) is not connected then
37:     while G(P) has multiple components do
38:         Select two components C₁, C₂
39:         Insert midpoint between C₁ and C₂ into P
40:     end while
41: end if

42: return P
```

The implementation is available in `lib.util.random_network`, in the function `continuous_network_gen`.

---
