from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import random

from pylib.dto.problems import ProblemP1, ProblemP2, ProblemP3, ProblemP4
from .problem_adapter import ProblemAdapter, Individual, Chromosome


# ============================================================
# Helper utilities (generic)
# ============================================================

def _rand_uniform_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Sample a point uniformly in an axis-aligned bounding box (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = box
    return (random.uniform(xmin, xmax), random.uniform(ymin, ymax))


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance in R^2."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


def _uniform_crossover_mask(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    """Uniform crossover for binary masks."""
    assert len(a) == len(b)
    c1, c2 = a[:], b[:]
    for i in range(len(a)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def _bitflip_mutation(mask: list[int], p: float) -> list[int]:
    """Bit-flip mutation with per-bit probability p."""
    out = mask[:]
    for i in range(len(out)):
        if random.random() < p:
            out[i] = 1 - out[i]
    return out


def _blend_crossover_vec(x: list[float], y: list[float], alpha: float = 0.5) -> tuple[list[float], list[float]]:
    """
    Simple BLX-like blend crossover for real vectors.
    This is a lightweight placeholder; you can replace with SBX later.
    """
    assert len(x) == len(y)
    c1, c2 = [], []
    for i in range(len(x)):
        lo = min(x[i], y[i])
        hi = max(x[i], y[i])
        span = hi - lo
        a = lo - alpha * span
        b = hi + alpha * span
        c1.append(random.uniform(a, b))
        c2.append(random.uniform(a, b))
    return c1, c2


def _gaussian_mutation_vec(x: list[float], sigma: float, bounds: tuple[list[float], list[float]] | None) -> list[float]:
    """
    Gaussian mutation for real vectors, optionally clamped to bounds.
    """
    out = x[:]
    for i in range(len(out)):
        out[i] = out[i] + random.gauss(0.0, sigma)
    if bounds is not None:
        lb, ub = bounds
        out = [_clamp(out[i], lb[i], ub[i]) for i in range(len(out))]
    return out


# ============================================================
# Problem 1: Continuous coverage with mobility
# Chromosome: list of N points in Omega (continuous placement)
# ============================================================

class Problem1ContinuousMobilityAdapter(ProblemAdapter):
    """
    Problem 1 adapter.

    Chromosome representation:
      chromosome := P = [ (x1,y1), ..., (xN,yN) ] where each (xi,yi) in Omega.

    Notes:
    - In practice, Omega is often handled as a bounding box plus optional obstacles.
      Here we treat Omega as a bounding box for sampling/mutation.
    - Feasibility over continuous time is typically approximated by sampling time points.
    """

    @property
    def n_objectives(self) -> int:
        # Multiobjective criteria (energy, latency, throughput) can be 3 by default.
        return int(self.problem.get("problem_parameters", {}).get("n_objectives", 3))

    def _box(self) -> tuple[float, float, float, float]:
        pp = self.problem.get("problem_parameters", {})
        # Default unit square if not provided
        return tuple(pp.get("omega_box", [0.0, 0.0, 1.0, 1.0]))  # xmin,ymin,xmax,ymax

    def _N(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("N_fixed", 20))

    def bounds(self):
        # Bounds for a flattened vector are awkward; we keep points. Return None.
        return None

    def sample_initial_population(self, size: int) -> list[Individual]:
        box = self._box()
        N = self._N()
        pop: list[Individual] = []
        for _ in range(size):
            P = [_rand_uniform_point(box) for _ in range(N)]
            pop.append(Individual(chromosome=P))
        return pop

    def crossover(self, parents: Sequence[Chromosome]) -> list[Chromosome]:
        """
        Crossover for continuous placements:
        - pairwise blend crossover on each corresponding sensor position.
        """
        P1: list[tuple[float, float]] = parents[0]
        P2: list[tuple[float, float]] = parents[1]
        assert len(P1) == len(P2)

        box = self._box()
        xmin, ymin, xmax, ymax = box

        C1, C2 = [], []
        for (x1, y1), (x2, y2) in zip(P1, P2):
            # Simple arithmetic crossover
            w = random.random()
            cx1 = _clamp(w * x1 + (1.0 - w) * x2, xmin, xmax)
            cy1 = _clamp(w * y1 + (1.0 - w) * y2, ymin, ymax)
            cx2 = _clamp((1.0 - w) * x1 + w * x2, xmin, xmax)
            cy2 = _clamp((1.0 - w) * y1 + w * y2, ymin, ymax)
            C1.append((cx1, cy1))
            C2.append((cx2, cy2))
        return [C1, C2]

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutation for continuous placements:
        - Gaussian perturbation of a subset of points.
        """
        P: list[tuple[float, float]] = chromosome
        box = self._box()
        xmin, ymin, xmax, ymax = box
        sigma = float(self.problem.get("problem_parameters", {}).get("sigma_pos", 0.02))
        p_point = float(self.problem.get("problem_parameters", {}).get("pm_point", 0.2))

        out = []
        for (x, y) in P:
            if random.random() < p_point:
                x = _clamp(x + random.gauss(0.0, sigma), xmin, xmax)
                y = _clamp(y + random.gauss(0.0, sigma), ymin, ymax)
            out.append((x, y))
        return out

    def encode_simulation_input(self, ind: Individual) -> dict[str, Any]:
        """
        Map placement P + mobility trajectories reference into Simulation parameters.

        You should implement:
        - how trajectories Gamma are referenced (by id, by file in GridFS, etc.)
        - how to build the simulation input config for your master-node/Cooja pipeline
        """
        pp = self.problem.get("problem_parameters", {})
        return {
            "problem_id": "problem1_continuous_mobility",
            "P_fixed_positions": ind.chromosome,  # list of (x,y)
            "sink_position": pp.get("sink_position"),
            "trajectories_ref": pp.get("trajectories_ref"),  # e.g., GridFS id or document reference
            "R_com": pp.get("R_com"),
            "R_inter": pp.get("R_inter"),
        }

    def decode_simulation_output(self, sim_doc: Mapping[str, Any], ind: Individual) -> None:
        """
        Decode objectives from Simulation metrics.

        Expected sim_doc structure is project-specific. A common pattern:
          sim_doc["metrics"] = {"energy_avg":..., "latency_avg":..., "throughput":..., "feasible":...}

        You should also define constraint_violation if you want to penalize infeasibility.
        """
        metrics = sim_doc.get("metrics", {})
        # Default ordering: minimize energy, minimize latency, maximize throughput -> store as negative for minimization
        energy = float(metrics.get("energy_avg", 1e9))
        latency = float(metrics.get("latency_avg", 1e9))
        throughput = float(metrics.get("throughput", 0.0))

        ind.objectives = [energy, latency, -throughput]

        feasible = bool(metrics.get("feasible", True))
        ind.constraint_violation = 0.0 if feasible else float(metrics.get("cv", 1.0))


# ============================================================
# Problem 2: Discrete coverage with mobility
# Chromosome: binary mask over candidate positions Q (install / not install)
# ============================================================

class Problem2DiscreteMobilityAdapter(ProblemAdapter):
    """
    Problem 2 adapter.

    Chromosome representation:
      chromosome := mask in {0,1}^J selecting subset P ⊆ Q.
    """

    @property
    def n_objectives(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("n_objectives", 3))

    def _Q(self) -> list[tuple[float, float]]:
        pp = self.problem.get("problem_parameters", {})
        Q = pp.get("Q_candidates")
        if Q is None:
            raise ValueError("problem_parameters.Q_candidates must be provided for Problem 2.")
        return [tuple(p) for p in Q]

    def sample_initial_population(self, size: int) -> list[Individual]:
        Q = self._Q()
        J = len(Q)
        # Bias toward sparse selections (since the primary goal is to minimize |P|)
        p_on = float(self.problem.get("problem_parameters", {}).get("p_on_init", 0.15))

        pop: list[Individual] = []
        for _ in range(size):
            mask = [1 if random.random() < p_on else 0 for _ in range(J)]
            # Ensure not empty (optional)
            if sum(mask) == 0:
                mask[random.randrange(J)] = 1
            pop.append(Individual(chromosome=mask))
        return pop

    def crossover(self, parents: Sequence[Chromosome]) -> list[Chromosome]:
        m1: list[int] = parents[0]
        m2: list[int] = parents[1]
        c1, c2 = _uniform_crossover_mask(m1, m2)
        return [c1, c2]

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        mask: list[int] = chromosome
        p_bit = float(self.problem.get("problem_parameters", {}).get("p_bit_mut", 0.01))
        out = _bitflip_mutation(mask, p_bit)
        # Keep at least one selected position (optional)
        if sum(out) == 0 and len(out) > 0:
            out[random.randrange(len(out))] = 1
        return out

    def encode_simulation_input(self, ind: Individual) -> dict[str, Any]:
        pp = self.problem.get("problem_parameters", {})
        Q = self._Q()
        mask: list[int] = ind.chromosome
        P = [Q[i] for i, b in enumerate(mask) if b == 1]

        return {
            "problem_id": "problem2_discrete_mobility",
            "P_fixed_positions": P,
            "mask": mask,
            "sink_position": pp.get("sink_position"),
            "trajectories_ref": pp.get("trajectories_ref"),
            "R_com": pp.get("R_com"),
            "R_inter": pp.get("R_inter"),
        }

    def decode_simulation_output(self, sim_doc: Mapping[str, Any], ind: Individual) -> None:
        metrics = sim_doc.get("metrics", {})
        energy = float(metrics.get("energy_avg", 1e9))
        latency = float(metrics.get("latency_avg", 1e9))
        throughput = float(metrics.get("throughput", 0.0))

        # Add a sparsity term as implicit pressure (optional):
        # primary goal of Problem 2 is minimize |P|; you can model it as a 4th objective,
        # or keep it as a constraint/penalty. Here we keep it as penalty into constraint_violation.
        mask: list[int] = ind.chromosome
        size_P = float(sum(mask))
        max_P = float(self.problem.get("problem_parameters", {}).get("max_P_soft", 1e9))
        cv_size = max(0.0, size_P - max_P)

        ind.objectives = [energy, latency, -throughput]
        feasible = bool(metrics.get("feasible", True))
        cv_feas = 0.0 if feasible else float(metrics.get("cv", 1.0))
        ind.constraint_violation = cv_feas + cv_size


# ============================================================
# Problem 3: Sensing coverage with targets
# Chromosome: binary mask over candidate positions Q
# Constraint-style feasibility: k-coverage of targets and g-min-degree connectivity
# ============================================================

class Problem3TargetCoverageAdapter(ProblemAdapter):
    """
    Problem 3 adapter.

    Chromosome representation:
      chromosome := mask in {0,1}^J selecting subset P ⊆ Q.

    Feasibility (intended):
      - Each target is covered by at least k sensors within R_cov.
      - Each installed sensor has degree at least g in communication graph (within R_com).

    Notes:
    - You can enforce these as hard constraints (constraint_violation > 0 if violated),
      or you can keep them as objectives/penalties depending on your algorithm settings.
    """

    @property
    def n_objectives(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("n_objectives", 3))

    def _Q(self) -> list[tuple[float, float]]:
        pp = self.problem.get("problem_parameters", {})
        Q = pp.get("Q_candidates")
        if Q is None:
            raise ValueError("problem_parameters.Q_candidates must be provided for Problem 3.")
        return [tuple(p) for p in Q]

    def _targets(self) -> list[tuple[float, float]]:
        pp = self.problem.get("problem_parameters", {})
        Xi = pp.get("targets")
        if Xi is None:
            raise ValueError("problem_parameters.targets must be provided for Problem 3.")
        return [tuple(p) for p in Xi]

    def _k(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("k_coverage", 1))

    def _g(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("g_min_degree", 0))

    def _Rcov(self) -> float:
        return float(self.problem.get("problem_parameters", {}).get("R_cov", 0.2))

    def _Rcom(self) -> float:
        return float(self.problem.get("problem_parameters", {}).get("R_com", 0.2))

    def sample_initial_population(self, size: int) -> list[Individual]:
        Q = self._Q()
        J = len(Q)
        # Bias toward sparse selections but ensure some minimum
        p_on = float(self.problem.get("problem_parameters", {}).get("p_on_init", 0.2))
        min_on = int(self.problem.get("problem_parameters", {}).get("min_on_init", 1))

        pop: list[Individual] = []
        for _ in range(size):
            mask = [1 if random.random() < p_on else 0 for _ in range(J)]
            while sum(mask) < min_on:
                mask[random.randrange(J)] = 1
            pop.append(Individual(chromosome=mask))
        return pop

    def crossover(self, parents: Sequence[Chromosome]) -> list[Chromosome]:
        m1: list[int] = parents[0]
        m2: list[int] = parents[1]
        c1, c2 = _uniform_crossover_mask(m1, m2)
        return [c1, c2]

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        mask: list[int] = chromosome
        p_bit = float(self.problem.get("problem_parameters", {}).get("p_bit_mut", 0.02))
        out = _bitflip_mutation(mask, p_bit)
        if sum(out) == 0 and len(out) > 0:
            out[random.randrange(len(out))] = 1
        return out

    def _compute_constraint_violation_static(self, mask: list[int]) -> float:
        """
        Compute a static (geometry-only) feasibility penalty:
        - k-coverage of targets within R_cov
        - g-min-degree among installed sensors within R_com

        This does not include MAC/interference/traffic, which can be handled by simulation.
        """
        Q = self._Q()
        Xi = self._targets()
        k = self._k()
        g = self._g()
        Rcov = self._Rcov()
        Rcom = self._Rcom()

        P = [Q[i] for i, b in enumerate(mask) if b == 1]
        if len(P) == 0:
            # Heavily penalize empty deployment
            return 1e6

        cv = 0.0

        # k-coverage violations
        for xi in Xi:
            covered = sum(1 for p in P if _euclid(p, xi) <= Rcov)
            if covered < k:
                cv += float(k - covered)

        # g-min-degree violations
        if g > 0:
            for i in range(len(P)):
                deg = 0
                for j in range(len(P)):
                    if i == j:
                        continue
                    if _euclid(P[i], P[j]) <= Rcom:
                        deg += 1
                if deg < g:
                    cv += float(g - deg)

        return cv

    def encode_simulation_input(self, ind: Individual) -> dict[str, Any]:
        """
        For Problem 3, you may run:
        - a purely geometric evaluation (fast) AND/OR
        - a full simulation (Cooja) to estimate energy/latency/throughput.

        Here we pass both the selected positions and targets.
        """
        pp = self.problem.get("problem_parameters", {})
        Q = self._Q()
        mask: list[int] = ind.chromosome
        P = [Q[i] for i, b in enumerate(mask) if b == 1]

        # Optional: compute static constraint violation early and store in meta
        ind.meta["cv_static"] = self._compute_constraint_violation_static(mask)

        return {
            "problem_id": "problem3_target_coverage",
            "P_fixed_positions": P,
            "mask": mask,
            "targets": self._targets(),
            "sink_position": pp.get("sink_position"),
            "R_cov": pp.get("R_cov"),
            "R_com": pp.get("R_com"),
            "R_inter": pp.get("R_inter"),
            "k_coverage": self._k(),
            "g_min_degree": self._g(),
        }

    def decode_simulation_output(self, sim_doc: Mapping[str, Any], ind: Individual) -> None:
        metrics = sim_doc.get("metrics", {})
        energy = float(metrics.get("energy_avg", 1e9))
        latency = float(metrics.get("latency_avg", 1e9))
        throughput = float(metrics.get("throughput", 0.0))

        ind.objectives = [energy, latency, -throughput]

        # Combine static feasibility + simulation feasibility (if provided).
        cv_static = float(ind.meta.get("cv_static", 0.0))
        feasible_sim = bool(metrics.get("feasible", True))
        cv_sim = 0.0 if feasible_sim else float(metrics.get("cv", 1.0))

        ind.constraint_violation = cv_static + cv_sim


# ============================================================
# Problem 4: Data collection with mobile sink
# Chromosome: (route, sojourn_times)
#   route: list[int] indices over L, starting/ending at base index
#   sojourn_times: list[float] nonnegative, length = len(route)
# ============================================================

class Problem4MobileSinkCollectionAdapter(ProblemAdapter):
    """
    Problem 4 adapter.

    Chromosome representation:
      chromosome := (route, tau)
        - route: [v0, v1, ..., vT] indices over L, with v0 = vT = base_index
        - tau:   [tau0, tau1, ..., tauT] nonnegative real times

    Notes:
    - Route feasibility requires adjacency in mobility graph A.
    - You can either enforce feasibility via repair (recommended) or penalize violations.
    - This adapter is designed to match your earlier mixed-encoding concept:
      a symbolic path (route) + real vector (sojourn times).
    """

    @property
    def n_objectives(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("n_objectives", 3))

    def _L(self) -> list[tuple[float, float]]:
        pp = self.problem.get("problem_parameters", {})
        L = pp.get("L_stops")
        if L is None:
            raise ValueError("problem_parameters.L_stops must be provided for Problem 4.")
        return [tuple(p) for p in L]

    def _base_index(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("base_index", 0))

    def _A(self) -> set[tuple[int, int]]:
        """
        Mobility edges as pairs of indices (u,v).
        We treat A as directed or undirected depending on input.
        """
        pp = self.problem.get("problem_parameters", {})
        A = pp.get("A_edges")
        if A is None:
            raise ValueError("problem_parameters.A_edges must be provided for Problem 4.")
        return set((int(u), int(v)) for (u, v) in A)

    def _max_route_len(self) -> int:
        return int(self.problem.get("problem_parameters", {}).get("max_route_len", 10))

    def _tau_bounds(self) -> tuple[float, float]:
        pp = self.problem.get("problem_parameters", {})
        return (float(pp.get("tau_min", 0.0)), float(pp.get("tau_max", 100.0)))

    def _random_route(self) -> list[int]:
        """
        Sample a random feasible route in the mobility graph, starting/ending at base.
        Simple random walk with a return to base.
        """
        base = self._base_index()
        A = self._A()
        max_len = self._max_route_len()

        route = [base]
        cur = base
        # Random walk
        for _ in range(max_len - 1):
            neighbors = [v for (u, v) in A if u == cur]
            if not neighbors:
                break
            nxt = random.choice(neighbors)
            route.append(nxt)
            cur = nxt
            # Occasionally decide to stop early
            if random.random() < 0.2:
                break

        # Ensure return to base with one more hop if possible; otherwise force base (may be infeasible and repaired later)
        if cur != base:
            neighbors = [v for (u, v) in A if u == cur]
            if base in neighbors:
                route.append(base)
            else:
                route.append(base)  # will be repaired/penalized
        else:
            route.append(base)  # trivial loop

        return route

    def _repair_route(self, route: list[int]) -> list[int]:
        """
        Repair route to ensure:
        - starts/ends at base
        - each consecutive edge is in A

        Strategy:
        - enforce endpoints = base
        - if an invalid edge occurs, truncate and return to base if possible
        """
        base = self._base_index()
        A = self._A()

        if not route:
            return [base, base]

        route = route[:]
        route[0] = base
        route[-1] = base

        repaired = [route[0]]
        cur = repaired[0]
        for nxt in route[1:]:
            if (cur, nxt) in A:
                repaired.append(nxt)
                cur = nxt
            else:
                # Try to return to base directly; otherwise truncate hard and append base
                if (cur, base) in A:
                    repaired.append(base)
                else:
                    repaired.append(base)
                return repaired

        return repaired

    def _random_tau(self, length: int) -> list[float]:
        """Sample nonnegative sojourn times with bounds."""
        lo, hi = self._tau_bounds()
        return [random.uniform(lo, hi) for _ in range(length)]

    def sample_initial_population(self, size: int) -> list[Individual]:
        pop: list[Individual] = []
        for _ in range(size):
            route = self._repair_route(self._random_route())
            tau = self._random_tau(len(route))
            pop.append(Individual(chromosome=(route, tau)))
        return pop

    def crossover(self, parents: Sequence[Chromosome]) -> list[Chromosome]:
        """
        Crossover for (route, tau):
        - Route: single cut splice (very simple), then repair
        - Tau: blend crossover on aligned prefix (pad/trim to match repaired route)
        """
        (r1, t1) = parents[0]
        (r2, t2) = parents[1]

        # Route crossover: splice
        cut1 = random.randrange(1, max(2, len(r1))) if len(r1) > 2 else 1
        cut2 = random.randrange(1, max(2, len(r2))) if len(r2) > 2 else 1

        child_r1 = r1[:cut1] + r2[cut2:]
        child_r2 = r2[:cut2] + r1[cut1:]

        child_r1 = self._repair_route(child_r1)
        child_r2 = self._repair_route(child_r2)

        # Tau crossover: align to new route lengths
        def make_child_tau(rt: list[int], a: list[float], b: list[float]) -> list[float]:
            L = len(rt)
            # Build base vectors by trunc/pad
            aa = (a + self._random_tau(L))[:L]
            bb = (b + self._random_tau(L))[:L]
            c, _ = _blend_crossover_vec(aa, bb, alpha=0.25)
            lo, hi = self._tau_bounds()
            return [_clamp(x, lo, hi) for x in c]

        child_t1 = make_child_tau(child_r1, t1, t2)
        child_t2 = make_child_tau(child_r2, t2, t1)

        return [(child_r1, child_t1), (child_r2, child_t2)]

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Mutation for (route, tau):
        - With some probability, perform a local route edit (random node replacement) then repair.
        - Apply Gaussian mutation to tau.
        """
        (route, tau) = chromosome
        route = route[:]
        tau = tau[:]

        # Route mutation
        p_route = float(self.problem.get("problem_parameters", {}).get("pm_route", 0.2))
        if len(route) > 3 and random.random() < p_route:
            # Mutate one internal node
            idx = random.randrange(1, len(route) - 1)
            # Replace with a random node index
            n_nodes = len(self._L())
            route[idx] = random.randrange(0, n_nodes)
            route = self._repair_route(route)
            # Resize tau accordingly
            if len(tau) != len(route):
                tau = (tau + self._random_tau(len(route)))[:len(route)]

        # Tau mutation
        p_tau = float(self.problem.get("problem_parameters", {}).get("pm_tau", 0.5))
        if random.random() < p_tau:
            sigma = float(self.problem.get("problem_parameters", {}).get("sigma_tau", 5.0))
            lo, hi = self._tau_bounds()
            tau = [ _clamp(x + random.gauss(0.0, sigma), lo, hi) for x in tau ]

        return (route, tau)

    def encode_simulation_input(self, ind: Individual) -> dict[str, Any]:
        pp = self.problem.get("problem_parameters", {})
        (route, tau) = ind.chromosome

        return {
            "problem_id": "problem4_mobile_sink_collection",
            "route": route,
            "sojourn_times": tau,
            "base_index": self._base_index(),
            "L_stops": pp.get("L_stops"),   # optionally avoid duplication and pass an id/reference
            "A_edges": pp.get("A_edges"),
            "fixed_sensors": pp.get("fixed_sensors"),  # positions + rates + energy + buffers; could be a reference
            "R_com": pp.get("R_com"),
            "R_inter": pp.get("R_inter"),
        }

    def decode_simulation_output(self, sim_doc: Mapping[str, Any], ind: Individual) -> None:
        metrics = sim_doc.get("metrics", {})
        energy = float(metrics.get("energy_avg", 1e9))
        latency = float(metrics.get("latency_avg", 1e9))
        throughput = float(metrics.get("throughput", 0.0))

        ind.objectives = [energy, latency, -throughput]

        feasible = bool(metrics.get("feasible", True))
        ind.constraint_violation = 0.0 if feasible else float(metrics.get("cv", 1.0))
