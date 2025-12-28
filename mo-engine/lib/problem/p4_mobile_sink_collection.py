from typing import Any, Mapping, Sequence, cast
import random
import math

from pylib.dto.simulator import FixedMote, MobileMote, SimulationElements
from pylib.dto.problems import ProblemP4
from pylib.dto.algorithm import GeneticAlgorithmConfigDto
from .adapter import ProblemAdapter, ChromosomeP4

# Aliases (coerentes com seus DTOs)
Position = tuple[float, float]
PathSeg = tuple[str, str]  # (x(t), y(t)) com t em [0,1]

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return lo if x < lo else hi if x > hi else x

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

# ============================================================
# Problem 4: Data collection with mobile sink
# ChromosomeP4: (route, sojourn_times)
#   route: list[int] indices over L, starting/ending at base index
#   sojourn_times: list[float] nonnegative, length = len(route)
# ============================================================

class Problem4MobileSinkCollectionAdapter(ProblemAdapter):
    """
    Problem 4 adapter.

    ChromosomeP4 representation:
      chromosome := (route, tau)
        - route: [v0, v1, ..., vT] indices over L, with v0 = vT = base_index
        - tau:   [tau0, tau1, ..., tauT] nonnegative real times

    Notes:
    - Route feasibility requires adjacency in mobility graph A.
    - You can either enforce feasibility via repair (recommended) or penalize violations.
    - This adapter is designed to match your earlier mixed-encoding concept:
      a symbolic path (route) + real vector (sojourn times).
    """
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        # -----------------------------
        # Núcleo homogêneo mínimo
        # -----------------------------
        _ = float(problem["radius_of_reach"])
        _ = float(problem["radius_of_inter"])

        region = problem["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("problem['region'] must be [xmin,ymin,xmax,ymax].")

        # -----------------------------
        # Campos específicos do P4
        # -----------------------------
        # nodes: list[Position]
        nodes = problem["nodes"]
        if not isinstance(nodes, list) or any(
            (not isinstance(p, tuple) or len(p) != 2) for p in nodes
        ):
            raise TypeError("problem['nodes'] must be list[Position] where Position is tuple[float,float].")

        # sink_base: Position
        sink_base = problem["sink_base"]
        if not isinstance(sink_base, tuple) or len(sink_base) != 2:
            raise TypeError("problem['sink_base'] must be a Position (tuple[float,float]).")

        # Scalars
        initial_energy = problem["initial_energy"]
        buffer_capacity = problem["buffer_capacity"]
        data_rate = problem["data_rate"]
        speed = problem["speed"]
        time_step = problem["time_step"]

        if float(initial_energy) <= 0:
            raise ValueError("problem['initial_energy'] must be > 0.")
        if float(buffer_capacity) <= 0:
            raise ValueError("problem['buffer_capacity'] must be > 0.")
        if float(data_rate) < 0:
            raise ValueError("problem['data_rate'] must be >= 0.")
        if float(speed) <= 0:
            raise ValueError("problem['speed'] must be > 0.")
        if float(time_step) <= 0:
            raise ValueError("problem['time_step'] must be > 0.")

        # sojourns: list[SojournLocation]
        sojourns = problem["sojourns"]
        if not isinstance(sojourns, list) or len(sojourns) == 0:
            raise TypeError("problem['sojourns'] must be a non-empty list[SojournLocation].")

        # Validate minimal SojournLocation structure + adjacency integrity
        sojourn_ids: set[int] = set()
        for s in sojourns:
            if not isinstance(s, dict):
                raise TypeError("Each sojourn must be a mapping/dict.")
            if "id" not in s or "position" not in s or "adjacency" not in s or "visibleNodes" not in s:
                raise KeyError("Each sojourn must have keys: id, position, adjacency, visibleNodes.")

            sid = s["id"]
            if not isinstance(sid, int):
                raise TypeError("sojourn['id'] must be int.")
            if sid in sojourn_ids:
                raise ValueError(f"Duplicate sojourn id: {sid}")
            sojourn_ids.add(sid)

            pos = s["position"]
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise TypeError("sojourn['position'] must be Position (tuple[float,float]).")

            adj = s["adjacency"]
            if not isinstance(adj, list) or any(not isinstance(a, int) for a in adj):
                raise TypeError("sojourn['adjacency'] must be list[int].")

            vis = s["visibleNodes"]
            if not isinstance(vis, list) or any(not isinstance(i, int) for i in vis):
                raise TypeError("sojourn['visibleNodes'] must be list[int].")

        # Ensure adjacency refers to existing sojourn ids
        for s in sojourns:
            sid = s["id"]
            for nbr in s["adjacency"]:
                if nbr not in sojourn_ids:
                    raise ValueError(f"Sojourn {sid} has adjacency to unknown sojourn id: {nbr}")

        # validar visibleNodes dentro do range de nodes
        n_nodes = len(nodes)
        for s in sojourns:
            for i in s["visibleNodes"]:
                if not (0 <= i < n_nodes):
                    raise ValueError(
                        f"Sojourn {s['id']} references visibleNodes index {i}, "
                        f"but nodes has size {n_nodes}."
                    )

        self.problem: ProblemP4 = ProblemP4.cast(problem)

    def set_ga_parameters(self, parameters: GeneticAlgorithmConfigDto):    
        self._p_on_init = float(parameters.get("p_on_init", 0.15))    
        self._p_cx = float(parameters.get("prob_cx", 0.9))
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))
        self._ensure_non_empty = bool(parameters.get("ensure_non_empty", True))

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

    def random_individual_generator(self, size: int) -> list[ChromosomeP4]:
        pop: list[ChromosomeP4] = []
        for _ in range(size):
            route = self._repair_route(self._random_route())
            tau = self._random_tau(len(route))
            pop.append(ChromosomeP4(chromosome=(route, tau)))
        return pop

    def crossover(self, parents: Sequence[ChromosomeP4]) -> list[ChromosomeP4]:
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

    def mutate(self, chromosome: ChromosomeP4) -> ChromosomeP4:
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

    def _linseg_expr(self, p0: Position, p1: Position) -> PathSeg:
        """
        Segmento linear de duração 1 passo (Δt), usando parâmetro local t ∈ [0,1].
        """
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0

        # Expressões simbólicas compatíveis com o padrão já usado no SimLab
        # (ex.: "70 - 140 * t", "44 + 31 * t")
        def fmt(a: float) -> str:
            # evita "-0.0"
            if abs(a) < 1e-12:
                a = 0.0
            # string limpa (sem notação científica desnecessária)
            s = f"{a:.10g}"
            return s

        x_expr = f"{fmt(x0)} + {fmt(dx)} * t" if dx != 0 else f"{fmt(x0)}"
        y_expr = f"{fmt(y0)} + {fmt(dy)} * t" if dy != 0 else f"{fmt(y0)}"
        return (x_expr, y_expr)


    def _discretize_travel(self, p0: Position, p1: Position, speed: float, time_step: float) -> list[Position]:
        """
        Gera uma sequência de pontos intermediários para deslocamento p0 -> p1.
        Cada passo corresponde a Δt = time_step e percorre aprox. speed*time_step.
        Retorna pontos incluindo p0 e p1.
        """
        x0, y0 = p0
        x1, y1 = p1
        dist = math.hypot(x1 - x0, y1 - y0)

        if dist == 0:
            return [p0]

        step_len = speed * time_step
        if step_len <= 0:
            raise ValueError("ProblemP4.speed and time_step must be positive to discretize travel.")

        n_steps = max(1, int(math.ceil(dist / step_len)))
        pts: list[Position] = []
        for k in range(n_steps + 1):
            alpha = k / n_steps
            pts.append((x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)))
        return pts


    def _discretize_dwell(self, p: Position, dwell_time: float, time_step: float) -> list[Position]:
        """
        Gera repetição de pontos para permanência (sojourn) durante dwell_time.
        Retorna uma lista de pontos (todos iguais a p). Cada ponto corresponde a 1 passo (Δt).
        """
        if dwell_time <= 0:
            return []
        if time_step <= 0:
            raise ValueError("ProblemP4.time_step must be positive to discretize dwell.")

        n = int(math.ceil(dwell_time / time_step))
        return [p] * n


    def encode_simulation_input(self, ind: ChromosomeP4) -> SimulationElements:
        fixed: list[FixedMote] = []
        mobile: list[MobileMote] = []

        # -------------------------------------------------
        # Structural checks (cromossomo)
        # -------------------------------------------------
        if len(ind.route) == 0:
            raise ValueError("ChromosomeP4.route must be non-empty.")
        if len(ind.sojourn_times) != len(ind.route):
            raise ValueError(
                f"ChromosomeP4.sojourn_times length ({len(ind.sojourn_times)}) must match "
                f"route length ({len(ind.route)})."
            )

        L = self.problem.sojourns
        if any((idx < 0 or idx >= len(L)) for idx in ind.route):
            raise ValueError("ChromosomeP4.route has invalid sojourn index (out of bounds).")

        if self.problem.speed <= 0:
            raise ValueError("ProblemP4.speed must be > 0.")
        if self.problem.time_step <= 0:
            raise ValueError("ProblemP4.time_step must be > 0.")

        # -------------------------------------------------
        # Fixed motes: nodes N (sensores)
        # -------------------------------------------------
        for i, (x, y) in enumerate(self.problem.nodes):
            fixed.append({
                "name": f"node_{i}",
                "sourceCode": "node.c",
                "position": [x, y],
                "radiusOfReach": self.problem.radius_of_reach,
                "radiusOfInter": self.problem.radius_of_inter,
            })

        # -------------------------------------------------
        # Build sink path: Base -> sojourns(route) -> Base
        # plus dwell at each visited sojourn according to sojourn_times
        # -------------------------------------------------
        base: Position = self.problem.sink_base

        # Extrai posições dos sojourns na ordem da rota
        route_positions: list[Position] = [tuple(L[idx].position) for idx in ind.route]
        
        # Discretização total em pontos (cada passo ≈ Δt)
        timeline_points: list[Position] = []

        # Move: base -> first
        travel_pts = self._discretize_travel(
            base, 
            route_positions[0], 
            self.problem.speed, 
            self.problem.time_step
            )
        timeline_points.extend(travel_pts)

        # Para cada sojourn i: dwell e travel para o próximo
        for j, p in enumerate(route_positions):
            # Dwell at current sojourn p (não duplicar ponto inicial da permanência se já está no timeline_points)
            dwell_pts = self._discretize_dwell(p, float(ind.sojourn_times[j]), self.problem.time_step)
            timeline_points.extend(dwell_pts)

            # Travel to next sojourn (ou retorno à base no final)
            next_p = route_positions[j + 1] if (j + 1) < len(route_positions) else base
            travel_pts = self._discretize_travel(p, next_p, self.problem.speed, self.problem.time_step)

            # Evita duplicar o primeiro ponto (p), pois já estamos em p
            if len(travel_pts) > 1:
                timeline_points.extend(travel_pts[1:])

        # -------------------------------------------------
        # Convert timeline points to functionPath (segments)
        # Each consecutive pair defines one Δt segment.
        # -------------------------------------------------
        if len(timeline_points) < 2:
            # Degenerado: ainda assim precisamos de ao menos um segmento
            timeline_points = [base, base]

        function_path: list[PathSeg] = []
        for p0, p1 in zip(timeline_points[:-1], timeline_points[1:]):
            function_path.append(self._linseg_expr(p0, p1))

        # -------------------------------------------------
        # Mobile sink
        # -------------------------------------------------
        mobile.append({
            "name": "sink",
            "sourceCode": "sink.c",
            "functionPath": function_path,
            "isClosed": True,        # Base -> ... -> Base
            "isRoundTrip": False,    
            "speed": self.problem.speed,
            "timeStep": self.problem.time_step,
            "radiusOfReach": self.problem.radius_of_reach,
            "radiusOfInter": self.problem.radius_of_inter,
        })

        return {
            "fixedMotes": fixed,
            "mobileMotes": mobile,
        }