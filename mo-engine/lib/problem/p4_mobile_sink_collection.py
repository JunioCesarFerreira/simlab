from collections import deque
from typing import Any, Mapping, Sequence
import math

from pylib.config.simulator import FixedMote, MobileMote, SimulationElements
from pylib.config.problems import ProblemP4
from pylib.config.algorithm import GeneticAlgorithmConfigDto

from .adapter import ProblemAdapter, ChromosomeP4, Random

# Aliases (coerentes com seus DTOs)
Position = tuple[float, float]
PathSeg = tuple[str, str]  # (x(t), y(t)) com t em [0,1]

BASE_INDEX = 0 # fixed base index for base sojourn

# Same base as P1–P3: above any measured objective, so a penalised individual
# never reaches front 0.
_PENALTY_BASE = 1e9


def _is_position(value: Any) -> bool:
    """[x, y] or (x, y): JSON and BSON give lists, Python callers tuples."""
    return (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return lo if x < lo else hi if x > hi else x

def _blend_crossover_vec(
    x: list[float], 
    y: list[float], 
    rng: Random,
    alpha: float = 0.5
    ) -> tuple[list[float], list[float]]:
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
        c1.append(rng.uniform(a, b))
        c2.append(rng.uniform(a, b))
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

    Feasibility:
    - Route: every move is an edge of the mobility graph A; the repair keeps
      the valid prefix and returns to the base by the shortest walk over A.
    - Data: every mote is collected at some stop (it is in the connected part
      of the network that a stop sees, and the sink dwells there), and no
      buffer overflows between two collections over the repeated route
      (data_rate × longest gap ≤ buffer_capacity). Violations are penalised.
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
        if not isinstance(nodes, list) or not all(_is_position(p) for p in nodes):
            raise TypeError("problem['nodes'] must be a list of positions [x, y].")

        # sink_base: Position
        sink_base = problem["sink_base"]
        if not _is_position(sink_base):
            raise TypeError("problem['sink_base'] must be a position [x, y].")

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
        max_route_length = problem["max_route_length"]
        if isinstance(max_route_length, bool) or not isinstance(max_route_length, int) or max_route_length < 2:
            raise ValueError("problem['max_route_length'] must be a whole number >= 2.")
        tau_bounds = problem["tau_bounds"]
        if not _is_position(tau_bounds) or not 0 <= tau_bounds[0] <= tau_bounds[1]:
            raise ValueError("problem['tau_bounds'] must be [min, max] with 0 <= min <= max.")

        # sojourns: list[SojournLocation]
        sojourns = problem["sojourns"]
        if not isinstance(sojourns, list) or len(sojourns) == 0:
            raise TypeError("problem['sojourns'] must be a non-empty list[SojournLocation].")

        # Validate minimal SojournLocation structure + adjacency integrity
        sojourn_ids: set[int] = set()
        for index, s in enumerate(sojourns):
            if not isinstance(s, dict):
                raise TypeError("Each sojourn must be a mapping/dict.")
            if "id" not in s or "position" not in s or "adjacency" not in s or "visibleNodes" not in s:
                raise KeyError("Each sojourn must have keys: id, position, adjacency, visibleNodes.")

            sid = s["id"]
            if isinstance(sid, bool) or not isinstance(sid, int):
                raise TypeError("sojourn['id'] must be int.")
            # The route holds indices into this list and the adjacency holds
            # ids: they have to be the same numbers. Sojourn 0 is the base.
            if sid != index:
                raise ValueError(
                    f"Sojourn ids must equal their position in the list: "
                    f"the sojourn at {index} has id {sid}.")
            sojourn_ids.add(sid)

            pos = s["position"]
            if not _is_position(pos):
                raise TypeError("sojourn['position'] must be a position [x, y].")

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
        self._successors_of: dict[int, list[int]] = {}
        for s in self.problem.sojourns:
            self._successors_of[int(s.id)] = sorted({int(v) for v in s.adjacency})
        # Which motes each stop collects from: those in a connected part of
        # the network (multi-hop, within R_com) that holds a mote it sees.
        component = self._components(self.problem.nodes, self.problem.radius_of_reach)
        self._collects: list[frozenset[int]] = [
            frozenset(i for i in range(len(self.problem.nodes))
                      if component[i] in {component[v] for v in s.visibleNodes})
            for s in self.problem.sojourns
        ]


    # Operators are fixed by design: route splice crossover with repair + tau
    # blend; mutation combines route perturbation and Gaussian tau mutation.
    CONSUMED_GA_KEYS = frozenset({"per_gene_prob", "pm_tau", "sigma_tau"})

    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto):
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))
        self._pm_tau = float(parameters.get("pm_tau", 0.5))
        self._sigma_tau = float(parameters.get("sigma_tau", 5.0))
        self._rng = rng


    def _A(self) -> set[tuple[int, int]]:
        """
        Mobility edges A derived from sojourn adjacency lists.

        Each sojourn s contributes edges (s.id, v) for v in s.adjacency.
        """
        A: set[tuple[int, int]] = set()

        for s in self.problem.sojourns:
            u = int(s.id)
            for v in s.adjacency:
                A.add((u, int(v)))

        if not A:
            raise ValueError("Mobility graph A is empty (check sojourn adjacency lists).")

        return A


    @staticmethod
    def _components(points: Sequence[Position], radius: float) -> list[int]:
        """Connected-component label of each point, edges within ``radius``."""
        label = [-1] * len(points)
        for start in range(len(points)):
            if label[start] >= 0:
                continue
            label[start] = start
            stack = [start]
            while stack:
                u = stack.pop()
                for v in range(len(points)):
                    if label[v] < 0 and math.dist(points[u], points[v]) <= radius:
                        label[v] = start
                        stack.append(v)
        return label


    def _path_to_base(self, start: int) -> list[int] | None:
        """The shortest walk over A from ``start`` to the base, both included,
        or ``None`` when the base cannot be reached from it."""
        base = BASE_INDEX
        if start == base:
            return [base]
        previous: dict[int, int | None] = {start: None}
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in self._successors_of.get(u, []):
                if v in previous:
                    continue
                previous[v] = u
                if v == base:
                    path = [v]
                    while previous[path[-1]] is not None:
                        path.append(previous[path[-1]])
                    return path[::-1]
                queue.append(v)
        return None


    def _close(self, walk: list[int]) -> list[int]:
        """``walk`` (starting at the base, every move an edge) closed at the base
        by the shortest walk, within ``max_route_length`` stops when possible.
        Stops from which the base cannot be reached are dropped."""
        base = BASE_INDEX
        walk = list(walk) or [base]
        limit = max(2, self.problem.max_route_length)
        while True:
            back = self._path_to_base(walk[-1])
            if back is not None and (len(walk) + len(back) - 1 <= limit or len(walk) == 1):
                break
            walk.pop()
        route = walk + back[1:]
        return route if len(route) >= 2 else [base, base]


    def _random_route(self) -> list[int]:
        """
        A random walk over the mobility graph from the base, closed at the base
        by the shortest walk over existing edges.
        """
        base = BASE_INDEX
        walk = [base]
        for _ in range(self.problem.max_route_length - 2):
            neighbors = self._successors_of.get(walk[-1], [])
            if not neighbors:
                break
            walk.append(self._rng.choice(neighbors))
            # Occasionally decide to stop early
            if self._rng.random() < 0.2:
                break
        return self._close(walk)

    def _repair_route(self, route: list[int]) -> list[int]:
        """
        A route that starts and ends at the base with every move an edge of A.

        Keeps the longest valid prefix from the base and returns to the base by
        the shortest walk over A. The previous repair appended the base after
        the first invalid move, even where that last move did not exist.
        """
        base = BASE_INDEX
        walk = [base]
        for nxt in list(route)[1:]:
            if nxt in self._successors_of.get(walk[-1], []):
                walk.append(nxt)
            else:
                break
        return self._close(walk)


    def collection_deficits(self, chromosome: ChromosomeP4) -> tuple[float, float]:
        """How far the route is from collecting every mote's data, each in [0, 1].

        - uncollected: share of the motes that no stop with a positive dwell
          collects from;
        - overflow: mean over the collected motes of the data lost in the
          longest gap between two collections, over the buffer, capped at 1.

        The route repeats (the sink's path is closed), so the gap from the last
        collection to the first one of the next round counts too.
        """
        n = len(self.problem.nodes)
        if n == 0:
            return 0.0, 0.0
        L = self.problem.sojourns
        speed = float(self.problem.speed)
        windows: list[list[tuple[float, float]]] = [[] for _ in range(n)]
        t, here = 0.0, tuple(self.problem.sink_base)
        for stop, dwell in zip(chromosome.route, chromosome.sojourn_times):
            position = tuple(L[stop].position)
            t += math.dist(here, position) / speed
            here = position
            if dwell > 0:
                for i in self._collects[stop]:
                    windows[i].append((t, t + dwell))
            t += dwell
        period = t + math.dist(here, tuple(self.problem.sink_base)) / speed

        collected = [w for w in windows if w]
        uncollected = 1.0 - len(collected) / n
        rate, capacity = float(self.problem.data_rate), float(self.problem.buffer_capacity)
        if not collected or rate <= 0:
            return uncollected, 0.0
        losses = []
        for w in collected:
            w = sorted(w)
            gaps = [b[0] - a[1] for a, b in zip(w, w[1:])]
            gaps.append(w[0][0] + period - w[-1][1])
            losses.append(min(1.0, max(0.0, rate * max(gaps) - capacity) / capacity))
        return uncollected, sum(losses) / len(losses)


    def penalty_objectives(self, chromosome: ChromosomeP4, n_objectives: int) -> list[float] | None:
        """
        A minimization-space penalty when the route leaves a mote uncollected
        or lets a buffer overflow, ``None`` when it is feasible.

          deficit = mean of the two deficits  ∈ (0, 1]
          penalty = 1e9 × (1 + deficit)
        """
        deficits = self.collection_deficits(chromosome)
        if not any(deficits):
            return None
        return [_PENALTY_BASE * (1.0 + sum(deficits) / len(deficits))] * n_objectives


    def _random_tau(self, length: int) -> list[float]:
        """Sample nonnegative sojourn times with bounds."""
        lo, hi = self.problem.tau_bounds
        return [self._rng.uniform(lo, hi) for _ in range(length)]

    def random_individual_generator(self, size: int) -> list[ChromosomeP4]:
        pop: list[ChromosomeP4] = []
        for _ in range(size):
            route = self._repair_route(self._random_route())
            tau = self._random_tau(len(route))
            pop.append(ChromosomeP4(
                mac_protocol=self._rng.randint(0, 1),
                route=route,
                sojourn_times=tau,
            ))
        return pop


    def crossover(self, parents: Sequence[ChromosomeP4]) -> list[ChromosomeP4]:
        """
        Crossover for (route, sojourn_times):
        - Route: single cut splice (very simple), then repair
        - Tau: blend crossover on aligned prefix (pad/trim to match repaired route)
        - MAC gene inheritance (simple uniform choice)
        """
        p1, p2 = parents[0], parents[1]
        r1, t1 = p1.route, p1.sojourn_times
        r2, t2 = p2.route, p2.sojourn_times

        # Route crossover: splice
        cut1 = self._rng.randrange(1, max(2, len(r1))) if len(r1) > 2 else 1
        cut2 = self._rng.randrange(1, max(2, len(r2))) if len(r2) > 2 else 1

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
            c, _ = _blend_crossover_vec(aa, bb, rng=self._rng, alpha=0.25)
            lo, hi = self.problem.tau_bounds
            return [_clamp(x, lo, hi) for x in c]

        child_t1 = make_child_tau(child_r1, t1, t2)
        child_t2 = make_child_tau(child_r2, t2, t1)

        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP4(mac_protocol=mac1, route=child_r1, sojourn_times=child_t1),
            ChromosomeP4(mac_protocol=mac2, route=child_r2, sojourn_times=child_t2),
        ]


    def mutate(self, chromosome: ChromosomeP4) -> ChromosomeP4:
        """
        Mutation for (route, sojourn_times):
        - With some probability, perform a local route edit (random node replacement) then repair.
        - Apply Gaussian mutation to tau.
        - Bit-flip mutation on MAC gene.
        """
        route = chromosome.route[:]
        tau = chromosome.sojourn_times[:]

        # Route mutation
        p_route = self._p_bit_mut
        if len(route) > 3 and self._rng.random() < p_route:
            # Mutate one internal node
            idx = self._rng.randrange(1, len(route) - 1)
            # Replace with a random node index
            n_nodes = len(self.problem.sojourns)
            route[idx] = self._rng.randrange(0, n_nodes)
            route = self._repair_route(route)
            # Resize tau accordingly
            if len(tau) != len(route):
                tau = (tau + self._random_tau(len(route)))[:len(route)]

        # Tau mutation
        if self._rng.random() < self._pm_tau:
            sigma = self._sigma_tau
            lo, hi = self.problem.tau_bounds
            tau = [ _clamp(x + self._rng.gauss(0.0, sigma), lo, hi) for x in tau ]

        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if self._rng.random() < self._p_bit_mut:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP4(mac_protocol=mac, route=route, sojourn_times=tau)


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