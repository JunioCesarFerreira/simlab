from typing import Any, Mapping, Sequence
import math

from pylib.config.simulator import FixedMote, MobileMote, SimulationElements
from pylib.config.problems import ProblemP1
from pylib.config.algorithm import GeneticAlgorithmConfigDto

from lib.util.random_network import continuous_network_gen
from lib.util.connectivity import make_graph_connected_to_sink, is_connected
from lib.util.region_partition import TrajectoryConstraintP1
from lib.genetic_operators.crossover.simulated_binary_crossover import sbx
from lib.genetic_operators.mutation.polynomial_mutation import poly_mut

from .chromosomes import ChromosomeP1, Position
from .adapter import ProblemAdapter, Random

import logging
log = logging.getLogger(__name__)

# ============================================================
# Problem 1: Continuous coverage with mobility
# ChromosomeP1: list of N points in Omega (continuous placement)
# ============================================================

class Problem1ContinuousMobilityAdapter(ProblemAdapter):
    """
    Problem 1 adapter.

    ChromosomeP1 representation:
      chromosome := P = [ (x1,y1), ..., (xN,yN) ] where each (xi,yi) in Omega.

    Notes:
    - In practice, Omega is often handled as a bounding box plus optional obstacles.
      Here we treat Omega as a bounding box for sampling/mutation.
    - Feasibility over continuous time is typically approximated by sampling time points.
    """
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        # Valida núcleo homogêneo mínimo
        _ = float(problem["radius_of_reach"])
        _ = float(problem["radius_of_inter"])
        region = problem["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("problem['region'] must be [xmin,ymin,xmax,ymax].")

        # Valida campos específicos do P1
        if "sink" not in problem:
            raise KeyError("Missing 'sink' in P1 problem.")
        if "mobile_nodes" not in problem:
            raise KeyError("Missing 'mobile_nodes' in P1 problem.")
        if "number_of_relays" not in problem:
            raise KeyError("Missing 'number_of_relays' in P1 problem.")
                
        self.problem: ProblemP1 = ProblemP1.cast(problem)

        # Build trajectory coverage constraint once for this problem instance.
        # Uses the region-partition algorithm (W1..W8) for spatial pruning;
        # the sampled points W and their reachable regions are cached here.
        R = self.problem.radius_of_reach
        self._trajectory_constraint = TrajectoryConstraintP1(
            sink=self.problem.sink,
            mobile_nodes=self.problem.mobile_nodes,
            R=R,
            region=self.problem.region,
        )
        log.info(
            "[P1] Trajectory coverage constraint built: %d sampled points, R=%.2f",
            self._trajectory_constraint.n_points, R,
        )

    def coverage_score(self, relays: list[Position]) -> float:
        """Return the trajectory coverage score in [0, 100] (percent) for given relay positions."""
        return self._trajectory_constraint.check_coverage(relays)

    def penalty_objectives(self, chromosome: ChromosomeP1, n_objectives: int) -> list[float] | None:
        """
        Return a minimization-space penalty vector when trajectory coverage
        falls below min_coverage_percentage, or None if feasible.

        The penalty scales with the coverage deficit so that less-covered
        chromosomes are strictly dominated by more-covered ones:

            score    ∈ [0, 100]  (percent)
            min_pct  ∈ [0, 100]  (percent)

            deficit  = (min_pct − score) / 100  ∈ (0, 1]
            penalty  = 1e9 × (1 + deficit)
        """
        score = self.coverage_score(chromosome.relays)
        threshold = self.problem.min_coverage_percentage

        if score >= threshold:
            return None  # feasible — simulate normally

        _PENALTY_BASE = 1e9
        deficit = (threshold - score) / 100.0
        penalty = _PENALTY_BASE * (1.0 + deficit)
        return [penalty] * n_objectives

    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto):
        N = 2 * self.problem.number_of_relays # x and y for each relay  
        self._eta_cx = float(parameters.get("eta_cx", 20.0))
        self._eta_mt = float(parameters.get("eta_mt", 25.0))
        self._per_gene_prob = float(parameters.get("per_gene_prob", 1.0 / N))
        self._crossover_method = str(parameters.get("crossover_method"))
        self._mutation_method = str(parameters.get("mutation_method"))
        self._coverage_repair_budget = int(parameters.get("repair_coverage_budget", parameters.get("coverage_repair_budget", 2)))
        self._coverage_repair_candidates = int(parameters.get("repair_coverage_candidates", parameters.get("coverage_repair_candidates", 4)))
        self._coverage_repair_relay_candidates = int(parameters.get("repair_coverage_relay_candidates", parameters.get("coverage_repair_relay_candidates", 3)))
        self._connectivity_repair_step_ratio = float(parameters.get("connectivity_repair_step_ratio", 0.1))
        self._connectivity_repair_retries = int(parameters.get("connectivity_repair_retries", 2))
        self._rng = rng
                
                
    def random_individual_generator(self, size: int) -> list[ChromosomeP1]:
        box = tuple(self.problem.region) # xmin,ymin,xmax,ymax
        N = self.problem.number_of_relays
        R = self.problem.radius_of_reach
        pop: list[ChromosomeP1] = []
        for i in range(size):
            chrm = ChromosomeP1(
                mac_protocol = self._rng.randint(0, 1),
                relays = self._repair_relays(continuous_network_gen(
                    amount=N, 
                    region=box, 
                    radius=R, 
                    sink=self.problem.sink, 
                    rng=self._rng))
            )
            pop.append(chrm)
        return pop


    def _clip_position(self, pos: Position) -> Position:
        xmin, ymin, xmax, ymax = map(float, self.problem.region)
        x, y = pos
        return (
            min(max(float(x), xmin), xmax),
            min(max(float(y), ymin), ymax),
        )


    def _is_connected_to_sink(self, relays: list[Position]) -> bool:
        return is_connected([self.problem.sink, *relays], self.radius_of_reach)


    def _repair_connectivity_to_sink(self, relays: list[Position]) -> list[Position]:
        if not relays:
            return []

        radius = self.radius_of_reach
        step = max(radius * self._connectivity_repair_step_ratio, radius * 1e-6)
        repaired = [self._clip_position(pos) for pos in relays]

        for _ in range(max(1, self._connectivity_repair_retries)):
            repaired = make_graph_connected_to_sink(repaired, self.problem.sink, radius, step)
            repaired = [self._clip_position(pos) for pos in repaired]
            if self._is_connected_to_sink(repaired):
                return repaired

        log.warning("[P1] Sink-aware connectivity repair failed; regenerating relay network.")
        box = tuple(self.problem.region)
        regenerated = continuous_network_gen(len(relays), box, radius, self.problem.sink, self._rng)
        if len(regenerated) > len(relays):
            regenerated = regenerated[:len(relays)]
        elif len(regenerated) < len(relays):
            regenerated = regenerated + repaired[len(regenerated):]
        regenerated = [self._clip_position(pos) for pos in regenerated]
        if not self._is_connected_to_sink(regenerated):
            regenerated = make_graph_connected_to_sink(regenerated, self.problem.sink, radius, step)
        return [self._clip_position(pos) for pos in regenerated]


    def _top_uncovered_targets(self, relays: list[Position]) -> list[Position]:
        uncovered = self._trajectory_constraint.uncovered_points(relays)
        if not uncovered:
            return []

        radius_sq = self.radius_of_reach * self.radius_of_reach
        scored: list[tuple[int, float, Position]] = []

        for candidate in uncovered:
            cx, cy = candidate
            gain = 0
            for ux, uy in uncovered:
                dx, dy = ux - cx, uy - cy
                if dx * dx + dy * dy <= radius_sq:
                    gain += 1
            scored.append((gain, math.dist(candidate, self.problem.sink), candidate))

        scored.sort(key=lambda item: (-item[0], item[1]))
        limit = max(1, self._coverage_repair_candidates)
        return [candidate for _, _, candidate in scored[:limit]]


    def _project_towards_network(self, target: Position, anchors: list[Position]) -> Position:
        radius = self.radius_of_reach
        anchor = min(anchors, key=lambda pos: math.dist(pos, target))
        distance = math.dist(anchor, target)
        if distance <= radius or distance == 0.0:
            return self._clip_position(target)

        ratio = (radius * 0.98) / distance
        ax, ay = anchor
        tx, ty = target
        return self._clip_position((ax + (tx - ax) * ratio, ay + (ty - ay) * ratio))


    def _greedy_coverage_repair(self, relays: list[Position]) -> list[Position]:
        budget = max(0, self._coverage_repair_budget)
        if budget == 0 or not relays:
            return relays

        threshold = self.problem.min_coverage_percentage
        repaired = relays[:]

        for _ in range(budget):
            current_score = self.coverage_score(repaired)
            if current_score >= threshold:
                break

            targets = self._top_uncovered_targets(repaired)
            if not targets:
                break

            exclusive_counts = self._trajectory_constraint.relay_exclusive_cover_counts(repaired)
            relay_limit = max(1, min(self._coverage_repair_relay_candidates, len(repaired)))
            relay_indices = sorted(
                range(len(repaired)),
                key=lambda idx: (exclusive_counts[idx], -math.dist(repaired[idx], self.problem.sink)),
            )[:relay_limit]

            best_relays: list[Position] | None = None
            best_score = current_score
            best_movement = math.inf

            for idx in relay_indices:
                anchors = [self.problem.sink, *[pos for j, pos in enumerate(repaired) if j != idx]]
                for target in targets:
                    candidates = [
                        self._clip_position(target),
                        self._project_towards_network(target, anchors),
                    ]

                    for candidate in candidates:
                        trial = repaired[:]
                        trial[idx] = candidate
                        trial = self._repair_connectivity_to_sink(trial)
                        score = self.coverage_score(trial)
                        movement = math.dist(repaired[idx], trial[idx])

                        if score > best_score + 1e-9 or (
                            abs(score - best_score) <= 1e-9 and movement < best_movement
                        ):
                            best_relays = trial
                            best_score = score
                            best_movement = movement

            if best_relays is None:
                break

            repaired = best_relays

        return repaired


    def _repair_relays(self, relays: list[Position]) -> list[Position]:
        repaired = self._repair_connectivity_to_sink(relays)
        repaired = self._greedy_coverage_repair(repaired)
        return self._repair_connectivity_to_sink(repaired)

    
    def _sbx_with_radial_translate(
        self, 
        p1: list[Position], 
        p2: list[Position]
    ) -> tuple[list[Position], list[Position]]:      
        c1: list[Position] = []
        c2: list[Position] = []
  
        x_min, y_min, x_max, y_max = tuple(self.problem.region)
        eta = self._eta_cx
        for (x1, y1), (x2, y2) in zip(p1, p2):
            # Apply SBX independently to x and y
            cx1, cx2 = sbx(x1, x2, self._rng, eta, (x_min, x_max))
            cy1, cy2 = sbx(y1, y2, self._rng, eta, (y_min, y_max))

            c1.append((cx1, cy1))
            c2.append((cx2, cy2))
        
        c1_rep = self._repair_relays(c1)
        c2_rep = self._repair_relays(c2)
        
        return c1_rep, c2_rep


    def _rand_network(self) -> tuple[list[Position], list[Position]]:
        box = tuple(self.problem.region) # xmin,ymin,xmax,ymax
        N = self.problem.number_of_relays
        R = self.problem.radius_of_reach
        
        c1: list[Position] = self._repair_relays(continuous_network_gen(N, box, R, self.problem.sink, self._rng))
        c2: list[Position] = self._repair_relays(continuous_network_gen(N, box, R, self.problem.sink, self._rng))
        return c1, c2


    def crossover(self, parents: Sequence[ChromosomeP1]) -> list[ChromosomeP1]:
        assert len(parents) == 2, "P1 crossover requires exactly two parents"

        p1, p2 = parents
        relays1: list[Position] = p1.relays
        relays2: list[Position] = p2.relays

        assert len(relays1) == len(relays2), "Parents must have same number of relays"

        c1_relays: list[Position] = []
        c2_relays: list[Position] = []
  
        cx_method = self._crossover_method.lower()
        if cx_method == 'sbx_with_radial_translate':
            c1_relays, c2_relays = self._sbx_with_radial_translate(relays1, relays2)
        elif cx_method == 'rand' or cx_method=='rand_network':
            c1_relays, c2_relays = self._rand_network()
        else:
            raise ValueError(f"crossover method {self._crossover_method} not found.")

        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP1(mac_protocol=mac1, relays=c1_relays),
            ChromosomeP1(mac_protocol=mac2, relays=c2_relays),
        ]


    def mutate(self, chromosome: ChromosomeP1) -> ChromosomeP1:
        """
        Mutation for Problem P1:
        - Polynomial mutation on relay positions (x, y)
        - Bit-flip mutation on MAC gene
        """
        xmin, ymin, xmax, ymax = map(float, self.problem.region)
        bound_x = (xmin, xmax)
        bound_y = (ymin, ymax)

        new_relays: list[Position] = []

        for (x, y) in chromosome.relays:
            # Mutate x
            if self._rng.random() < self._per_gene_prob:
                x_new = poly_mut(x, self._rng, self._eta_mt, bound_x)
            else:
                x_new = x

            # Mutate y
            if self._rng.random() < self._per_gene_prob:
                y_new = poly_mut(y, self._rng, self._eta_mt, bound_y)
            else:
                y_new = y

            new_relays.append((x_new, y_new))

        new_relays_rep = self._repair_relays(new_relays)
        
        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if self._rng.random() < self._per_gene_prob:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP1(
            mac_protocol=mac,
            relays=new_relays_rep,
        )


    def encode_simulation_input(self, ind: ChromosomeP1) -> SimulationElements:
        fixed: list[FixedMote] = []
        mobile: list[MobileMote] = []

        # -------------------------------------------------
        # Sink σ
        # -------------------------------------------------
        fixed.append({
            "name": "sink",
            "sourceCode": "sink.c",
            "position": list(self.problem.sink),
            "radiusOfReach": self.problem.radius_of_reach,
            "radiusOfInter": self.problem.radius_of_inter,
        })

        # -------------------------------------------------
        # Relays R(ind)
        # -------------------------------------------------
        for i, (x, y) in enumerate(ind.relays):
            fixed.append({
                "name": f"relay_{i}",
                "sourceCode": "node.c",
                "position": [x, y],
                "radiusOfReach": self.problem.radius_of_reach,
                "radiusOfInter": self.problem.radius_of_inter,
            })

        # -------------------------------------------------
        # Mobile Motes Γ
        # -------------------------------------------------
        for i, mobile_node in enumerate(self.problem.mobile_nodes):
            mobile.append({
                "name": f"mobile_{i}",
                "sourceCode": "node.c",
                "functionPath": mobile_node.path_segments,
                "isClosed": mobile_node.is_closed,
                "isRoundTrip": mobile_node.is_round_trip,
                "speed": mobile_node.speed,
                "timeStep": mobile_node.time_step,
                "radiusOfReach": self.problem.radius_of_reach,
                "radiusOfInter": self.problem.radius_of_inter,
            })

        return {
            "fixedMotes": fixed,
            "mobileMotes": mobile,
        }
