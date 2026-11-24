from typing import Any, Mapping, Sequence
import math
import logging

from pylib.config.simulator import FixedMote, SimulationElements
from pylib.config.problems import ProblemP3
from pylib.config.algorithm import GeneticAlgorithmConfigDto

from lib.util.random_network import stochastic_reachability_mask
from lib.util.connectivity import repair_connectivity_to_sink, repair_k_coverage

from lib.genetic_operators.crossover.uniform_crossover_mask import uniform_crossover_mask
from lib.genetic_operators.mutation.bitflip_mutation import bitflip_mutation

from .adapter import ProblemAdapter, ChromosomeP3, Random

log = logging.getLogger(__name__)

# Same base as P1 and P2: above any measured objective, below the refusal
# placeholder, so a penalised individual never reaches front 0.
_PENALTY_BASE = 1e9


# ============================================================
# Problem 3: Sensing coverage with targets
# ChromosomeP3: binary mask over candidate positions Q
# Constraint-style feasibility: k-coverage of targets and g-min-degree connectivity
# ============================================================

class Problem3TargetCoverageAdapter(ProblemAdapter):
    """
    Problem 3 adapter.

    ChromosomeP3 representation:
      chromosome := mask in {0,1}^J selecting subset P ⊆ Q.

    Feasibility:
      - Every installed sensor reaches the sink over the communication graph
        (within R_com).
      - Each target is covered by at least k sensors within R_cov.
      - Each installed sensor has at least g neighbours within R_com, the sink
        included (``g_required``, default 1).

    The operators repair k-coverage, connectivity and the degree; what still
    violates a constraint is penalised instead of simulated.
    """
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        # Valida núcleo homogêneo mínimo
        _ = float(problem["radius_of_reach"])
        _ = float(problem["radius_of_inter"])
        region = problem["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("problem['region'] must be [xmin,ymin,xmax,ymax].")

        # Valida campos específicos do P3
        if "sink" not in problem:
            raise KeyError("Missing 'sink' in P3 problem.")
        if "targets" not in problem:
            raise KeyError("Missing 'targets' in P3 problem.")
        if "candidates" not in problem:
            raise KeyError("Missing 'candidates' in P3 problem.")
        if "radius_of_cover" not in problem:
            raise KeyError("Missing 'radius_of_cover' in P3 problem.")
        if "k_required" not in problem:
            raise KeyError("Missing 'k_required' in P3 problem.")
                
        self.problem: ProblemP3 = ProblemP3.cast(problem)


    # Operators are fixed by design: uniform mask crossover + bit-flip mutation,
    # both followed by the k-coverage and connectivity repairs.
    CONSUMED_GA_KEYS = frozenset({"per_gene_prob"})

    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto):
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))
        self._rng = rng


    def random_individual_generator(self, size: int) -> list[ChromosomeP3]:
        """
        Generate ``size`` individuals, retrying for feasibility and uniqueness.

        When the feasible subspace (k-coverage + connectivity) is small, the
        rejection loop below can keep landing on the same mask, producing
        duplicate chromosomes far more often than in an unconstrained
        discrete space. Callers (e.g. NSGA-II's generation enqueue)
        de-duplicate genomes within a generation, so undetected duplicates
        here would silently shrink the initial population below ``size``.
        Retry a bounded number of times per slot to keep the population at
        full size whenever the search space allows it.
        """
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach

        pop: list[ChromosomeP3] = []
        seen: set[str] = set()
        max_attempts = 20

        for _ in range(size):
            chrm = None
            for attempt in range(max_attempts):
                mask = stochastic_reachability_mask(Q, S, R, self._rng)
                candidate = ChromosomeP3(
                    mac_protocol=self._rng.randint(0, 1),
                    mask=mask
                )
                if self._is_feasible_static(mask) and candidate.get_hash() not in seen:
                    chrm = candidate
                    break
            if chrm is None:
                # Fallback: no feasible-and-unique mask found in `max_attempts`
                # tries (feasible subspace too small, or infeasible search
                # space). Accept the last draw rather than shrinking the
                # population; if it is infeasible it is penalised.
                log.warning(
                    "[P3] Random gen fallback: could not produce a feasible, "
                    "unique individual after %d attempts.", max_attempts,
                )
                chrm = candidate
            seen.add(chrm.get_hash())
            pop.append(chrm)

        return pop


    def _repair_degree(self, mask: list[int]) -> list[int]:
        """Switch on, for each mote with fewer than g neighbours, its nearest
        candidates within reach, until it has g or none is left.

        A candidate switched on lies within reach of a connected mote, so the
        network stays connected and covered; it may lack neighbours itself, and
        the loop goes on until nothing changes.
        """
        g = self.problem.g_required
        if g <= 1:
            return mask
        Q = [tuple(q) for q in self.problem.candidates]
        sink = tuple(self.problem.sink)
        radius = self.problem.radius_of_reach
        mask = list(mask)

        def degree(i: int) -> int:
            neighbours = sum(1 for j, bit in enumerate(mask)
                             if bit == 1 and j != i and math.dist(Q[i], Q[j]) <= radius)
            return neighbours + (1 if math.dist(Q[i], sink) <= radius else 0)

        changed = True
        while changed:
            changed = False
            for i, bit in enumerate(mask):
                if bit != 1:
                    continue
                missing = g - degree(i)
                if missing <= 0:
                    continue
                spare = sorted(
                    (j for j, b in enumerate(mask) if b == 0 and math.dist(Q[i], Q[j]) <= radius),
                    key=lambda j: math.dist(Q[i], Q[j]))
                for j in spare[:missing]:
                    mask[j] = 1
                    changed = True
        return mask


    def _repair(self, mask: list[int], fallback: list[int]) -> list[int]:
        """k-coverage, then connectivity to the sink, then the degree g.

        Each only switches candidates on, so a later one cannot undo an earlier
        one. In the opposite order the k-coverage repair could switch on a
        sensor out of reach and disconnect the network the first had just joined.
        """
        covered = repair_k_coverage(
            self.problem.candidates,
            self.problem.targets,
            mask,
            self.problem.radius_of_cover,
            self.problem.k_required,
        )
        err, connected = repair_connectivity_to_sink(
            self.problem.candidates, covered, self.problem.sink, self.problem.radius_of_reach)
        if err:
            log.error("[P3] Connectivity repair failed; keeping the parent's mask.")
            return list(fallback)
        return self._repair_degree(connected)


    def crossover(self, parents: Sequence[ChromosomeP3]) -> list[ChromosomeP3]:
        p1: ChromosomeP3 = parents[0]
        p2: ChromosomeP3 = parents[1]
        c1, c2 = uniform_crossover_mask(p1.mask, p2.mask, self._rng)
        c1 = self._repair(c1, p1.mask)
        c2 = self._repair(c2, p2.mask)

        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP3(mac_protocol=mac1, mask=c1),
            ChromosomeP3(mac_protocol=mac2, mask=c2),
        ]


    def mutate(self, chromosome: ChromosomeP3) -> ChromosomeP3:
        mask: list[int] = chromosome.mask
        bitflip_result = bitflip_mutation(mask, self._p_bit_mut, self._rng)
        out = self._repair(bitflip_result, mask)

        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if self._rng.random() < self._p_bit_mut:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP3(
            mac_protocol=mac,
            mask=out,
        )


    def constraint_deficits(self, mask: list[int]) -> tuple[float, float, float]:
        """How far ``mask`` is from each constraint, each in [0, 1], all 0 when feasible.

        - connectivity: share of the installed sensors that cannot reach the sink;
        - k-coverage: missing covers summed over the targets, over ``k × |targets|``;
        - degree: missing neighbours summed over the sensors, over ``g × |sensors|``.
        """
        active = [q for q, bit in zip(self.problem.candidates, mask) if bit == 1]
        nodes = [tuple(self.problem.sink), *map(tuple, active)]  # 0 is the sink
        radius = self.problem.radius_of_reach
        neighbours = [
            [j for j in range(len(nodes)) if j != i and math.dist(nodes[i], nodes[j]) <= radius]
            for i in range(len(nodes))
        ]
        reached, stack = {0}, [0]
        while stack:
            for j in neighbours[stack.pop()]:
                if j not in reached:
                    reached.add(j)
                    stack.append(j)

        sensors = len(active)
        connectivity = (sensors + 1 - len(reached)) / sensors if sensors else 0.0

        k = self.problem.k_required
        targets = self.problem.targets
        missing = sum(
            max(0, k - sum(1 for p in active if math.dist(p, t) <= self.problem.radius_of_cover))
            for t in targets)
        coverage = missing / (k * len(targets)) if targets else 0.0

        g = self.problem.g_required
        lacking = sum(max(0, g - len(neighbours[i])) for i in range(1, len(nodes)))
        degree = lacking / (g * sensors) if sensors else 0.0
        return connectivity, coverage, degree


    def penalty_objectives(self, chromosome: ChromosomeP3, n_objectives: int) -> list[float] | None:
        """
        A minimization-space penalty when ``chromosome`` violates a constraint,
        or ``None`` when it is feasible and is to be simulated.

          deficit = mean of the three deficits  ∈ (0, 1]
          penalty = 1e9 × (1 + deficit)

        As in P1 and P2: above every measurement, and larger the further the
        individual is from feasibility.
        """
        deficits = self.constraint_deficits(chromosome.mask)
        if not any(deficits):
            return None
        deficit = sum(deficits) / len(deficits)
        return [_PENALTY_BASE * (1.0 + deficit)] * n_objectives


    def _is_feasible_static(self, mask: list[int]) -> bool:
        return not any(self.constraint_deficits(mask))


    def encode_simulation_input(self, ind: ChromosomeP3) -> SimulationElements:
        fixed: list[FixedMote] = []

        # -------------------------------------------------
        # Structural checks
        # -------------------------------------------------
        if len(ind.mask) != len(self.problem.candidates):
            raise ValueError(
                f"Chromosome length ({len(ind.mask)}) does not match "
                f"number of candidates ({len(self.problem.candidates)})"
            )

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
        # Selected Relays R(ind) ⊆ Q
        # -------------------------------------------------
        for idx, (bit, position) in enumerate(zip(ind.mask, self.problem.candidates)):
            if bit == 1:
                fixed.append({
                    "name": f"relay_{idx}",
                    "sourceCode": "node.c",
                    "position": list(position),
                    "radiusOfReach": self.problem.radius_of_reach,
                    "radiusOfInter": self.problem.radius_of_inter,
                })

        return {
            "fixedMotes": fixed,
            "mobileMotes": [],
            "targets": [list(t) for t in self.problem.targets],
        }
