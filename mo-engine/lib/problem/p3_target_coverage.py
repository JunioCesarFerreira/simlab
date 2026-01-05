from typing import Any, Mapping, Sequence
import math
import random

from pylib.dto.simulator import FixedMote, SimulationElements
from pylib.dto.problems import ProblemP3
from pylib.dto.algorithm import GeneticAlgorithmConfigDto

from lib.genetic_operators.crossover.uniform_crossover_mask import uniform_crossover_mask
from lib.genetic_operators.mutation.bitflip_mutation import bitflip_mutation

from .adapter import ProblemAdapter, ChromosomeP3


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance in R^2."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


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

    Feasibility (intended):
      - Each target is covered by at least k sensors within R_cov.
      - Each installed sensor has degree at least g in communication graph (within R_com).

    Notes:
    - You can enforce these as hard constraints (constraint_violation > 0 if violated),
      or you can keep them as objectives/penalties depending on your algorithm settings.
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


    def set_ga_operator_configs(self, parameters: GeneticAlgorithmConfigDto): 
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))   
        # Bias toward sparse selections but ensure some minimum
        self._p_on_init = float(parameters.get("p_on_init", 0.15))    
        self._min_on_init = int(parameters.get("min_on_init", 1))
        self._ensure_non_empty = bool(parameters.get("ensure_non_empty", True))


    def random_individual_generator(self, size: int) -> list[ChromosomeP3]:
        Q = self.problem.candidates
        J = len(Q)

        pop: list[ChromosomeP3] = []
        for _ in range(size):
            mask = [1 if random.random() < self._p_on_init else 0 for _ in range(J)]
            while sum(mask) < self._min_on_init:
                mask[random.randrange(J)] = 1
            chrm = ChromosomeP3(
                mac_protocol = random.randint(0,1),
                mask = mask
            )
            pop.append(chrm)
        return pop


    def crossover(self, parents: Sequence[ChromosomeP3]) -> list[ChromosomeP3]:
        p1: ChromosomeP3 = parents[0]
        p2: ChromosomeP3 = parents[1]
        c1, c2 = uniform_crossover_mask(p1.mask, p2.mask)
        
        rng = random.Random()
        
        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP3(mac_protocol=mac1, mask=c1),
            ChromosomeP3(mac_protocol=mac2, mask=c2),
        ]


    def mutate(self, chromosome: ChromosomeP3) -> ChromosomeP3:
        mask: list[int] = chromosome.mask
        out = bitflip_mutation(mask, self._p_bit_mut)
        if sum(out) == 0 and len(out) > 0:
            out[random.randrange(len(out))] = 1
            
        rng = random.Random()
            
        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if rng.random() < self._p_bit_mut:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP3(
            mac_protocol=mac,
            mask=out,
        )


    def _compute_constraint_violation_static(self, mask: list[int]) -> float:
        """
        Compute a static (geometry-only) feasibility penalty:
        - k-coverage of targets within R_cov
        - g-min-degree among installed sensors within R_com

        This does not include MAC/interference/traffic, which can be handled by simulation.
        """
        Q = self.problem.candidates
        Xi = self.problem.targets
        k = self.problem.k_required
        g = self.problem.g_required
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
        }
