from typing import Any, Mapping, Sequence
import random

from pylib.dto.simulator import FixedMote, MobileMote, SimulationElements
from pylib.dto.problems import ProblemP2
from pylib.dto.algorithm import GeneticAlgorithmConfigDto

from lib.genetic_operators.crossover.uniform_crossover_mask import uniform_crossover_mask
from lib.genetic_operators.mutation.bitflip_mutation import bitflip_mutation

from .adapter import ProblemAdapter, ChromosomeP2



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
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        # Checks homogeneous core
        _ = float(problem["radius_of_reach"])
        _ = float(problem["radius_of_inter"])
        region = problem["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("problem['region'] must be [xmin,ymin,xmax,ymax].")

        # Checks specifics of P2
        if "sink" not in problem:
            raise KeyError("Missing 'sink' in P2 problem.")
        if "mobile_nodes" not in problem:
            raise KeyError("Missing 'mobile_nodes' in P2 problem.")
        if "candidates" not in problem:
            raise KeyError("Missing 'candidates' in P2 problem.")
                
        self.problem: ProblemP2 = ProblemP2.cast(problem)

    def random_individual_generator(self, size: int) -> list[ChromosomeP2]:
        Q = self.problem.candidates
        J = len(Q)
        # Bias toward sparse selections (since the primary goal is to minimize |P|)
        p_on = float(self._p_on_init)

        pop: list[ChromosomeP2] = []
        for _ in range(size):
            mask = [1 if random.random() < p_on else 0 for _ in range(J)]
            # Ensure not empty (optional)
            if sum(mask) == 0:
                mask[random.randrange(J)] = 1
            chrm = ChromosomeP2(                
                mac_protocol = random.randint(0, 1),
                mask=mask
            )
            pop.append(chrm)
        return pop
    
    def set_ga_operator_configs(self, parameters: GeneticAlgorithmConfigDto):    
        self._p_on_init = float(parameters.get("p_on_init", 0.15))    
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))
        self._ensure_non_empty = bool(parameters.get("ensure_non_empty", True))
        
    def crossover(self, parents: Sequence[ChromosomeP2]) -> list[ChromosomeP2]:
        p1: ChromosomeP2 = parents[0]
        p2: ChromosomeP2 = parents[1]
        c1, c2 = uniform_crossover_mask(p1.mask, p2.mask)
        
        rng = random.Random()
        
        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP2(mac_protocol=mac1, mask=c1),
            ChromosomeP2(mac_protocol=mac2, mask=c2),
        ]

    def mutate(self, chromosome: ChromosomeP2) -> ChromosomeP2:
        mask: list[int] = chromosome.mask
        out = bitflip_mutation(mask, self._p_bit_mut)
        # Keep at least one selected position (optional)
        if sum(out) == 0 and len(out) > 0:
            out[random.randrange(len(out))] = 1
            
        rng = random.Random()
            
        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if rng.random() < self._p_bit_mut:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP2(
            mac_protocol=mac,
            mask=out,
        )

    def encode_simulation_input(self, ind: ChromosomeP2) -> SimulationElements:
        fixed: list[FixedMote] = []
        mobile: list[MobileMote] = []

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



