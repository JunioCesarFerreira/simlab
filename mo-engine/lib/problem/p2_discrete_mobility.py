from typing import Any, Mapping, Sequence
import logging

from pylib.dto.simulator import FixedMote, MobileMote, SimulationElements
from pylib.dto.problems import ProblemP2
from pylib.dto.algorithm import GeneticAlgorithmConfigDto

from lib.util.random_network import stochastic_reachability_mask
from lib.util.connectivity import repair_connectivity_to_sink

from lib.genetic_operators.crossover.uniform_crossover_mask import uniform_crossover_mask
from lib.genetic_operators.mutation.bitflip_mutation import bitflip_mutation

from .adapter import ProblemAdapter, ChromosomeP2, Random

log = logging.getLogger(__name__)

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


    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto):    
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))
        self._rng = rng


    def random_individual_generator(self, size: int) -> list[ChromosomeP2]:
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach

        pop: list[ChromosomeP2] = []
        for _ in range(size):
            mask = stochastic_reachability_mask(Q, S, R, self._rng)
            chrm = ChromosomeP2(                
                mac_protocol = self._rng.randint(0, 1),
                mask=mask
            )
            pop.append(chrm)            
        return pop
                
        
    def crossover(self, parents: Sequence[ChromosomeP2]) -> list[ChromosomeP2]:
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach
        p1: ChromosomeP2 = parents[0]
        p2: ChromosomeP2 = parents[1]
        c1, c2 = uniform_crossover_mask(p1.mask, p2.mask)
        
        err, c1 = repair_connectivity_to_sink(Q, c1, S, R)
        if err:
            log.error(f"[P2] Repair failed. c1 crossover.")
            c1 = p1.mask
        err, c2 = repair_connectivity_to_sink(Q, c2, S, R)
        if err:
            log.error(f"[P2] Repair failed. c2 crossover.")
            c2 = p2.mask
                
        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP2(mac_protocol=mac1, mask=c1),
            ChromosomeP2(mac_protocol=mac2, mask=c2),
        ]


    def mutate(self, chromosome: ChromosomeP2) -> ChromosomeP2:
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach
        mask: list[int] = chromosome.mask
        bitflip_result = bitflip_mutation(mask, self._p_bit_mut)
        
        err, out = repair_connectivity_to_sink(Q, bitflip_result, S, R)
        if err:
            log.error(f"[P2] Repair failed. mutation.")
            out = mask
                    
        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if self._rng.random() < self._p_bit_mut:
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