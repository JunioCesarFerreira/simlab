from typing import Any, Mapping, Sequence
import math
import logging

from pylib.dto.simulator import FixedMote, SimulationElements
from pylib.dto.problems import ProblemP3
from pylib.dto.algorithm import GeneticAlgorithmConfigDto

from lib.util.random_network import stochastic_reachability_mask
from lib.util.connectivity import repair_connectivity_to_sink, repair_k_coverage
from lib.util.connectivity import is_connected_and_k_covered

from lib.genetic_operators.crossover.uniform_crossover_mask import uniform_crossover_mask
from lib.genetic_operators.mutation.bitflip_mutation import bitflip_mutation

from .adapter import ProblemAdapter, ChromosomeP3, Random

log = logging.getLogger(__name__)


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


    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto): 
        self._p_bit_mut = float(parameters.get("per_gene_prob", 0.1))
        self._rng = rng


    def random_individual_generator(self, size: int) -> list[ChromosomeP3]:
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach

        pop: list[ChromosomeP3] = []

        max_attempts = 20

        for _ in range(size):
            for _ in range(max_attempts):
                mask = stochastic_reachability_mask(Q, S, R, self._rng)

                if self._is_feasible_static(mask):
                    break
            else:
                # fallback: Accept even if unfeasible (GA corrects later)
                log.warning("[P3] Random gen fallback: infeasible individual")

            pop.append(
                ChromosomeP3(
                    mac_protocol=self._rng.randint(0, 1),
                    mask=mask
                )
            )

        return pop


    def crossover(self, parents: Sequence[ChromosomeP3]) -> list[ChromosomeP3]:
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach
        p1: ChromosomeP3 = parents[0]
        p2: ChromosomeP3 = parents[1]
        c1, c2 = uniform_crossover_mask(p1.mask, p2.mask)
        
        err, c1 = repair_connectivity_to_sink(Q, c1, S, R)
        if err:
            log.error(f"[P2] Repair failed. c1 crossover.")
            c1 = p1.mask
        else:            
            c1 = repair_k_coverage(
                Q,
                self.problem.targets,
                c1,
                self.problem.radius_of_cover,
                self.problem.k_required
            )
            
        err, c2 = repair_connectivity_to_sink(Q, c2, S, R)
        if err:
            log.error(f"[P2] Repair failed. c2 crossover.")
            c2 = p2.mask
        else:                
            c2 = repair_k_coverage(
                Q,
                self.problem.targets,
                c2,
                self.problem.radius_of_cover,
                self.problem.k_required
            )
                
        # MAC gene inheritance (simple uniform choice)
        mac1 = p1.mac_protocol if self._rng.random() < 0.5 else p2.mac_protocol
        mac2 = p2.mac_protocol if self._rng.random() < 0.5 else p1.mac_protocol

        return [
            ChromosomeP3(mac_protocol=mac1, mask=c1),
            ChromosomeP3(mac_protocol=mac2, mask=c2),
        ]


    def mutate(self, chromosome: ChromosomeP3) -> ChromosomeP3:
        Q = self.problem.candidates
        S = self.problem.sink
        R = self.problem.radius_of_reach
        mask: list[int] = chromosome.mask
        bitflip_result = bitflip_mutation(mask, self._p_bit_mut)
        
        err, out = repair_connectivity_to_sink(Q, bitflip_result, S, R)
        if err:
            log.error(f"[P2] Repair failed. mutation.")
            out = mask
        else:                    
            out = repair_k_coverage(
                Q,
                self.problem.targets,
                out,
                self.problem.radius_of_cover,
                self.problem.k_required
            )
                    
        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if self._rng.random() < self._p_bit_mut:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP3(
            mac_protocol=mac,
            mask=out,
        )


    def _is_feasible_static(self, mask: list[int]) -> bool:
        Q = self.problem.candidates
        P = [Q[i] for i, b in enumerate(mask) if b == 1]

        if not P:
            return False

        return is_connected_and_k_covered(
            sensor_positions=P,
            target_positions=self.problem.targets,
            communication_radius=self.problem.radius_of_reach,
            sensing_radius=self.problem.radius_of_cover,
            k=self.problem.k_required,
        )


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
