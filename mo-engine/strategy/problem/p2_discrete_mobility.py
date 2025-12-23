from typing import Any, Mapping, Sequence, cast
import random

from pylib.dto.simulation import SimulationElements
from pylib.dto.problems import ProblemP2
from .adapter import ProblemAdapter, ChromosomeP2

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
        # Valida núcleo homogêneo mínimo
        _ = float(problem["radius_of_reach"])
        _ = float(problem["radius_of_inter"])
        region = problem["region"]
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            raise ValueError("problem['region'] must be [xmin,ymin,xmax,ymax].")

        # Valida campos específicos do P2
        if "sink" not in problem:
            raise KeyError("Missing 'sink' in P2 problem.")
        if "mobile_nodes" not in problem:
            raise KeyError("Missing 'mobile_nodes' in P2 problem.")
        if "candidates" not in problem:
            raise KeyError("Missing 'candidates' in P2 problem.")
                
        self.problem = cast(ProblemP2, problem)

    def random_individual_generator(self, size: int) -> list[ChromosomeP2]:
        Q = self.problem["candidates"]
        J = len(Q)
        # Bias toward sparse selections (since the primary goal is to minimize |P|)
        p_on = float(self.problem.get("problem_parameters", {}).get("p_on_init", 0.15))

        pop: list[ChromosomeP2] = []
        for _ in range(size):
            mask = [1 if random.random() < p_on else 0 for _ in range(J)]
            # Ensure not empty (optional)
            if sum(mask) == 0:
                mask[random.randrange(J)] = 1
            pop.append(ChromosomeP2(chromosome=mask))
        return pop

    def crossover(self, parents: Sequence[ChromosomeP2]) -> list[ChromosomeP2]:
        m1: list[int] = parents[0]
        m2: list[int] = parents[1]
        c1, c2 = _uniform_crossover_mask(m1, m2)
        return [c1, c2]

    def mutate(self, chromosome: ChromosomeP2) -> ChromosomeP2:
        mask: list[int] = chromosome
        p_bit = float(self.problem.get("problem_parameters", {}).get("p_bit_mut", 0.01))
        out = _bitflip_mutation(mask, p_bit)
        # Keep at least one selected position (optional)
        if sum(out) == 0 and len(out) > 0:
            out[random.randrange(len(out))] = 1
        return out

    def encode_simulation_input(self, ind: ChromosomeP2) -> SimulationElements:
        raise NotImplementedError

