from typing import Any, Mapping, Sequence, cast
import math
import random

from pylib.dto.simulation import SimulationElements
from pylib.dto.problems import ProblemP3
from .adapter import ProblemAdapter, ChromosomeP3


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance in R^2."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


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
                
        self.problem = cast(ProblemP3, problem)


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

    def random_individual_generator(self, size: int) -> list[ChromosomeP3]:
        Q = self._Q()
        J = len(Q)
        # Bias toward sparse selections but ensure some minimum
        p_on = float(self.problem.get("problem_parameters", {}).get("p_on_init", 0.2))
        min_on = int(self.problem.get("problem_parameters", {}).get("min_on_init", 1))

        pop: list[ChromosomeP3] = []
        for _ in range(size):
            mask = [1 if random.random() < p_on else 0 for _ in range(J)]
            while sum(mask) < min_on:
                mask[random.randrange(J)] = 1
            pop.append(ChromosomeP3(chromosome=mask))
        return pop

    def crossover(self, parents: Sequence[ChromosomeP3]) -> list[ChromosomeP3]:
        m1: list[int] = parents[0]
        m2: list[int] = parents[1]
        c1, c2 = _uniform_crossover_mask(m1, m2)
        return [c1, c2]

    def mutate(self, chromosome: ChromosomeP3) -> ChromosomeP3:
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

    def encode_simulation_input(self, ind: ChromosomeP3) -> SimulationElements:
        raise NotImplementedError

