from typing import Any, Mapping, Sequence, cast
import random

from strategy.util.genetic_operators.crossover import make_sbx_crossover
from strategy.util.genetic_operators.mutation import make_polynomial_mutation

from pylib.dto.simulator import SimulationElements
from pylib.dto.problems import ProblemP1
from .chromosomes import ChromosomeP1
from .adapter import ProblemAdapter

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
                
        self.problem = cast(ProblemP1, problem)
                
                
    def _rand_uniform_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
        """Sample a point uniformly in an axis-aligned bounding box (xmin, ymin, xmax, ymax)."""
        xmin, ymin, xmax, ymax = box
        return (random.uniform(xmin, xmax), random.uniform(ymin, ymax))

    def random_individual_generator(self, size: int) -> list[ChromosomeP1]:
        box = tuple(self.problem["region"]) # xmin,ymin,xmax,ymax
        N = self.problem["number_of_relays"]
        pop: list[ChromosomeP1] = []
        for _ in range(size):
            P = [self._rand_uniform_point(box) for _ in range(N)]
            pop.append(ChromosomeP1(P))
        return pop
    
    
    def set_ga_parameters(self, parameters):
        N = self.problem["number_of_relays"]
        def gene_bounds() -> list[tuple[float, float]]:
            x1, y1, x2, y2 = tuple(self.problem["region"])
            bounds: list[tuple[float, float]] = []
            for _ in range(N):
                bounds.append((x1, x2))  # x
                bounds.append((y1, y2))  # y
            return bounds
        
        bounds = gene_bounds()
        eta_cx = float(parameters.get("eta_cx", 20.0))
        eta_mt = float(parameters.get("eta_mt", 25.0))
        pgp = float(parameters.get("per_gene_prob", per_gene_prob=1.0 / (2 * N)))
        self._sbx = make_sbx_crossover(eta=eta_cx, bounds=bounds)
        self._poly = make_polynomial_mutation(eta=eta_mt, bounds=bounds, per_gene_prob= pgp)


    def crossover(self, parents: Sequence[ChromosomeP1]) -> list[ChromosomeP1]:
        """
        Crossover for continuous placements:
        - pairwise blend crossover on each corresponding sensor position.
        """
        P1: list[tuple[float, float]] = parents[0]
        P2: list[tuple[float, float]] = parents[1]
        assert len(P1) == len(P2)
        rng = random.Random()
        self._sbx(P1, P2, rng)
    

    def mutate(self, chromosome: ChromosomeP1) -> ChromosomeP1:
        """
        Mutation for continuous placements:
        - Gaussian perturbation of a subset of points.
        """
        P: list[tuple[float, float]] = chromosome        
        rng = random.Random()        
        return self._poly(P, rng)


    def encode_simulation_input(self, ind: ChromosomeP1) -> SimulationElements:
        raise NotImplementedError
