from typing import Any, Mapping, Sequence

from pylib.dto.simulator import FixedMote, MobileMote, SimulationElements
from pylib.dto.problems import ProblemP1
from pylib.dto.algorithm import GeneticAlgorithmConfigDto

from lib.util.random_network import continuous_network_gen
from lib.util.connectivity import make_graph_connected
from lib.genetic_operators.crossover.simulated_binary_crossover import sbx
from lib.genetic_operators.mutation.polynomial_mutation import poly_mut

from .chromosomes import ChromosomeP1, Position
from .adapter import ProblemAdapter, Random

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
    
    
    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto):
        N = 2 * self.problem.number_of_relays # x and y for each relay  
        self._eta_cx = float(parameters.get("eta_cx", 20.0))
        self._eta_mt = float(parameters.get("eta_mt", 25.0))
        self._per_gene_prob = float(parameters.get("per_gene_prob", 1.0 / N))
        self._crossover_method = str(parameters.get("crossover_method"))
        self._mutation_method = str(parameters.get("mutation_method"))
        self._rng = rng
                
                
    def random_individual_generator(self, size: int) -> list[ChromosomeP1]:
        box = tuple(self.problem.region) # xmin,ymin,xmax,ymax
        N = self.problem.number_of_relays
        R = self.problem.radius_of_reach
        pop: list[ChromosomeP1] = []
        for i in range(size):
            chrm = ChromosomeP1(
                mac_protocol = self._rng.randint(0, 1),
                relays = continuous_network_gen(N, box, R, self._rng)
            )
            pop.append(chrm)
        return pop

    
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
        
        # makes graph connected by radial contraction
        radius = self.radius_of_reach
        step = radius * 0.1
        c1 = make_graph_connected(c1, radius, step)
        c2 = make_graph_connected(c2, radius, step)
        
        return c1, c2


    def _rand_network(self) -> tuple[list[Position], list[Position]]:
        box = tuple(self.problem.region) # xmin,ymin,xmax,ymax
        N = self.problem.number_of_relays
        R = self.problem.radius_of_reach
        
        c1: list[Position] = continuous_network_gen(N, box, R, self._rng)
        c2: list[Position] = continuous_network_gen(N, box, R, self._rng)        
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

        # makes graph connected by radial contraction
        radius = self.radius_of_reach
        step = radius * 0.1
        new_relays = make_graph_connected(new_relays, radius, step)
        
        # MAC mutation (bit-flip)
        mac = chromosome.mac_protocol
        if self._rng.random() < self._per_gene_prob:
            mac = 1 - mac  # 0 ↔ 1

        return ChromosomeP1(
            mac_protocol=mac,
            relays=new_relays,
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