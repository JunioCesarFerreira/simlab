from typing import Any, Mapping, Sequence

from pylib.config.simulator import SimulationElements
from pylib.config.algorithm import GeneticAlgorithmConfigDto

from lib.genetic_operators.crossover.simulated_binary_crossover import sbx
from lib.genetic_operators.mutation.polynomial_mutation import poly_mut

from .chromosomes import ChromosomeP0
from .adapter import ProblemAdapter, Random

import logging
log = logging.getLogger(__name__)

# ============================================================
# Problem 0: Pure synthetic benchmark (algorithm validation)
# ChromosomeP0: real-valued decision vector x ∈ [0,1]^n
# ============================================================


class Problem0SyntheticAdapter(ProblemAdapter):
    """
    Problem 0 adapter — pure analytical benchmark problem.

    Purpose
    -------
    Validate the *algorithmic* behaviour of the multi-objective optimizers
    (NSGA-II / NSGA-III and variants) against classical test problems
    (DTLZ2, ZDT1, SCH1), decoupled from the WSN topology machinery of P1–P4.

    Representation
    --------------
    chromosome := x = [x_0, ..., x_{n-1}] with each x_i ∈ [0, 1].

    The decision vector is a *pure* GA genotype: there is no MAC gene, no sink,
    no relays and no connectivity/coverage repair. Genetic operators are the
    textbook real-coded pair — Simulated Binary Crossover (SBX) and Polynomial
    Mutation — applied independently per variable over the unit interval.

    Evaluation
    ----------
    The benchmark function itself is evaluated downstream (master-node) directly
    on ``x`` — see ``encode_simulation_input``, which exposes the decision vector
    verbatim under ``simulationElements.decisionVector`` (no coordinate
    round-trip, no region scaling).
    """

    # Decision variables live in the unit hypercube [0, 1]^n. These bounds are
    # the canonical domain of the normalised DTLZ2/ZDT1/SCH1 benchmarks.
    _LB: float = 0.0
    _UB: float = 1.0

    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        n = problem.get("n", problem.get("number_of_variables"))
        if n is None:
            raise KeyError("Missing 'n' (number of decision variables) in P0 problem.")
        n = int(n)
        if n < 1:
            raise ValueError(f"P0 requires n >= 1 decision variables, got {n}.")
        self._n: int = n
        # Formal unit-box region: analytical evaluation is bound-free, but the
        # SimulationConfig schema still carries a region for every problem.
        self._region: list[float] = [self._LB, self._LB, self._UB, self._UB]

    # ------------------------------------------------------------------
    # In-process (analytical) evaluation — P0 is evaluated closed-form.
    # ------------------------------------------------------------------
    @property
    def is_analytical(self) -> bool:
        return True

    def decision_vector(self, ind: ChromosomeP0) -> list[float]:
        return list(ind.x)

    # ------------------------------------------------------------------
    # Structural properties — overridden: P0 has no WSN radius/region.
    # ------------------------------------------------------------------
    @property
    def radius_of_reach(self) -> float:
        return 1.0

    @property
    def radius_of_inter(self) -> float:
        return 1.0

    @property
    def bounds(self) -> list[float]:
        return list(self._region)

    # ------------------------------------------------------------------
    # Genetic algorithm configuration
    # ------------------------------------------------------------------
    # Operators are fixed by design: textbook SBX + polynomial mutation.
    CONSUMED_GA_KEYS = frozenset({"eta_cx", "eta_mt", "per_gene_prob"})

    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto) -> None:
        self._eta_cx = float(parameters.get("eta_cx", 20.0))
        self._eta_mt = float(parameters.get("eta_mt", 20.0))
        # Standard per-gene mutation probability defaults to 1/n.
        self._per_gene_prob = float(parameters.get("per_gene_prob", 1.0 / self._n))
        self._rng = rng

    # ------------------------------------------------------------------
    # Initial population — uniform sampling over [0,1]^n (no bias).
    # ------------------------------------------------------------------
    def random_individual_generator(self, size: int) -> list[ChromosomeP0]:
        return [
            ChromosomeP0(x=[self._rng.uniform(self._LB, self._UB) for _ in range(self._n)])
            for _ in range(size)
        ]

    # ------------------------------------------------------------------
    # Genetic operators — textbook SBX + polynomial mutation, no repair.
    # ------------------------------------------------------------------
    def crossover(self, parents: Sequence[ChromosomeP0]) -> list[ChromosomeP0]:
        assert len(parents) == 2, "P0 crossover requires exactly two parents"
        p1, p2 = parents
        assert len(p1.x) == len(p2.x), "Parents must have the same number of variables"

        bounds = (self._LB, self._UB)
        c1x: list[float] = []
        c2x: list[float] = []
        for a, b in zip(p1.x, p2.x):
            ca, cb = sbx(a, b, self._rng, self._eta_cx, bounds)
            c1x.append(ca)
            c2x.append(cb)

        return [ChromosomeP0(x=c1x), ChromosomeP0(x=c2x)]

    def mutate(self, chromosome: ChromosomeP0) -> ChromosomeP0:
        bounds = (self._LB, self._UB)
        new_x: list[float] = []
        for v in chromosome.x:
            if self._rng.random() < self._per_gene_prob:
                new_x.append(poly_mut(v, self._rng, self._eta_mt, bounds))
            else:
                new_x.append(v)
        return ChromosomeP0(x=new_x)

    # ------------------------------------------------------------------
    # Integration with SimLab — decision vector exposed verbatim.
    # ------------------------------------------------------------------
    def encode_simulation_input(self, ind: ChromosomeP0) -> SimulationElements:
        # The benchmark is evaluated analytically on the decision vector itself;
        # there are no motes. The vector is carried under 'decisionVector' so the
        # master-node reads it directly, without any physical-coordinate scaling.
        return {
            "fixedMotes": [],
            "mobileMotes": [],
            "decisionVector": list(ind.x),
        }
