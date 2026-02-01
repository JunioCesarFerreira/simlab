import random
from abc import ABC, abstractmethod
from .chromosomes import (
    ChromosomeP1,
    ChromosomeP2,
    ChromosomeP3,
    ChromosomeP4
)
from typing import Any, Mapping, Sequence
from pylib.dto.simulator import SimulationElements
from pylib.dto.algorithm import GeneticAlgorithmConfigDto

# Type aliases for clarity
Chromosome = ChromosomeP1 | ChromosomeP2 | ChromosomeP3 | ChromosomeP4

Random = random.Random

class ProblemAdapter(ABC):
    """
    Adapter that encapsulates all problem-specific logic.

    This class is responsible for:
    - defining the chromosome representation
    - generating the initial population
    - implementing genetic operators (crossover, mutation)
    - evaluating individuals (analytically or via simulation)
    - encoding/decoding Simulation documents for the SimLab workflow
    """

    def __init__(self, problem: Mapping[str, Any]) -> None:
        """
        Parameters
        ----------
        problem:
            Full problem document (or DTO) as loaded from MongoDB.
            It should include:
            - problem_type / problem_parameters
            - strategy / nsga3_parameters
            - any additional fields needed by the adapter
        """
        self.assert_problem(problem) # Validate and store the problem

    # ------------------------------------------------------------------
    # Structural information about the problem base
    # ------------------------------------------------------------------  
    @property
    def radius_of_reach(self) -> float:
        value = self.problem.radius_of_reach
        if value is None:
            raise KeyError("Missing 'radius_of_reach' in problem.")
        return float(value)
    
    @property
    def radius_of_inter(self) -> float:
        value = self.problem.radius_of_inter
        if value is None:
            raise KeyError("Missing 'radius_of_inter' in problem.")
        return float(value)

    @property
    def bounds(self) -> list[float]:
        # 'region' is your Ω ⊂ R² bounds (e.g., [xmin, xmax, ymin, ymax] or similar convention)
        region = self.problem.region
        if region is None:
            raise KeyError("Missing 'region' in problem.")
        if not isinstance(region, (list, tuple)):
            raise TypeError(f"'region' must be a list/tuple of floats, got {type(region).__name__}.")
        return [float(x) for x in region]


    @abstractmethod
    def assert_problem(self, problem: Mapping[str, Any]) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------
    @abstractmethod
    def random_individual_generator(self, size: int) -> list[Chromosome]:
        """
        Generate an initial population of valid individuals.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Genetic algorithm configuration
    # ------------------------------------------------------------------
    @abstractmethod
    def set_ga_operator_configs(self, rng: Random, parameters: GeneticAlgorithmConfigDto) -> None:
        """
        Configure problem-specific parameters of the genetic algorithm.

        This method allows the adapter to receive and interpret
        strategy-level parameters (e.g., crossover rate, mutation rate,
        distribution indices, penalty coefficients, or any other
        problem-dependent hyperparameters).

        The adapter is responsible for:
        - validating the parameters
        - mapping them to internal attributes
        - ensuring consistency with the chromosome representation
          and genetic operators
        """
        raise NotImplementedError
    
    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    @abstractmethod
    def crossover(self, parents: Sequence[Chromosome]) -> Sequence[Chromosome]:
        """
        Apply crossover operator to parent chromosomes.

        Parameters
        ----------
        parents:
            Sequence of parent chromosomes (commonly length 2).

        Returns
        -------
        list[Chromosome]
            New child chromosomes produced by the crossover.
        """
        raise NotImplementedError


    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply mutation operator to a single chromosome.

        The mutation logic is entirely problem-specific and must respect
        the underlying representation and constraints (e.g., projection
        back to a grammar language, repair heuristics, etc.).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Integration with SimLab / MongoDB / Cooja
    # ------------------------------------------------------------------
    @abstractmethod
    def encode_simulation_input(self, ind: Chromosome) -> SimulationElements:
        """
        Map an individual to the input payload for a Simulation document.

        This method translates the chromosome into a configuration that
        the master-node / Cooja simulation can understand.

        The returned dictionary is typically stored inside the Simulation
        document (e.g., under 'parameters', 'configuration', or a similar field).

        Example: For a WSN problem, this may include:
            - node positions
            - MAC/PHY parameters
            - traffic rates
            - sink mobility pattern
        """
        raise NotImplementedError(
            "encode_simulation_input must be implemented for simulation-based problems."
        )