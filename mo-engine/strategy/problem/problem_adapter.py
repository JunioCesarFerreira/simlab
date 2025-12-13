from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Type aliases for clarity
Chromosome = Any  # Can be np.ndarray, tuple, dict, custom class, etc.
Objectives = list[float]
ConstraintViolation = float | list[float] | None
Meta = dict[str, Any]


@dataclass
class Individual:
    """
    Generic representation of a candidate solution.

    The NSGA3 core and loop strategy only need:
    - a chromosome (opaque representation)
    - objective values
    - constraint violation (if any)
    - optional metadata (e.g., Simulation/Generation ids in MongoDB)
    """
    chromosome: Chromosome
    objectives: Objectives | None = None
    constraint_violation: ConstraintViolation = None
    meta: Meta = field(default_factory=dict)


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

    def __init__(self, experiment: Mapping[str, Any]) -> None:
        """
        Parameters
        ----------
        experiment:
            Full experiment document (or DTO) as loaded from MongoDB.
            It should include:
            - problem_type / problem_parameters
            - strategy / nsga3_parameters
            - any additional fields needed by the adapter
        """
        self.experiment = experiment

    # ------------------------------------------------------------------
    # Structural information about the problem
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def n_objectives(self) -> int:
        """Number of objective functions."""
        raise NotImplementedError

    @property
    def n_constraints(self) -> int:
        """
        Number of constraints.

        Default is 0. Override if the problem has explicit constraints
        and you want to track constraint violations.
        """
        return 0

    def bounds(
        self,
    ) -> tuple[Sequence[float], Sequence[float]] | None:
        """
        Optional: lower and upper bounds for real/integer vector problems.

        Returns
        -------
        (lower_bounds, upper_bounds) or None
            If the problem has a box-constrained representation, return
            the bounds here. Otherwise return None.
        """
        return None

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------
    @abstractmethod
    def sample_initial_population(self, size: int) -> list[Individual]:
        """
        Generate an initial population of valid individuals.

        This method is responsible for sampling admissible chromosomes
        (e.g., random vectors, grammar-based paths, mixed encodings)
        and wrapping them into Individual instances.

        The objectives and constraint_violation are usually left as None
        and will be filled later by evaluation/simulation.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    @abstractmethod
    def crossover(self, parents: Sequence[Chromosome]) -> list[Chromosome]:
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

    def crossover_probability(self) -> float:
        """
        Probability of applying crossover.

        By default this reads from nsga3_parameters.pc if available,
        otherwise falls back to 0.9.
        """
        return self.experiment.get("nsga3_parameters", {}).get("pc", 0.9)

    def mutation_probability(self) -> float:
        """
        Probability of applying mutation.

        By default this reads from nsga3_parameters.pm if available,
        otherwise falls back to 0.1.
        """
        return self.experiment.get("nsga3_parameters", {}).get("pm", 0.1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_sync(
        self,
        chromosome: Chromosome,
    ) -> tuple[Objectives, ConstraintViolation]:
        """
        Purely analytical evaluation (no external simulation).

        This is intended for synthetic problems such as DTLZ, ZDT, SCH,
        etc., where the objective functions can be computed directly
        from the chromosome.

        For problems that depend on external simulations (e.g., Cooja),
        you usually do not implement this and rely instead on:
        - encode_simulation_input
        - decode_simulation_output

        Raises
        ------
        NotImplementedError
            If the problem must be evaluated via simulation.
        """
        raise NotImplementedError(
            "evaluate_sync is not implemented for this problem; "
            "use encode_simulation_input/decode_simulation_output instead."
        )

    # ------------------------------------------------------------------
    # Integration with SimLab / MongoDB / Cooja
    # ------------------------------------------------------------------
    def encode_simulation_input(self, ind: Individual) -> dict[str, Any]:
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

    def decode_simulation_output(
        self,
        sim_doc: Mapping[str, Any],
        ind: Individual,
    ) -> None:
        """
        Update the given Individual with objectives and constraint violation
        based on the simulation result document.

        Parameters
        ----------
        sim_doc:
            Simulation document coming from MongoDB after the master-node
            finishes the Cooja execution. Expected to contain metrics such as
            latency, energy, packet loss, throughput, etc.
        ind:
            Individual to be updated in-place. This method should set:
            - ind.objectives
            - ind.constraint_violation (if applicable)
        """
        raise NotImplementedError(
            "decode_simulation_output must be implemented for simulation-based problems."
        )

    # ------------------------------------------------------------------
    # Problem-specific stopping criterion (optional)
    # ------------------------------------------------------------------
    def should_stop(
        self,
        generation_index: int,
        population: Sequence[Individual],
    ) -> bool:
        """
        Optional problem-specific stopping criterion.

        The NSGA3LoopStrategy can combine this with global stopping rules
        (maximum generations, simulation budget, time budget, etc.).

        Default implementation uses nsga3_parameters.max_generations if
        present in the experiment document.
        """
        max_gen = (
            self.experiment
            .get("nsga3_parameters", {})
            .get("max_generations", 50)
        )
        return generation_index >= max_gen
