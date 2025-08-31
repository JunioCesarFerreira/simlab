from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar, Generic, Optional
import random
import math

T = TypeVar("T")  # Individual type (e.g., list[float], custom dataclass, etc.)

# -------- GA class --------

SelectionFn = Callable[[Sequence[T], Sequence[float], int, random.Random], list[int]]
CrossoverFn = Callable[[T, T, random.Random], tuple[T, T]]
MutationFn  = Callable[[T, random.Random], T]


@dataclass
class GAConfig:
    """Config knobs for the GA step."""
    n_offspring: int
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    maximize: bool = True      # if False, scores are negated before selection
    seed: Optional[int] = None


class GeneticAlgorithm(Generic[T]):
    """
    Pluggable GA building block.

    This class does not run the generational loop. Call `step(pop, fitness)`
    from your external loop to produce `n_offspring` children using the
    configured selection, crossover and mutation operators.

    - Selection sees a 1-D score where higher is better (use `maximize=False`
      to negate your fitness if you are minimizing).
    - Crossover is applied with probability `crossover_prob`.
    - Mutation is applied independently to each child with probability `mutation_prob`.
    """

    def __init__(
        self,
        config: GAConfig,
        selection: SelectionFn[T],
        crossover: CrossoverFn[T],
        mutation: MutationFn[T],
    ) -> None:
        self.cfg = config
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.rng = random.Random(config.seed)

    def step(self, population: Sequence[T], fitness: Sequence[float]) -> list[T]:
        """Produce `n_offspring` children from `population` using `fitness`."""
        if len(population) == 0:
            return []
        if len(population) != len(fitness):
            raise ValueError("population and fitness must have the same length.")

        # Prepare selection scores (higher is better)
        scores = list(fitness) if self.cfg.maximize else [-f for f in fitness]

        # Number of parents needed (2 per mating)
        n_matings = math.ceil(self.cfg.n_offspring / 2)
        parent_indices = self.selection(population, scores, 2 * n_matings, self.rng)

        children: list[T] = []
        it = iter(parent_indices)
        for _ in range(n_matings):
            i = next(it)
            j = next(it)
            p1, p2 = population[i], population[j]

            # Crossover (or cloning)
            if self.rng.random() < self.cfg.crossover_prob:
                c1, c2 = self.crossover(p1, p2, self.rng)
            else:
                c1, c2 = p1, p2

            # Mutation
            if self.rng.random() < self.cfg.mutation_prob:
                c1 = self.mutation(c1, self.rng)
            if self.rng.random() < self.cfg.mutation_prob:
                c2 = self.mutation(c2, self.rng)

            children.append(c1)
            if len(children) < self.cfg.n_offspring:
                children.append(c2)

        return children
