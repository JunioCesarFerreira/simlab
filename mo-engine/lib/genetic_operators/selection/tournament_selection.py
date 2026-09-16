import random

from lib.nsga.crowding_distance import crowding_distance
from lib.problem.chromosomes import Chromosome


def tournament_selection(
    population: list[Chromosome],
    individual_ranks: dict[int, int],
    rng: random.Random,
    crowding: dict[int, float] | None = None,
) -> Chromosome:
    """Binary tournament between two individuals drawn without replacement.

    Lower rank wins. On a tie, *crowding* — the NSGA-II crowded-comparison
    operator — breaks it in favour of the larger distance, i.e. the individual
    in the emptier region of its front. Without it the tie is a coin flip, which
    drops the only pressure NSGA-II applies towards spread during mating: rank
    alone cannot distinguish between members of the same front, and every
    individual of the first front is tied with every other.

    *crowding* is optional because it is not universally the right secondary
    criterion. NSGA-III deliberately leaves it out — diversity there is enforced
    by reference-point niching during environmental selection, and Deb & Jain
    mate by random selection rather than by a crowded tournament.
    """
    i1, i2 = rng.sample(range(len(population)), 2)
    rank1: int = individual_ranks[i1]
    rank2: int = individual_ranks[i2]
    if rank1 < rank2:
        return population[i1]
    if rank2 < rank1:
        return population[i2]

    if crowding is not None:
        d1 = crowding.get(i1, 0.0)
        d2 = crowding.get(i2, 0.0)
        if d1 > d2:
            return population[i1]
        if d2 > d1:
            return population[i2]

    return population[rng.choice([i1, i2])]


def compute_individual_ranks(fronts: list[list[int]]) -> dict[int, int]:
    individual_ranks: dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            individual_ranks[idx] = rank
    return individual_ranks


def compute_crowding_distances(
    fronts: list[list[int]],
    objectives: list[list[float]],
) -> dict[int, float]:
    """Crowding distance of every individual, keyed by its index in the population.

    Computed per front, as NSGA-II defines it: the measure only compares
    individuals of equal rank, which is exactly when the tournament needs it.
    """
    distances: dict[int, float] = {}
    for front in fronts:
        for idx, dist in zip(front, crowding_distance(front, objectives)):
            distances[idx] = dist
    return distances
