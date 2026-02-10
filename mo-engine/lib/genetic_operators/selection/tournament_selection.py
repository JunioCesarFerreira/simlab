import random
from lib.problem.chromosomes import Chromosome

def tournament_selection(
    population: list[Chromosome], 
    individual_ranks: dict[int, int],
    rng: random.Random,
) -> Chromosome:
    i1, i2 = rng.sample(range(len(population)), 2)
    rank1: int = individual_ranks[i1]
    rank2: int = individual_ranks[i2]
    if rank1 < rank2:
        return population[i1]
    elif rank2 < rank1:
        return population[i2]
    else:
        return population[rng.choice([i1, i2])]
    
    
def compute_individual_ranks(fronts: list[list[int]]) -> dict[int, int]:
    individual_ranks: dict[int, int] = {}
    for rank, front in enumerate(fronts):
        for idx in front:
            individual_ranks[idx] = rank
    return individual_ranks