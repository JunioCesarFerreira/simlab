from typing import TypedDict

class GeneticAlgorithmConfigDto(TypedDict):
    population_size: int
    number_of_generations: int
    random_seed: int
    prob_cx: float
    prob_mt: float
    per_gene_prob: float
    selection_method: str
    crossover_method: str
    mutation_method: str
    # Specific attributes
    eta_cx: float # sbx
    eta_mt: float # polynomial
    pm_tau: float # route mutation prob P4
    sigma_tau: float # standard deviation of Gaussian distribution tau mutation P4

class NsgaIIIConfigDto(GeneticAlgorithmConfigDto):
    divisions: int
    