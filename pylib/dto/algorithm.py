from typing import TypedDict

class GeneticAlgorithmConfigDto(TypedDict):
    population_size: int
    number_of_generations: int
    prob_cx: float
    prob_mt: float
    selection_method: str
    crossover_method: str
    mutation_method: str
    per_gene_prob: float
    eta_cx: float
    eta_mt: float
    p_on_init: float
    min_on_init: int
    ensure_non_empty: bool
    pm_tau: float
    sigma_tau: float

class NsgaIIIConfigDto(GeneticAlgorithmConfigDto):
    divisions: int
    