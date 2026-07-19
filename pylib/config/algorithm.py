from typing import TypedDict


class GeneticAlgorithmConfigDto(TypedDict):
    # Strategy-level (consumed by lib/strategy/*, any problem)
    population_size: int
    number_of_generations: int
    random_seed: int
    prob_cx: float
    prob_mt: float
    selection_method: str  # accepted but fixed: strategies always use tournament
    # Problem-level (consumed by the ProblemAdapter named in each comment;
    # adapters declare what they read in CONSUMED_GA_KEYS and other keys are
    # ignored with a warning at adapter build time)
    per_gene_prob: float   # P0-P4: per-gene mutation probability
    crossover_method: str  # P1 only: sbx_with_radial_translate (default) | rand_network
    eta_cx: float    # P0/P1: SBX distribution index
    eta_mt: float    # P0/P1: polynomial mutation distribution index
    pm_tau: float    # P4: route mutation prob
    sigma_tau: float  # P4: standard deviation of Gaussian tau mutation
    apply_coverage_repair: bool  # P1/P2: enable trajectory coverage repair, default True
    repair_coverage_budget: int  # P1/P2: max relay moves (P1) / candidate activations (P2) per repair


class NsgaIIIConfigDto(GeneticAlgorithmConfigDto):
    divisions: int
