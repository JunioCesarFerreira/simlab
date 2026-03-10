import random


def resolve_simulation_seeds(
    simulation_config: dict,
    rng: random.Random,
    default_count: int,
    fallback_seed: int = 123456,
) -> list[int]:
    explicit_seeds = simulation_config.get("random_seeds")
    if explicit_seeds:
        return [int(seed) for seed in explicit_seeds]

    count = int(simulation_config.get("random_seeds_count", default_count))
    if count <= 0:
        return [fallback_seed]

    return [rng.randint(1, 999999) for _ in range(count)]
