"""In-process analytical evaluation shared by the strategies.

For analytical problems (P0 synthetic benchmarks) the objectives are a closed-form
function of the decision vector, so evaluating them directly — in the mo-engine
process — avoids the MongoDB Simulation document, the change-stream round-trip and
the master-node hop used for Cooja simulations. The generation then completes via
the existing "no simulations pending → mark generation done" path.
"""
from __future__ import annotations

import random
from typing import Sequence

from pylib import benchmarks


def analytical_objectives(
    adapter,
    genome,
    genome_hash: str,
    *,
    bench: str,
    noise_std: float,
    sch1_domain: tuple[float, float],
    seeds: Sequence[int],
    n_obj: int,
    objective_goals: Sequence[int],
) -> tuple[list[float], list[float]]:
    """Evaluate *genome* analytically.

    Returns ``(original, minimization)`` objective vectors:
      * ``original``     — benchmark values in the objectives' natural orientation
        (what is persisted on the Individual document);
      * ``minimization`` — the same values mapped to minimisation space via
        ``objective_goals`` (what the NSGA loop consumes).

    The benchmark is evaluated once per simulation seed and averaged (mean),
    matching the multi-seed aggregation used for simulation-based problems. The
    per-(seed, genome) RNG makes noisy evaluations reproducible.
    """
    dv = adapter.decision_vector(genome)
    seed_list = list(seeds) or [0]
    per_seed = [
        benchmarks.evaluate_noisy(
            bench, dv, n_obj, noise_std,
            random.Random(f"{seed}:{genome_hash}"),
            sch1_domain=sch1_domain,
        )
        for seed in seed_list
    ]
    original = [sum(col) / len(col) for col in zip(*per_seed)]
    minimization = [g * v for g, v in zip(objective_goals, original)]
    return original, minimization
