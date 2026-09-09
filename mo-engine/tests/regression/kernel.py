"""Synchronous NSGA kernel harness for the metrics-regression baseline.

Phase 0 of ``docs/markdown/NSGA_METRICS_FIX_PLAN.md``. It drives the *real*
SimLab mating and environmental-selection methods over an analytical benchmark
with no MongoDB, no change streams and no simulation workers, so a run is a
pure function of ``(config, seed)``.

Three sets are recorded per generation, because the audit
(``experiments/nsga-metrics-audit``) showed the platform currently plots one of
them while the notebooks plot another:

  * ``offspring``  — ND(Q_t), the newly generated children. This is what the
    ``/hv-gd`` endpoint measures today.
  * ``survivors``  — ND(P_t), the population kept by environmental selection.
    This is what the NSGA-Studies notebooks measure.
  * ``archive``    — ND of every individual evaluated so far.

Deliberately different from production in one respect: the kernel applies
environmental selection to the LAST offspring batch as well. Production skips
it (finding 8), which Phase 1.2 will correct; the kernel already encodes the
intended behaviour so the baseline does not have to be re-frozen for it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import logging

import moocore
import numpy as np

from pylib import benchmarks, moo_metrics

from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy

# The frozen seed set of the audit. Kept as a tuple so a baseline can never be
# silently regenerated over a different sample.
SEEDS: tuple[int, ...] = (1, 2, 3, 5, 7)

_STRATEGIES = {"nsga2": NSGA2LoopStrategy, "nsga3": NSGA3LoopStrategy}


@dataclass(frozen=True)
class KernelConfig:
    """One point of the baseline grid.

    ``prob_mt``/``per_gene_prob`` default to the values the GUI actually sends
    (finding 4), not to the textbook ones: the baseline must capture the
    platform as it is today.
    """

    algorithm: str          # "nsga2" | "nsga3"
    bench: str              # "DTLZ2" | "ZDT1" | "SCH1"
    m: int
    n: int
    pop_size: int = 50
    generations: int = 20
    divisions: int = 10
    prob_cx: float = 0.9
    prob_mt: float = 0.1
    per_gene_prob: float = 0.05
    eta_cx: float = 20.0
    eta_mt: float = 20.0

    @property
    def label(self) -> str:
        return f"{self.algorithm}-{self.bench.lower()}-m{self.m}-n{self.n}"


# The grid frozen for the pre-correction baseline. DTLZ2 M=3/n=10 mirrors the
# GUI's synthetic draft default; ZDT1 and SCH1 cover the two-objective path,
# SCH1 with the single decision variable used by the notebooks.
BASELINE_CONFIGS: tuple[KernelConfig, ...] = (
    KernelConfig(algorithm="nsga2", bench="DTLZ2", m=3, n=10),
    KernelConfig(algorithm="nsga3", bench="DTLZ2", m=3, n=10),
    KernelConfig(algorithm="nsga2", bench="ZDT1", m=2, n=10),
    KernelConfig(algorithm="nsga3", bench="ZDT1", m=2, n=10),
    KernelConfig(algorithm="nsga2", bench="SCH1", m=2, n=1),
    KernelConfig(algorithm="nsga3", bench="SCH1", m=2, n=1),
)


def build_strategy(config: KernelConfig, seed: int):
    """Instantiate a real strategy with no Mongo handle.

    Only the in-process pieces are exercised (problem adapter, GA operators,
    environmental selection), none of which touch ``self.mongo``.
    """
    cls = _STRATEGIES[config.algorithm]
    return cls(
        {
            "parameters": {
                "problem": {"name": "problem0", "n": config.n},
                "simulation": {
                    "synthetic": {"enabled": True, "bench": config.bench, "noise_std": 0.0},
                    "random_seeds": [42],
                },
                "algorithm": {
                    "population_size": config.pop_size,
                    "number_of_generations": config.generations,
                    "random_seed": seed,
                    "prob_cx": config.prob_cx,
                    "prob_mt": config.prob_mt,
                    "per_gene_prob": config.per_gene_prob,
                    "eta_cx": config.eta_cx,
                    "eta_mt": config.eta_mt,
                    "divisions": config.divisions,
                },
                "objectives": [
                    {"metric_name": f"f{i + 1}", "goal": "min"} for i in range(config.m)
                ],
            }
        },
        None,
    )


def nondominated(points: np.ndarray) -> np.ndarray:
    """Deduplicated non-dominated subset, in minimisation space."""
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return np.asarray(moocore.filter_dominated(np.unique(arr, axis=0)), dtype=float)


def indicators(
    points: np.ndarray,
    reference_front: np.ndarray,
    hv_ref: list[float],
    *,
    bench: str,
    gd_scale: float,
) -> dict:
    """HV / GD / IGD / IGD+ of a point set, exactly as ``/hv-gd`` computes them.

    HV counts only points strictly dominating the reference point; GD is the
    EXACT distance to the analytical front; IGD and IGD+ average over the
    sampled reference and are normalised by the benchmark's theoretical
    ideal-nadir range. All three match
    ``rest-api/api/endpoints/experiment.py``, so a baseline number can be
    compared against a real experiment's series.

    Phase 0 recorded a separate ``radial_error`` here as a reference-free
    convergence signal. Phase 4 promoted exactly that quantity to GD itself, so
    the extra column would now be a duplicate and is gone.
    """
    front = nondominated(points)
    hv_ref_arr = np.asarray(hv_ref, dtype=float)
    dominating = front[np.all(front < hv_ref_arr, axis=1)] if len(front) else front
    return {
        "hv": float(moocore.hypervolume(dominating, ref=hv_ref)) if len(dominating) else 0.0,
        "gd": moo_metrics.gd_analytical(
            benchmarks.front_distance(bench, front, front.shape[1]), scale=gd_scale
        ) if len(front) else None,
        "igd": moo_metrics.igd(front, reference_front, bounds=_bounds(bench, len(hv_ref))),
        "igd_plus": moo_metrics.igd_plus(front, reference_front, bounds=_bounds(bench, len(hv_ref))),
        "front_size": int(len(front)),
    }


def _bounds(bench: str, m: int) -> "tuple[np.ndarray, np.ndarray]":
    return (
        np.asarray(benchmarks.ideal(bench, m), dtype=float),
        np.asarray(benchmarks.nadir(bench, m), dtype=float),
    )


def run_kernel(config: KernelConfig, seed: int) -> list[dict]:
    """Run one (config, seed) and return the per-generation record list."""
    strategy = build_strategy(config, seed)
    reference_front = benchmarks.true_front(config.bench, config.m)
    hv_ref = [v * 1.1 for v in benchmarks.nadir(config.bench, config.m)]
    scale = moo_metrics.analytical_scale(*_bounds(config.bench, config.m))
    assert np.allclose(scale, scale[0]), "the analytical GD needs an isotropic range"
    measure = {"bench": config.bench, "gd_scale": float(scale[0])}

    def evaluate(population) -> np.ndarray:
        for chromosome in population:
            strategy._map_genome_objectives[chromosome] = benchmarks.evaluate(
                config.bench, chromosome.x, config.m
            )
        return np.array(
            [strategy._map_genome_objectives[c] for c in population], dtype=float
        )

    strategy._parents = strategy._problem_adapter.random_individual_generator(
        config.pop_size
    )
    objectives = evaluate(strategy._parents)
    archive = nondominated(objectives)

    initial = indicators(objectives, reference_front, hv_ref, **measure)
    history = [
        {
            "generation": 0,
            # Generation 0 has no offspring yet: the initial population is both
            # the newly evaluated set and the surviving one.
            "offspring": initial,
            "survivors": initial,
            "archive": indicators(archive, reference_front, hv_ref, **measure),
        }
    ]

    for generation in range(1, config.generations + 1):
        children = strategy._run_genetic_algorithm()
        child_objectives = evaluate(children)

        union = strategy._parents + children
        union_objectives = [
            strategy._map_genome_objectives[c] for c in union
        ]
        strategy._parents = strategy._select_next_parents(union, union_objectives)
        survivor_objectives = np.array(
            [strategy._map_genome_objectives[c] for c in strategy._parents], dtype=float
        )
        archive = nondominated(np.vstack([archive, child_objectives]))

        history.append(
            {
                "generation": generation,
                "offspring": indicators(child_objectives, reference_front, hv_ref, **measure),
                "survivors": indicators(survivor_objectives, reference_front, hv_ref, **measure),
                "archive": indicators(archive, reference_front, hv_ref, **measure),
            }
        )
    return history


def run_grid(
    configs: "tuple[KernelConfig, ...]" = BASELINE_CONFIGS,
    seeds: "tuple[int, ...]" = SEEDS,
    *,
    progress=None,
) -> list[dict]:
    """Run every (config, seed) pair and return the flat run list."""
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        runs = []
        for config in configs:
            for seed in seeds:
                runs.append(
                    {
                        "config": asdict(config),
                        "label": config.label,
                        "seed": seed,
                        "history": run_kernel(config, seed),
                    }
                )
            if progress is not None:
                progress(config.label)
        return runs
    finally:
        logging.disable(previous_level)
