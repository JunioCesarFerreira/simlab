#!/usr/bin/env python3
"""Cost/quality comparison: NSGA-III vs NSGA-III adaptive simulation.

Runs both strategies over the *same* P2 scenario, the same seeds and the same
deterministic evaluator, through the real SimLab generation lifecycle (driven
in-process by the test fakes, so no MongoDB / master-node / Cooja is needed).

It reports what the thesis needs to compare the two approaches:

  cost     - number of real evaluations, avoided evaluations, reduction ratio
  quality  - hypervolume of the published front, Pareto recall and precision
             of the adaptive front against the baseline front

Usage
-----
    python experiments/adaptive-simulation/run_comparison.py
    python experiments/adaptive-simulation/run_comparison.py --generations 10 \
        --population 24 --seed 7 --json out.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "mo-engine"))
sys.path.insert(0, str(_ROOT / "mo-engine" / "tests"))

import numpy as np  # noqa: E402
from bson import ObjectId  # noqa: E402

from adaptive_fakes import (  # noqa: E402
    FakeMongo,
    TopologyEvaluator,
    harness_strategy,
    run_to_completion,
)
from lib.adaptive.dominance import dominates  # noqa: E402
from lib.strategy.nsga3 import NSGA3LoopStrategy  # noqa: E402
from lib.strategy.nsga3_adaptive import NSGA3AdaptiveSimulationStrategy  # noqa: E402

SINK = (0.0, 0.0)
OBJECTIVES = [
    {"metric_name": "latency", "goal": "min"},
    {"metric_name": "energy", "goal": "min"},
    {"metric_name": "throughput", "goal": "max"},
]

ADAPTIVE_BLOCK: dict[str, Any] = {
    "enabled": True,
    "min_training_samples": 40,
    "estimator": {"type": "weighted_knn", "k": 7, "epsilon": 1e-9},
    "confidence": {"kappa": 1.96},
    "novelty": {"descriptor_weight": 0.7, "hamming_weight": 0.3, "threshold": 0.40},
    "uncertainty_threshold": 0.25,
    "dominance_margin": 0.02,
    "audit_probability": 0.05,
    "simulation_budget": {
        "enabled": False,
        "min_per_generation": 5,
        "max_per_generation": 20,
        "promotion_reserve": 5,
    },
    "require_simulated_survivors": True,
}


# ---------------------------------------------------------------------------
# Experiment construction
# ---------------------------------------------------------------------------
def build_problem(name: str, grid: int = 7, pitch: float = 20.0) -> dict:
    half = pitch * (grid - 1) / 2.0
    return {
        "name": name,
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink": list(SINK),
        "candidates": [
            [float(x * pitch - half), float(y * pitch - half)]
            for x in range(grid) for y in range(grid)
        ],
        "mobile_nodes": [
            {
                "path_segments": [["-60 + 120*t", "40"]],
                "is_closed": False,
                "is_round_trip": True,
                "speed": 5.0,
                "time_step": 1.0,
            },
            {
                "path_segments": [["-30 + 60*t", "-45"]],
                "is_closed": False,
                "is_round_trip": True,
                "speed": 5.0,
                "time_step": 1.0,
            },
        ],
        "radius_of_reach": 30.0,
        "radius_of_inter": 60.0,
        "min_coverage_percentage": 60.0,
    }


def build_experiment(
    strategy: str,
    problem_name: str,
    population: int,
    generations: int,
    seed: int,
    adaptive: dict | None = None,
) -> dict:
    algorithm: dict[str, Any] = {
        "population_size": population,
        "number_of_generations": generations,
        "random_seed": seed,
        "prob_cx": 0.9,
        "prob_mt": 0.3,
        "divisions": 6,
        "per_gene_prob": 0.08,
    }
    if adaptive is not None:
        algorithm["adaptive_evaluation"] = copy.deepcopy(adaptive)
    return {
        "_id": ObjectId(),
        "name": f"adaptive-comparison-{strategy}",
        "parameters": {
            "strategy": strategy,
            "algorithm": algorithm,
            "simulation": {
                "duration": 180,
                "random_seeds": [11],
                "synthetic": {"enabled": True},
            },
            "problem": build_problem(problem_name),
            "objectives": OBJECTIVES,
        },
        "source_repository_options": {},
        "data_conversion_config": {},
    }


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def front_to_minimization(front: Sequence[dict]) -> np.ndarray:
    """Objective matrix of a published front, in minimization space."""
    signs = [1 if o["goal"] == "min" else -1 for o in OBJECTIVES]
    keys = [o["metric_name"] for o in OBJECTIVES]
    if not front:
        return np.zeros((0, len(keys)))
    return np.asarray(
        [[item["objectives"][k] * s for k, s in zip(keys, signs)] for item in front],
        dtype=float,
    )


def reference_point(points: np.ndarray, margin: float = 0.1) -> np.ndarray:
    """Nadir-based reference box corner, valid for negative objectives too.

    Minimization space negates every maximised objective, so the values can be
    negative: scaling the maximum by a factor would move the reference the
    wrong way. Offsetting by a fraction of the observed *range* is sign-safe.
    """
    high = points.max(axis=0)
    spread = high - points.min(axis=0)
    spread[spread <= 0.0] = 1.0
    return high + margin * spread


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    """Monte-Carlo hypervolume of a minimization front (reference = nadir).

    Exact HV is exponential in the objective count; a fixed-seed Monte-Carlo
    estimate is enough to compare two fronts on the same reference box and
    keeps this script dependency-free.
    """
    if points.size == 0:
        return 0.0
    ideal = points.min(axis=0)
    box = reference - ideal
    if np.any(box <= 0):
        return 0.0
    rng = np.random.default_rng(12345)
    samples = ideal + rng.random((200_000, points.shape[1])) * box
    dominated = np.zeros(samples.shape[0], dtype=bool)
    for point in points:
        dominated |= np.all(point <= samples, axis=1)
    return float(dominated.mean() * np.prod(box))


def pareto_recall_precision(
    candidate: np.ndarray, reference: np.ndarray, tolerance: float = 1e-9
) -> tuple[float, float]:
    """How much of the reference front the candidate matches, and vice-versa.

    * recall    - reference points not dominated by the candidate front are
                  "missed"; recall is the fraction that *is* attained.
    * precision - fraction of candidate points that are not dominated by the
                  reference front.
    """
    if reference.size == 0 or candidate.size == 0:
        return 0.0, 0.0
    attained = sum(
        1 for r in reference
        if any(np.all(c <= r + tolerance) for c in candidate)
    )
    kept = sum(
        1 for c in candidate
        if not any(dominates(r, c) for r in reference)
    )
    return attained / len(reference), kept / len(candidate)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(cls, experiment: dict) -> dict[str, Any]:
    mongo = FakeMongo()
    evaluator = TopologyEvaluator(SINK)
    strategy = harness_strategy(cls, experiment, mongo)

    started = time.perf_counter()
    run_to_completion(strategy, mongo, evaluator)
    wall_clock = time.perf_counter() - started

    document = mongo.experiment_repo.documents[str(experiment["_id"])]
    result: dict[str, Any] = {
        "strategy": experiment["parameters"]["strategy"],
        "problem": experiment["parameters"]["problem"]["name"],
        "status": document.get("status"),
        "wall_clock_seconds": wall_clock,
        "evaluations": evaluator.calls,
        "simulation_documents": len(mongo.simulation_repo.documents),
        "generations": len(mongo.generation_repo.find_by_experiment(strategy._exp_id)),
        "pareto_front": document.get("pareto_front") or [],
    }
    if isinstance(strategy, NSGA3AdaptiveSimulationStrategy):
        result["adaptive"] = strategy.adaptive_metrics.experiment_summary()
    return result


def evaluate_quality(
    baseline_front: np.ndarray, adaptive_front: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    recall, precision = pareto_recall_precision(adaptive_front, baseline_front)
    baseline_hv = hypervolume(baseline_front, reference)
    adaptive_hv = hypervolume(adaptive_front, reference)
    return {
        "baseline_front_size": int(baseline_front.shape[0]),
        "adaptive_front_size": int(adaptive_front.shape[0]),
        "baseline_hypervolume": baseline_hv,
        "adaptive_hypervolume": adaptive_hv,
        "hypervolume_ratio": adaptive_hv / baseline_hv if baseline_hv else 0.0,
        "pareto_recall": recall,
        "pareto_precision": precision,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kappa", type=float, nargs="+", default=[1.96, 1.0, 0.5, 0.25],
        help="confidence widths to sweep; kappa scales the optimistic bound "
             "L(x) = f(x) - kappa*sigma(x), so smaller means more aggressive skipping",
    )
    parser.add_argument("--json", type=str, default=None, help="write the raw report here")
    args = parser.parse_args()

    # Two reference arms. The classic one is the historical baseline; the
    # topology-aware one isolates the *strategy* effect, since it shares the
    # structural repair operator with the adaptive runs.
    classic = run(
        NSGA3LoopStrategy,
        build_experiment("nsga3", "problem2", args.population, args.generations, args.seed),
    )
    baseline = run(
        NSGA3LoopStrategy,
        build_experiment(
            "nsga3", "problem2_topology_aware", args.population, args.generations, args.seed
        ),
    )
    baseline_front = front_to_minimization(baseline["pareto_front"])

    runs: list[dict[str, Any]] = []
    for kappa in args.kappa:
        block = copy.deepcopy(ADAPTIVE_BLOCK)
        block["confidence"] = {"kappa": kappa}
        result = run(
            NSGA3AdaptiveSimulationStrategy,
            build_experiment(
                "nsga3_adaptive_simulation", "problem2_topology_aware",
                args.population, args.generations, args.seed, block,
            ),
        )
        result["kappa"] = kappa
        runs.append(result)

    fronts = [baseline_front] + [front_to_minimization(r["pareto_front"]) for r in runs]
    stacked = [f for f in fronts if f.size]
    reference = reference_point(np.vstack(stacked)) if stacked else np.zeros(len(OBJECTIVES))
    for result, front in zip(runs, fronts[1:]):
        result["quality"] = evaluate_quality(baseline_front, front, reference)

    print()
    print("=" * 78)
    print(" NSGA-III baseline  vs  NSGA-III adaptive simulation")
    print("=" * 78)
    print(f" population={args.population}  generations={args.generations}  seed={args.seed}")
    print(f" reference arms: nsga3+problem2 = {classic['evaluations']} evaluations, "
          f"nsga3+problem2_topology_aware = {baseline['evaluations']} evaluations")
    print("-" * 78)
    header = (
        f"{'kappa':>6} {'sims':>6} {'reuse':>6} {'estim':>6} {'promo':>6} {'audit':>6} "
        f"{'reduct':>7} {'HVrat':>7} {'recall':>7} {'prec':>6} {'MAE':>7} {'fskip':>6}"
    )
    print(header)
    print("-" * 78)
    for result in runs:
        summary = result.get("adaptive", {})
        quality = result["quality"]
        false_skip = summary.get("false_skip_rate")
        mae = summary.get("prediction_mae")
        print(
            f"{result['kappa']:>6.2f} {result['evaluations']:>6} "
            f"{summary.get('exact_cache_hits', 0):>6} {summary.get('estimated_only', 0):>6} "
            f"{summary.get('promotions', 0):>6} {summary.get('audit_simulations', 0):>6} "
            f"{summary.get('simulation_reduction_ratio', 0.0):>7.3f} "
            f"{quality['hypervolume_ratio']:>7.3f} {quality['pareto_recall']:>7.3f} "
            f"{quality['pareto_precision']:>6.3f} "
            f"{('%7.3f' % mae) if mae is not None else '      -'} "
            f"{('%6.3f' % false_skip) if false_skip is not None else '     -'}"
        )
    print("-" * 78)
    print(" sims   = real evaluations (Simulation documents actually executed)")
    print(" reduct = 1 - N_adaptive / N_baseline, over the new individuals of the run")
    print(" HVrat  = hypervolume(adaptive front) / hypervolume(baseline front)")
    print(" fskip  = fraction of audited skips whose real objectives were non-dominated")
    print("=" * 78)

    if args.json:
        payload = {"classic": classic, "baseline": baseline, "adaptive_runs": runs}
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"raw report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
