#!/usr/bin/env python3
"""Repair-based vs representation-based feasibility on P2.

Runs the *same* strategy (NSGA-III) over the same scenario and seed with the
three P2 encodings, so the only variable is how feasibility is obtained:

| problem | feasibility comes from |
|---|---|
| ``problem2``                | global BFS repair after every operator |
| ``problem2_topology_aware`` | structure-driven repair after every operator |
| ``problem2_tree_encoded``   | nothing — the operators are closed over feasible trees |

Reports, per encoding: real evaluations, engine wall-clock, how often the
operators had to fall back, front size and hypervolume, and — the invariant
that matters — whether any persisted chromosome was ever disconnected.

Usage
-----
    python experiments/adaptive-simulation/run_encoding_comparison.py
    python experiments/adaptive-simulation/run_encoding_comparison.py \\
        --population 24 --generations 10 --seed 7 --repeats 3
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "mo-engine"))
sys.path.insert(0, str(_ROOT / "mo-engine" / "tests"))

import numpy as np  # noqa: E402

from adaptive_fakes import FakeMongo, TopologyEvaluator, harness_strategy, run_to_completion  # noqa: E402
from lib.problem.topology import build_sink_rooted_tree  # noqa: E402
from lib.strategy.nsga3 import NSGA3LoopStrategy  # noqa: E402
from run_comparison import (  # noqa: E402
    OBJECTIVES,
    SINK,
    build_experiment,
    front_to_minimization,
    hypervolume,
    reference_point,
)

ENCODINGS = ["problem2", "problem2_topology_aware", "problem2_tree_encoded"]


def run_once(problem_name: str, population: int, generations: int, seed: int) -> dict[str, Any]:
    experiment = build_experiment("nsga3", problem_name, population, generations, seed)
    mongo = FakeMongo()
    evaluator = TopologyEvaluator(SINK)
    strategy = harness_strategy(NSGA3LoopStrategy, experiment, mongo)

    started = time.perf_counter()
    run_to_completion(strategy, mongo, evaluator)
    wall_clock = time.perf_counter() - started

    scenario = getattr(strategy._problem_adapter, "scenario", None)
    disconnected = 0
    individuals = list(mongo.individual_repo.documents.values())
    if scenario is not None:
        for document in individuals:
            tree = build_sink_rooted_tree(scenario, document["chromosome"]["mask"])
            if tree.detached_nodes():
                disconnected += 1

    relays = [sum(d["chromosome"]["mask"]) for d in individuals] or [0]
    document = mongo.experiment_repo.documents[str(experiment["_id"])]
    return {
        "problem": problem_name,
        "evaluations": evaluator.calls,
        "individuals": len(individuals),
        "wall_clock_seconds": wall_clock,
        "disconnected": disconnected,
        "mean_relays": statistics.fmean(relays),
        "pareto_front": document.get("pareto_front") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="seeds to average over")
    args = parser.parse_args()

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in ENCODINGS}
    for offset in range(args.repeats):
        for name in ENCODINGS:
            results[name].append(
                run_once(name, args.population, args.generations, args.seed + offset)
            )

    fronts = {
        name: [front_to_minimization(run["pareto_front"]) for run in runs]
        for name, runs in results.items()
    }
    stacked = [f for runs in fronts.values() for f in runs if f.size]
    reference = reference_point(np.vstack(stacked)) if stacked else np.zeros(len(OBJECTIVES))

    print()
    print("=" * 92)
    print(" Feasibility by repair vs feasibility by representation  (strategy: nsga3)")
    print("=" * 92)
    print(f" population={args.population}  generations={args.generations}  "
          f"seed={args.seed}  repeats={args.repeats}")
    print("-" * 92)
    print(f"{'problem':28} {'evals':>7} {'indiv':>7} {'relays':>8} "
          f"{'front':>6} {'HV':>12} {'wall(s)':>9} {'disconn.':>9}")
    print("-" * 92)
    for name in ENCODINGS:
        runs = results[name]
        hv = statistics.fmean(hypervolume(f, reference) for f in fronts[name])
        print(
            f"{name:28} "
            f"{statistics.fmean(r['evaluations'] for r in runs):>7.1f} "
            f"{statistics.fmean(r['individuals'] for r in runs):>7.1f} "
            f"{statistics.fmean(r['mean_relays'] for r in runs):>8.1f} "
            f"{statistics.fmean(len(r['pareto_front']) for r in runs):>6.1f} "
            f"{hv:>12.1f} "
            f"{statistics.fmean(r['wall_clock_seconds'] for r in runs):>9.2f} "
            f"{sum(r['disconnected'] for r in runs):>9}"
        )
    print("-" * 92)
    print(" evals    = real evaluations (a lower count here means more exact-cache reuse,")
    print("            i.e. a more local search, not a cheaper one per individual)")
    print(" disconn. = persisted chromosomes whose mask is NOT connected to the sink;")
    print("            must be 0 for every encoding - the P2 feasibility invariant")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
