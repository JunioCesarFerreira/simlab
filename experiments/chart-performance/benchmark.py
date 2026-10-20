"""Deterministic chart CPU/query-count benchmark; no live database or HTTP calls.

Run with rest-api/.venv/bin/python experiments/chart-performance/benchmark.py.
An optional --baseline-file can point at the previous experiment endpoint source.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import MagicMock

import numpy as np
from bson import ObjectId

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rest-api")]

from api.endpoints import experiment
from api.metrics_cache import MetricsCache


def measure(endpoint, dimensions, include_cumulative=None):
    rng = np.random.default_rng(20260913)
    names = [f"f{i + 1}" for i in range(dimensions)]
    gens = [{"_id": ObjectId(), "index": i} for i in range(30)]
    populations = {}
    for gen in gens:
        points = rng.random((50, dimensions))
        points /= np.linalg.norm(points, axis=1)[:, None]
        populations[gen["_id"]] = [
            {"generation_id": gen["_id"], "individual_id": f'{gen["index"]}-{i}',
             "objectives": point.tolist()} for i, point in enumerate(points)
        ]
    factory = MagicMock()
    factory.experiment_repo.get.return_value = {
        "parameters": {"objectives": [{"metric_name": n} for n in names],
                       "simulation": {"synthetic": {"enabled": True, "bench": "DTLZ2"}}},
    }
    factory.generation_repo.find_by_experiment.return_value = gens
    factory.individual_repo.find_by_generation.side_effect = lambda gid: populations[gid]
    factory.individual_repo.find_grouped_by_experiment.return_value = populations
    if hasattr(endpoint, "hv_gd_cache"):
        endpoint.hv_gd_cache = MetricsCache()
    options = {} if include_cumulative is None else {"include_cumulative": include_cumulative}
    timings = []
    for _ in range(2):
        factory.individual_repo.reset_mock()
        start = perf_counter()
        result = endpoint.get_hv_gd(str(ObjectId()), names, ["true"] * dimensions,
                                    "offspring", True, factory, **options)
        timings.append(perf_counter() - start)
    return result, {
        "first_seconds": timings[0], "repeat_seconds": timings[1],
        "individual_queries_per_request": {
            "by_generation": factory.individual_repo.find_by_generation.call_count,
            "bulk": factory.individual_repo.find_grouped_by_experiment.call_count,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, default=3, choices=[2, 3, 4, 5, 6])
    parser.add_argument("--include-cumulative", action="store_true")
    parser.add_argument("--baseline-file", type=Path)
    args = parser.parse_args()
    result, current = measure(experiment, args.dimensions, args.include_cumulative)
    report = {"dimensions": args.dimensions, "generations": 30, "points_per_generation": 50,
              "include_cumulative": args.include_cumulative, "current": current}
    if args.baseline_file:
        spec = importlib.util.spec_from_file_location("chart_baseline", args.baseline_file)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        before, report["baseline"] = measure(baseline, args.dimensions)
        names = ["hv", "gd", "igd", "igd_plus"]
        if args.include_cumulative:
            names.append("hv_cumulative")
        errors = {}
        for name in names:
            np.testing.assert_allclose(before[name], result[name], rtol=1e-12, atol=1e-12)
            errors[name] = float(np.max(np.abs(np.asarray(before[name]) - result[name])))
        report["maximum_absolute_error"] = errors
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
