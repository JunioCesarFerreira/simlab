"""Exercise the actual HV/GD endpoint with an in-memory repository double.

Run with rest-api/.venv/bin/python. No HTTP server or MongoDB is contacted.
"""
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rest-api")]

from bson import ObjectId
from api.endpoints.experiment import get_hv_gd


def main():
    exp_id, gen_id = ObjectId(), ObjectId()
    factory = MagicMock()
    factory.generation_repo.find_by_experiment.return_value = [{"_id": gen_id, "index": 0}]
    factory.individual_repo.find_by_generation.return_value = [{"objectives": [1., 9.]}]
    factory.experiment_repo.get.return_value = {
        "parameters": {"objectives": [{"metric_name": f"f{i}", "goal": "min"} for i in (1, 2)],
                       "simulation": {}},
        "pareto_front": [{"objectives": {"f1": 1., "f2": 9.}}],
    }
    def call(names=("f1", "f2")):
        return get_hv_gd(str(exp_id), list(names), ["true", "true"], True, factory)
    normal = call()
    reordered = call(("f2", "f1"))
    result = {"gd_original_order": normal["gd"], "gd_reordered_names": reordered["gd"]}
    factory.individual_repo.find_by_generation.return_value = [{"objectives": [1e9, 1e9]}]
    try:
        result["no_feasible_individuals"] = call()
    except ValueError as exc:
        result["no_feasible_individuals"] = f"{type(exc).__name__}: {exc}"
    doc = factory.experiment_repo.get.return_value
    doc["parameters"]["simulation"] = {"synthetic": {"enabled": True, "bench": "ZDT1"}}
    doc["pareto_front"] = None
    factory.individual_repo.find_by_generation.return_value = [{"objectives": [0.5, 0.5]}]
    result["synthetic_without_final_front"] = call()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
