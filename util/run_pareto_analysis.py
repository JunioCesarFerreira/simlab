#!/usr/bin/env python3
"""
Run Pareto analysis for all completed experiments stored in MongoDB.

For each experiment with status "Done", invokes plot_pareto_results.py
and uploads the generated plots back to the experiment via the REST API.

Usage:
    python run_pareto_analysis.py [options]

Examples:
    python run_pareto_analysis.py
    python run_pareto_analysis.py --api-base http://server:8000/api/v1
    MONGO_URI=mongodb://remote:27017/ python run_pareto_analysis.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pymongo import MongoClient


PARETO_SCRIPT = Path(__file__).resolve().parent.parent / "pareto-analysis" / "plot_pareto_results.py"


def fetch_completed_experiments(uri: str, db_name: str) -> list[dict]:
    client = MongoClient(uri)
    try:
        db = client[db_name]
        return list(db["experiments"].find(
            {"status": "Done"},
            {"_id": 1, "name": 1, "parameters.objectives": 1}
        ))
    finally:
        client.close()


def run_analysis(
    exp_id: str,
    objectives: list[dict],
    api_base: str,
    api_key: str,
) -> str:
    """
    Invoke plot_pareto_results.py for one experiment.
    Returns "ok", "skip" or "error".
    """
    if len(objectives) != 3:
        print(f"  [SKIP] {len(objectives)} objective(s) found — script requires exactly 3")
        return "skip"

    obj_names = [o["metric_name"] for o in objectives]
    minimize   = ["True" if o["goal"] == "min" else "False" for o in objectives]

    cmd = [
        sys.executable, str(PARETO_SCRIPT),
        "--expid",      exp_id,
        "--api-base",   api_base,
        "--api-key",    api_key,
        "--objectives", *obj_names,
        "--minimize",   *minimize,
    ]

    result = subprocess.run(cmd, cwd=PARETO_SCRIPT.parent)

    if result.returncode != 0:
        print(f"  [ERROR] Process exited with code {result.returncode}")
        return "error"

    return "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Run Pareto analysis for all completed (Done) experiments"
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0"),
        help="MongoDB connection URI  [env: MONGO_URI]",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("DB_NAME", "simlab"),
        help="MongoDB database name  [env: DB_NAME]",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("SIMLAB_API_BASE", "http://localhost:8000/api/v1"),
        help="SimLab REST API base URL  [env: SIMLAB_API_BASE]",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SIMLAB_API_KEY", "api-password"),
        help="SimLab API key  [env: SIMLAB_API_KEY]",
    )
    args = parser.parse_args()

    print(f"Connecting to {args.uri} / {args.db} ...")
    experiments = fetch_completed_experiments(args.uri, args.db)

    if not experiments:
        print("No completed experiments found.")
        return

    total = len(experiments)
    print(f"Found {total} completed experiment(s).\n")

    counts = {"ok": 0, "skip": 0, "error": 0}

    for i, exp in enumerate(experiments, 1):
        exp_id = str(exp["_id"])
        name   = exp.get("name", "(unnamed)")
        objs   = exp.get("parameters", {}).get("objectives", [])

        print(f"[{i}/{total}] {name}  ({exp_id})")

        outcome = run_analysis(exp_id, objs, args.api_base, args.api_key)
        counts[outcome] += 1

    print(
        f"\nFinished — "
        f"{counts['ok']} OK · "
        f"{counts['error']} error(s) · "
        f"{counts['skip']} skipped"
    )


if __name__ == "__main__":
    main()
