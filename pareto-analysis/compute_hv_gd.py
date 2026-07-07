"""
Compute Hypervolume (HV) and Generational Distance (GD) per generation.
Outputs a single JSON object to stdout.
"""
import json
import os
import argparse

import numpy as np
import moocore

from lib.api import (
    build_session,
    get_generations_from_experiment,
    get_experiment_pareto_front,
)
from plot_pareto_results import (
    fast_nondominated_sort,
    compute_worst_point,
    to_minimization_array,
    compute_gd,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HV and GD per generation")
    parser.add_argument("--api-base", default="http://localhost:8000/api/v1")
    parser.add_argument("--api-key", default=os.getenv("SIMLAB_API_KEY", "api-password"))
    parser.add_argument("--expid", required=True)
    parser.add_argument("--objectives", nargs="+", required=True)
    parser.add_argument("--minimize", nargs="+", required=True)
    parser.add_argument(
        "--true-front-bench",
        choices=["DTLZ2", "ZDT1", "SCH1"],
        default=None,
        help=(
            "For synthetic experiments: use the benchmark's analytical Pareto "
            "front as the GD reference (convergence to the true optimum) instead "
            "of the experiment's own stored final front (self-reference)."
        ),
    )
    parser.add_argument(
        "--true-front-m",
        type=int,
        default=None,
        help="Number of objectives for the analytical front (default: len(objectives)).",
    )
    args = parser.parse_args()
    minimize: list[bool] = [s.lower() == "true" for s in args.minimize]
    objectives: list[str] = args.objectives

    session = build_session(args.api_key)

    individuals_per_gen = get_generations_from_experiment(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
        label_objectives=objectives,
    )

    if not individuals_per_gen:
        # No evaluated individuals — nothing to measure
        print(json.dumps({"generations": [], "hv": [], "gd": [], "worst_point": {}}))
        return

    # GD reference front — either the benchmark's analytical (true) front or the
    # experiment's own stored final front.
    if args.true_front_bench:
        from lib.true_fronts import sample_true_front
        m = args.true_front_m or len(objectives)
        true_front = sample_true_front(args.true_front_bench, m)
        final_front_min = to_minimization_array(true_front, objectives=objectives, minimize=minimize)
    else:
        stored_pf = get_experiment_pareto_front(
            session=session,
            api_base=args.api_base,
            experiment_id=args.expid,
        )
        if not stored_pf:
            # No reference front available yet — return empty result
            print(json.dumps({"generations": [], "hv": [], "gd": [], "worst_point": {}}))
            return
        # GD reference front: unique objective vectors from stored Pareto front (min space)
        stored_matrix = np.array([
            [p["objectives"][o] for o in objectives]
            for p in stored_pf
        ])
        stored_unique = np.unique(stored_matrix, axis=0)
        final_front_min = to_minimization_array(stored_unique, objectives=objectives, minimize=minimize)

    # Reference point: worst feasible objective + 5% margin
    worst_raw = compute_worst_point(individuals_per_gen, tuple(objectives), minimize=minimize)
    worst_point_ref = [coord + abs(coord) * 0.05 + 1.0 for coord in worst_raw]

    generations_sorted = sorted(individuals_per_gen.keys())
    hv_values: list[float] = []
    gd_values: list[float | None] = []

    for gen in generations_sorted:
        inds = individuals_per_gen[gen]
        if not inds:
            hv_values.append(0.0)
            gd_values.append(None)
            continue

        local_pop = [
            {"id": ind["id"], "generation": gen, "objectives": ind["objectives"]}
            for ind in inds
        ]
        local_fronts = fast_nondominated_sort(local_pop, objectives, minimize)
        if not local_fronts:
            hv_values.append(0.0)
            gd_values.append(None)
            continue

        pts = np.array([
            [p["objectives"][o] for o in objectives]
            for p in local_fronts[0]
        ])
        pts = np.unique(pts, axis=0)
        pts_min = to_minimization_array(pts, objectives=objectives, minimize=minimize)

        hv_val = float(moocore.hypervolume(pts_min, ref=worst_point_ref))
        gd_val = float(compute_gd(pts_min, final_front_min))
        hv_values.append(hv_val)
        gd_values.append(gd_val)

    print(json.dumps({
        "generations": generations_sorted,
        "hv": hv_values,
        "gd": gd_values,
        "worst_point": dict(zip(objectives, worst_point_ref)),
    }))


if __name__ == "__main__":
    main()
