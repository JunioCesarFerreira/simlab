"""
Compute Hypervolume (HV), Generational Distance (GD) and Inverted Generational
Distance (IGD / IGD+) per generation.  Outputs a single JSON object to stdout.

The metric loop itself lives in ``plot_pareto_results.compute_convergence_metrics``
— the same function the plotting CLI uses — so the numbers printed here, the
numbers in the uploaded PNG and the numbers the web GUI reads from
``GET /experiments/{id}/hv-gd`` all come from one definition.
"""
import json
import os
import argparse

import numpy as np

from lib import metrics
from lib.api import (
    build_session,
    get_generations_from_experiment,
    get_experiment_pareto_front,
)
from plot_pareto_results import (
    compute_worst_point,
    to_minimization_array,
    compute_convergence_metrics,
)


def _empty() -> str:
    return json.dumps({
        "generations": [], "hv": [], "hv_cumulative": [],
        "gd": [], "igd": [], "igd_plus": [],
        "reference": None, "reference_size": 0, "normalized": False,
        "worst_point": {}, "gd_method": None, "gd_formula": None, "normalization": None,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HV, GD and IGD per generation")
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
            "front as the GD/IGD reference (convergence to the true optimum) "
            "instead of the experiment's own stored final front (self-reference)."
        ),
    )
    parser.add_argument(
        "--true-front-m",
        type=int,
        default=None,
        help="Number of objectives for the analytical front (default: len(objectives)).",
    )
    parser.add_argument(
        "--raw-distances",
        action="store_true",
        default=False,
        help=(
            "Report GD/IGD/IGD+ in raw objective units instead of normalizing "
            "by the reference front's ideal-nadir range."
        ),
    )
    args = parser.parse_args()
    minimize: list[bool] = [s.lower() == "true" for s in args.minimize]
    objectives: list[str] = args.objectives
    normalized: bool = not args.raw_distances

    session = build_session(args.api_key)

    individuals_per_gen = get_generations_from_experiment(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
        label_objectives=objectives,
    )

    if not individuals_per_gen:
        # No evaluated individuals — nothing to measure
        print(_empty())
        return

    # Reference front — either the benchmark's analytical (true) front or the
    # experiment's own stored final front.
    bench = args.true_front_bench
    m = args.true_front_m or len(objectives)
    if bench:
        from lib.true_fronts import sample_true_front
        true_front = sample_true_front(bench, m)
        reference_rows = to_minimization_array(true_front, objectives=objectives, minimize=minimize)
        reference_kind = "true_front"
    else:
        stored_pf = get_experiment_pareto_front(
            session=session,
            api_base=args.api_base,
            experiment_id=args.expid,
        )
        if not stored_pf:
            # No reference front available yet — return empty result
            print(_empty())
            return
        stored_matrix = np.array([
            [p["objectives"][o] for o in objectives]
            for p in stored_pf
        ])
        reference_rows = to_minimization_array(stored_matrix, objectives=objectives, minimize=minimize)
        reference_kind = "final_front"

    # Penalized, duplicate and dominated rows would corrupt IGD, which averages
    # over the reference front rather than minimizing over it.
    reference_front = metrics.sanitize_reference_front(reference_rows)
    if reference_front.size == 0:
        print(_empty())
        return

    # HV reference point. With a known benchmark it is the FIXED analytical
    # nadir + 10%, matching the /hv-gd endpoint and plot_pareto_results — this
    # used to derive it from the observed worst point regardless, so the same
    # experiment reported one HV here and another one in the GUI, and no two
    # runs were comparable. Without a benchmark the observed worst point is all
    # there is.
    if bench:
        from lib.true_fronts import true_nadir
        worst_point_ref = [v * 1.1 for v in true_nadir(bench, m)]
    else:
        worst_raw = compute_worst_point(individuals_per_gen, tuple(objectives), minimize=minimize)
        worst_point_ref = [coord + abs(coord) * 0.05 + 1.0 for coord in worst_raw]

    conv = compute_convergence_metrics(
        individuals_per_gen=individuals_per_gen,
        objectives=objectives,
        minimize=minimize,
        hv_ref=worst_point_ref,
        reference_front_min=reference_front,
        normalized=normalized,
        bench=bench,
    )

    # NaN marks a generation with no feasible individual; JSON has no NaN, so it
    # is emitted as null and the consumer draws a gap instead of a fake zero.
    def _series(values: list[float]) -> list[float | None]:
        return [None if v is None or np.isnan(v) else float(v) for v in values]

    print(json.dumps({
        "generations": conv.generations,
        "hv": conv.hv,
        "hv_cumulative": conv.hv_cumulative,
        "gd": _series(conv.gd),
        "igd": _series(conv.igd),
        "igd_plus": _series(conv.igd_plus),
        "reference": reference_kind,
        "reference_size": int(len(reference_front)),
        "normalized": normalized,
        "worst_point": dict(zip(objectives, worst_point_ref)),
        "gd_method": "analytical" if bench else "reference_front",
        "gd_formula": "mean of each front point's distance to the true front (p=1)",
        "normalization": "analytical ideal-nadir range" if bench
                         else "reference front ideal-nadir range",
    }))


if __name__ == "__main__":
    main()
