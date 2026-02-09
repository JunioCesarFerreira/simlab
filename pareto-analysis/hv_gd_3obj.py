import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from deap.tools._hypervolume import hv

from lib.api import (
    get_pareto_per_generation_api,
    build_session,
    upload_analysis_file_api
)

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_hypervolume(front: list[list[float]], ref_point: list[float]) -> float:
    if not front:
        print("Warning: Empty front for hypervolume computation")
        return 0.0
    return hv.hypervolume(front, ref_point)


def compute_gd(front: np.ndarray, ref_front: np.ndarray) -> float:
    if len(front) == 0 or len(ref_front) == 0:
        return float("inf")

    dists = []
    for p in front:
        dist = np.min(np.linalg.norm(ref_front - p, axis=1))
        dists.append(dist)

    return float(np.mean(dists))


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_hv_gd(
    generations: list[int],
    hv_values: list[float],
    gd_values: list[float],
    output_path: Path
):
    fig, (ax_hv, ax_gd) = plt.subplots(
        1, 2,
        figsize=(14, 5),
        sharex=True
    )
    
    generations = sorted(generations)

    # -------------------------------
    # Hypervolume
    # -------------------------------
    ax_hv.plot(generations, hv_values, marker="o")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("Hypervolume Evolution")
    ax_hv.grid(True)

    # -------------------------------
    # Generational Distance
    # -------------------------------
    ax_gd.plot(generations, gd_values, marker="s")
    ax_gd.set_ylabel("Generational Distance")
    ax_gd.set_title("Generational Distance Evolution")
    ax_gd.grid(True)
    
    for ax in (ax_hv, ax_gd):
        ax.set_xlabel("Generation")
        ax.set_xticks(generations)
        ax.set_xticklabels([str(g) for g in generations])

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def to_minimization_array(points: np.ndarray, minimize: list[bool]) -> np.ndarray:
    """
    Convert objectives to an equivalent minimization space.
    For max objectives, multiply by -1 so that 'smaller is better' holds.
    """
    out = points.astype(float).copy()
    for j, is_min in enumerate(minimize):
        if not is_min:
            out[:, j] *= -1.0
    return out


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute Hypervolume and Generational Distance for Pareto fronts (via API)"
    )

    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/api/v1",
        help="Base URL of the SimLab API"
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("SIMLAB_API_KEY", ""),
        help="API key for SimLab (or env SIMLAB_API_KEY)"
    )

    parser.add_argument(
        "--expid",
        required=True,
        help="Experiment ID"
    )

    parser.add_argument(
        "--objectives",
        nargs=3,
        default=("energy", "latency", "throughput"),
        metavar=("X", "Y", "Z"),
        help="Objective names (order matters)"
    )

    parser.add_argument(
        "--ref-point",
        nargs=3,
        type=float,
        required=True,
        metavar=("RX", "RY", "RZ"),
        help="Reference point for hypervolume (minimization)"
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key (or env SIMLAB_API_KEY)")

    session = build_session(args.api_key)

    # ------------------------------------------------------------
    # Fetch Pareto fronts
    # ------------------------------------------------------------
    pareto_per_gen = get_pareto_per_generation_api(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
        to_minimization=True
    )

    generations = sorted(pareto_per_gen.keys())

    if not generations:
        raise SystemExit("No Pareto fronts found")

    # Final Pareto = reference for GD
    final_gen = generations[-1]
    final_front = np.array([
        [p["objectives"][o] for o in args.objectives]
        for p in pareto_per_gen[final_gen]
    ])

    hv_values = []
    gd_values = []

    for gen in generations:
        front = pareto_per_gen[gen]

        points = np.array([
            [p["objectives"][o] for o in args.objectives]
            for p in front
        ])
        
        points = to_minimization_array(points, minimize=[True, True, False])
        final_front = to_minimization_array(final_front, minimize=[True, True, False])

        hv_val = compute_hypervolume(points.tolist(), args.ref_point)
        gd_val = compute_gd(points, final_front)

        hv_values.append(hv_val)
        gd_values.append(gd_val)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    output_file = Path(f"hv_gd_{args.expid}.png")

    plot_hv_gd(
        generations=generations,
        hv_values=hv_values,
        gd_values=gd_values,
        output_path=output_file
    )

    print(f"[OK] HV/GD plot generated: {output_file}")

    # ------------------------------------------------------------
    # Upload + attach
    # ------------------------------------------------------------
    upload_analysis_file_api(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
        path=output_file,
        name="hv_gd_plot",
        description="Hypervolume and generational distance evolution"
    )

    print("[OK] HV/GD plot uploaded and registered in experiment")

    # Cleanup
    try:
        output_file.unlink(missing_ok=True)
        print("[OK] Temporary file removed")
    except Exception as ex:
        print(f"[WARN] Failed to remove temporary file: {ex}")


if __name__ == "__main__":
    main()
