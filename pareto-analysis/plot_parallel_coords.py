import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lib.api import (
    get_pareto_per_generation_api,
    build_session,
    upload_analysis_file_api
)


# -------------------------------------------------------------------
# Plotting utilities
# -------------------------------------------------------------------

def normalize_objectives(matrix: np.ndarray) -> np.ndarray:
    """
    Min-max normalize objective values column-wise to [0, 1].

    :param matrix: Array of shape (n_solutions, n_objectives)
    :return: Normalized array of same shape
    """
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)

    # Avoid division by zero
    denom = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
    return (matrix - mins) / denom



# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

def plot_parallel_coordinates(
    pareto_by_generation: dict[int, list[dict]],
    objective_names: tuple[str, ...],
    output_path: Path
):
    """
    Generate two parallel coordinates plots:
      (1) All Pareto-optimal solutions across generations (top)
      (2) Final Pareto front only, with simulation IDs in the legend (bottom)

    :param pareto_by_generation: Dict[generation_index -> list of Pareto solutions]
    :param objective_names: Ordered objective names
    :param output_path: Output image path
    """
    generations = sorted(pareto_by_generation.keys())
    num_objectives = len(objective_names)

    if num_objectives < 2:
        raise ValueError("Parallel coordinates require at least two objectives")

    x_positions = np.arange(num_objectives)
    colors = plt.cm.viridis(np.linspace(0.3, 1.0, len(generations)))

    fig, (ax_all, ax_final) = plt.subplots(
        2, 1, figsize=(18, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]}
    )

    # ================================================================
    # Top plot — All generations
    # ================================================================

    legend_handles = []
    legend_labels = []

    for idx_gen, gen in enumerate(generations):
        front = pareto_by_generation[gen]
        if not front:
            continue

        values = np.array([
            [p["objectives"][obj] for obj in objective_names]
            for p in front
        ])

        values_norm = normalize_objectives(values)

        for row in values_norm:
            ax_all.plot(
                x_positions,
                row,
                color=colors[idx_gen],
                alpha=0.6,
                linewidth=1.2
            )

        proxy, = ax_all.plot([], [], color=colors[idx_gen], linewidth=2)
        legend_handles.append(proxy)
        legend_labels.append(f"Generation {gen}")

    ax_all.set_ylabel("Normalized Objective Value")
    ax_all.set_title("Parallel Coordinates — Pareto-Optimal Solutions (All Generations)")
    ax_all.grid(True, axis="y", alpha=0.3)

    ax_all.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="Generations"
    )

    # ================================================================
    # Bottom plot — Final Pareto front only
    # ================================================================

    final_gen = generations[-1]
    final_front = pareto_by_generation.get(final_gen, [])

    if final_front:
        values = np.array([
            [p["objectives"][obj] for obj in objective_names]
            for p in final_front
        ])

        values_norm = normalize_objectives(values)

        cmap = plt.cm.tab20
        colors_final = cmap(np.linspace(0, 1, len(values_norm)))

        final_handles = []
        final_labels = []

        for idx, (row, sol) in enumerate(zip(values_norm, final_front)):
            h, = ax_final.plot(
                x_positions,
                row,
                color=colors_final[idx],
                linewidth=2.0,
                alpha=0.9
            )

            final_handles.append(h)
            final_labels.append(f"Sim {sol['simulation_id']}")

        ax_final.legend(
            final_handles,
            final_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Simulation ID"
        )

    ax_final.set_xticks(x_positions)
    ax_final.set_xticklabels(objective_names, rotation=30, ha="right")
    ax_final.set_xlabel("Objectives")
    ax_final.set_ylabel("Normalized Objective Value")
    ax_final.set_title(f"Parallel Coordinates — Final Pareto Front (Generation {final_gen})")
    ax_final.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate parallel coordinates plots for Pareto fronts (via SimLab API)"
    )

    parser.add_argument(
        "--api-base",
        default="http://localhost:8000/api/v1",
        help="Base URL of the SimLab API"
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("SIMLAB_API_KEY", ""),
        help="API key for SimLab (or set env SIMLAB_API_KEY)"
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
        help="Names of the three objectives"
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key (or env SIMLAB_API_KEY)")

    session = build_session(args.api_key)

    pareto_per_gen = get_pareto_per_generation_api(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid
    )

    output_file = Path(f"parallel_coordinates_{args.expid}.png")

    plot_parallel_coordinates(
        pareto_by_generation=pareto_per_gen,
        objective_names=tuple(args.objectives),
        output_path=output_file
    )

    print(f"[OK] Parallel coordinates plot generated: {output_file}")

    upload_analysis_file_api(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
        path=output_file,
        name="parallel_coordinates",
        description="Parallel coordinates plot of Pareto-optimal solutions (normalized objectives)"
    )

    print("[OK] Plot uploaded and registered in experiment")

    try:
        output_file.unlink(missing_ok=True)
        print("[OK] Temporary file removed")
    except Exception as ex:
        print(f"[WARN] Failed to remove temporary file: {ex}")


if __name__ == "__main__":
    main()
