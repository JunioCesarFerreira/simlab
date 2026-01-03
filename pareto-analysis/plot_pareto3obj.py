import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from lib.database_methods import (
    get_pareto_per_generation,
    upload_file,
    add_analysis_file_to_experiment
)


def plot_pareto_generations(
    pareto_by_generation: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    output_path: Path
):
    gens = sorted(pareto_by_generation.keys())
    num_gens = len(gens)

    colors = plt.cm.viridis(np.linspace(0.3, 1.0, num_gens))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    pairs = [(0, 1), (0, 2), (1, 2)]
    axes_2d = []

    # ---------- 2D projections ----------
    for idx_pair, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, idx_pair])
        axes_2d.append(ax)

        for idx_gen, gen in enumerate(gens):
            front = pareto_by_generation[gen]
            if not front:
                continue

            x = [p["objectives"][objective_names[i]] for p in front]
            y = [p["objectives"][objective_names[j]] for p in front]

            ax.scatter(
                x,
                y,
                color=colors[idx_gen],
                alpha=0.85
            )

        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.set_title(f"{objective_names[i]} vs {objective_names[j]}")
        ax.grid(True)

    # Shared legend (first & last generation)
    handles, labels = axes_2d[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    # ---------- 3D Pareto ----------
    ax3d = fig.add_subplot(gs[1, :], projection="3d")

    legend_handles = []
    legend_labels = []

    for idx_gen, gen in enumerate(gens):
        front = pareto_by_generation[gen]
        if not front:
            continue

        xs = [p["objectives"][objective_names[0]] for p in front]
        ys = [p["objectives"][objective_names[1]] for p in front]
        zs = [p["objectives"][objective_names[2]] for p in front]

        sc = ax3d.scatter(
            xs, ys, zs,
            color=colors[idx_gen],
            alpha=0.85
        )

        # Build legend entries explicitly
        legend_handles.append(sc)
        legend_labels.append(f"Gen {gen}")

    ax3d.set_xlabel(objective_names[0])
    ax3d.set_ylabel(objective_names[1])
    ax3d.set_zlabel(objective_names[2])
    ax3d.set_title("Pareto Front Evolution (3D)")

    # ---- Legend to the right of the 3D plot ----
    ax3d.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.25, 0.5),
        borderaxespad=0.0,
        title="Generations"
    )


    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Pareto front evolution plots for a SimLab experiment"
    )
    parser.add_argument(
        "--expid",
        required=True,
        help="MongoDB ObjectId of the experiment"
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27017/?replicaSet=rs0"
    )
    parser.add_argument(
        "--db-name",
        default="simlab"
    )
    parser.add_argument(
        "--objectives",
        nargs=3,
        default=("energy", "latency", "throughput"),
        metavar=("X", "Y", "Z"),
        help="Names of the three objectives (default: energy latency throughput)"
    )

    args = parser.parse_args()

    pareto_per_gen = get_pareto_per_generation(
        args.mongo_uri,
        args.db_name,
        args.expid
    )

    output_file = f"pareto_evolution_{args.expid}.png"

    plot_pareto_generations(
        pareto_by_generation=pareto_per_gen,
        objective_names=tuple(args.objectives),
        output_path=output_file
    )

    print(f"[OK] Pareto temporary plot saved at: {output_file}")
    
    file_id = upload_file(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        path=output_file,
        name=f"pareto_evolution_{args.expid}.png"
    )
    
    print(f"[OK] Pareto plot uploaded to database with File ID: {file_id}")
    
    add_analysis_file_to_experiment(
        mongo_uri=args.mongo_uri,
        db_name=args.db_name,
        experiment_id=args.expid,
        description="Pareto front evolution (2D projections and 3D view)",
        file_id=file_id
    )

    print("[OK] Analysis file registered in experiment")
        
    try:
        Path(output_file).unlink(missing_ok=True)
    except Exception as ex:
        print("Failed to remove temporary file %s: %s", output_file, ex)
        
    print(f"[OK] Temporary files removed.")
    


if __name__ == "__main__":
    main()
