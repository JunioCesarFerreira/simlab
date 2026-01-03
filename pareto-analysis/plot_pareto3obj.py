import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lib.pareto_by_generation import get_pareto_per_generation


def plot_pareto_generations(
    pareto_by_generation: dict[int, list[dict]],
    objective_names: tuple[str, str, str]
):
    gens = sorted(pareto_by_generation.keys())
    num_gens = len(gens)

    # Color gradient (early -> light, final -> strong)
    colors = plt.cm.viridis(np.linspace(0.3, 1.0, num_gens))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

    pairs = [(0, 1), (0, 2), (1, 2)]
    axes_2d = []

    # --- Top row: 2D projections ---
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
                alpha=0.85,
                label=f"Gen {gen}" if idx_gen in (0, num_gens - 1) else None
            )

        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.set_title(f"{objective_names[i]} vs {objective_names[j]}")
        ax.grid(True)

    # Shared legend (only first & last gen)
    handles, labels = axes_2d[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    # --- Bottom row: 3D Pareto ---
    ax3d = fig.add_subplot(gs[1, :], projection="3d")

    for idx_gen, gen in enumerate(gens):
        front = pareto_by_generation[gen]
        if not front:
            continue

        xs = [p["objectives"][objective_names[0]] for p in front]
        ys = [p["objectives"][objective_names[1]] for p in front]
        zs = [p["objectives"][objective_names[2]] for p in front]

        ax3d.scatter(
            xs, ys, zs,
            color=colors[idx_gen],
            alpha=0.85
        )

    ax3d.set_xlabel(objective_names[0])
    ax3d.set_ylabel(objective_names[1])
    ax3d.set_zlabel(objective_names[2])
    ax3d.set_title("Pareto Front Evolution (3D)")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Execution
# --------------------------------------------------

MONGO_URI = "mongodb://localhost:27017/?replicaSet=rs0"
DB_NAME = "simlab"
EXP_ID = "69582861f8e55d6ba70f0687"

pareto_per_gen = get_pareto_per_generation(
    MONGO_URI,
    DB_NAME,
    EXP_ID
)

plot_pareto_generations(
    pareto_by_generation=pareto_per_gen,
    objective_names=("energy", "latency", "throughput")
)
