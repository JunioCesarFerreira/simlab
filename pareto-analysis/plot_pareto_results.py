import os
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from lib.api import (
    build_session,
    get_individuals_per_generation_api,
    upload_analysis_file_api
)

# ------------------------------------------------------------
# Pareto dominance
# ------------------------------------------------------------

def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    Objectives:
      - latency:     minimize
      - energy:      minimize
      - throughput:  maximize
    """
    better_or_equal = (
        a["latency"] <= b["latency"] and
        a["energy"] <= b["energy"] and
        a["throughput"] >= b["throughput"]
    )

    strictly_better = (
        a["latency"] < b["latency"] or
        a["energy"] < b["energy"] or
        a["throughput"] > b["throughput"]
    )

    return better_or_equal and strictly_better


# ------------------------------------------------------------
# Fast non-dominated sorting
# ------------------------------------------------------------

def fast_nondominated_sort(
    population: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    S = {}
    n = {}
    fronts: List[List[Dict[str, Any]]] = [[]]

    for p in population:
        pid = p["id"]
        S[pid] = []
        n[pid] = 0

        for q in population:
            if pid == q["id"]:
                continue

            if dominates(p["objectives"], q["objectives"]):
                S[pid].append(q)
            elif dominates(q["objectives"], p["objectives"]):
                n[pid] += 1

        if n[pid] == 0:
            p["rank"] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p["id"]]:
                n[q["id"]] -= 1
                if n[q["id"]] == 0:
                    q["rank"] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


# ------------------------------------------------------------
# Population builder (API → flat list)
# ------------------------------------------------------------

def build_population_from_api(
    individuals_by_generation: dict[int, list[dict]]
) -> list[dict]:
    population = []

    for gen, inds in individuals_by_generation.items():
        for ind in inds:
            population.append({
                "id": ind["simulation_id"],
                "generation": gen,
                "objectives": ind["objectives"]
            })

    return population


def build_pareto_by_front(fronts: list[list[dict]]) -> dict[int, list[dict]]:
    return {i: front for i, front in enumerate(fronts)}


# ------------------------------------------------------------
# Plot: Pareto fronts (same visual pattern)
# ------------------------------------------------------------

def plot_pareto_fronts(
    pareto_by_front: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    output_path: Path
):
    fronts = sorted(pareto_by_front.keys())
    num_fronts = len(fronts)

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, num_fronts))

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    pairs = [(1, 0), (1, 2), (0, 2)]

    # ---------- 2D projections ----------
    for idx_pair, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, idx_pair])

        for idx_f, f in enumerate(fronts):
            front = pareto_by_front[f]
            if not front:
                continue

            x = [p["objectives"][objective_names[i]] for p in front]
            y = [p["objectives"][objective_names[j]] for p in front]

            ax.scatter(x, y, color=colors[idx_f], alpha=0.85)

        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.set_title(f"{objective_names[i]} vs {objective_names[j]}")
        ax.grid(True)

    # ---------- 3D Pareto ----------
    ax3d_a = fig.add_subplot(gs[1, 0], projection="3d")

    for idx_f, f in enumerate(fronts):
        front = pareto_by_front[f]
        if not front:
            continue

        xs = [p["objectives"][objective_names[0]] for p in front]
        ys = [p["objectives"][objective_names[1]] for p in front]
        zs = [p["objectives"][objective_names[2]] for p in front]

        ax3d_a.scatter(xs, ys, zs, color=colors[idx_f], alpha=0.85)

    ax3d_a.set_xlabel(objective_names[0])
    ax3d_a.set_ylabel(objective_names[1])
    ax3d_a.set_zlabel(objective_names[2])
    ax3d_a.set_title("Pareto Fronts (view XYZ)")
    
    # ---------- 3D Pareto ----------
    ax3d_b = fig.add_subplot(gs[1, 1], projection="3d")

    legend_handles = []
    legend_labels = []

    for idx_f, f in enumerate(fronts):
        front = pareto_by_front[f]
        if not front:
            continue

        xs = [p["objectives"][objective_names[1]] for p in front]
        ys = [p["objectives"][objective_names[0]] for p in front]
        zs = [p["objectives"][objective_names[2]] for p in front]

        sc = ax3d_b.scatter(xs, ys, zs, color=colors[idx_f], alpha=0.85)
        legend_handles.append(sc)
        legend_labels.append(f"F{f}")

    ax3d_b.set_xlabel(objective_names[1])
    ax3d_b.set_ylabel(objective_names[0])
    ax3d_b.set_zlabel(objective_names[2])
    ax3d_b.set_title("Pareto Fronts (swapped axes)")

    ax3d_b.legend(
        legend_handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        title="Fronts"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Plot: distribution by generation × front
# ------------------------------------------------------------

def plot_generation_front_distribution(
    population: list[dict],
    output_path: Path
):
    counts = defaultdict(lambda: defaultdict(int))

    for ind in population:
        counts[ind["generation"]][ind["rank"]] += 1

    generations = sorted(counts.keys())
    fronts = sorted({ind["rank"] for ind in population})

    x = np.arange(len(generations))
    bar_width = 0.8 / len(fronts)

    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(fronts)))

    fig, ax = plt.subplots(figsize=(18, 6))

    for idx_f, f in enumerate(fronts):
        values = [counts[g].get(f, 0) for g in generations]
        ax.bar(
            x + idx_f * bar_width,
            values,
            width=bar_width,
            color=colors[idx_f],
            label=f"F{f}"
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Individuals")
    ax.set_title("Individuals per Pareto Front by Generation")

    ax.set_xticks(x + bar_width * (len(fronts) - 1) / 2)
    ax.set_xticklabels(generations)

    ax.legend(title="Pareto Fronts")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pareto front dominance analysis (API-based)"
    )

    parser.add_argument("--api-base", default="http://localhost:8000/api/v1")
    parser.add_argument("--api-key", default=os.getenv("SIMLAB_API_KEY", ""))
    parser.add_argument("--expid", required=True)

    parser.add_argument(
        "--objectives",
        nargs=3,
        default=("latency", "energy", "throughput"),
        metavar=("X", "Y", "Z")
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key")

    session = build_session(args.api_key)

    individuals_per_gen = get_individuals_per_generation_api(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid
    )

    population = build_population_from_api(individuals_per_gen)
    fronts = fast_nondominated_sort(population)
    pareto_by_front = build_pareto_by_front(fronts)

    pareto_plot = Path(f"pareto_fronts_{args.expid}.png")
    dist_plot = Path(f"pareto_distribution_{args.expid}.png")

    plot_pareto_fronts(
        pareto_by_front,
        tuple(args.objectives),
        pareto_plot
    )

    plot_generation_front_distribution(
        population,
        dist_plot
    )

    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        pareto_plot,
        "pareto_fronts",
        "Pareto fronts (dominance layers)"
    )

    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        dist_plot,
        "pareto_distribution",
        "Distribution of individuals per front and generation"
    )

    print("[OK] Pareto dominance analysis completed")

    # Cleanup
    try:
        pareto_plot.unlink(missing_ok=True)
        dist_plot.unlink(missing_ok=True)
        print("[OK] Temporary files removed")
    except Exception as ex:
        print(f"[WARN] Failed to remove temporary file: {ex}")

if __name__ == "__main__":
    main()
