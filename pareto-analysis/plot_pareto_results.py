import os
import argparse
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from deap.tools._hypervolume import hv

from lib.api import (
    build_session,
    get_pareto_per_generation_api,
    get_individuals_per_generation_api,
    upload_analysis_file_api
)

# ------------------------------------------------------------
# Pareto dominance
# ------------------------------------------------------------
def dominates(
    a: dict[str, float], 
    b: dict[str, float],
    objectives: list[str] = ["latency", "energy", "throughput"],
    minimize: list[bool] = [True, True, False]
    ) -> bool:
    """
    Returns True if solution `a` Pareto-dominates solution `b`.

    Parameters
    ----------
    a, b : dict[str, float]
        Objective vectors.
    objectives : list[str]
        Objective names to evaluate.
    minimize : list[bool]
        Orientation vector:
            True  -> minimize objective
            False -> maximize objective

    Dominance definition
    --------------------
    a dominates b iff:
        ∀i: a_i <= b_i   (min)   or   a_i >= b_i (max)
        ∃j: a_j <  b_j   (min)   or   a_j >  b_j (max)
    """

    if len(objectives) != len(minimize):
        raise ValueError("`objectives` and `minimize` must have same length")

    better_or_equal = True
    strictly_better = False

    for obj, is_min in zip(objectives, minimize):
        va = a[obj]
        vb = b[obj]

        if is_min:
            if va > vb:
                better_or_equal = False
                break
            if va < vb:
                strictly_better = True
        else:  # maximization
            if va < vb:
                better_or_equal = False
                break
            if va > vb:
                strictly_better = True

    return better_or_equal and strictly_better


# ------------------------------------------------------------
# Fast non-dominated sorting
# ------------------------------------------------------------
def fast_nondominated_sort(
    population: list[dict[str, Any]],
    objectives: list[str] = ["latency", "energy", "throughput"],
    minimize: list[bool] = [True, True, False]
) -> list[list[dict[str, Any]]]:
    S = {}
    n = {}
    fronts: list[list[dict[str, Any]]] = [[]]

    for p in population:
        pid = p["id"]
        S[pid] = []
        n[pid] = 0

        for q in population:
            if pid == q["id"]:
                continue

            if dominates(p["objectives"], q["objectives"], objectives=objectives, minimize=minimize):
                S[pid].append(q)
            elif dominates(q["objectives"], p["objectives"], objectives=objectives, minimize=minimize):
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


def convert_pareto_by_front(fronts: list[list[dict]]) -> dict[int, list[dict]]:
    return {i: front for i, front in enumerate(fronts)}


def normalize_objectives(
    values: np.ndarray,
    minimize: list[bool]
    ) -> np.ndarray:
    """
    Min-max normalization.
    If objective is to minimize, invert scale so that
    'better' is always higher after normalization.
    """
    vmin = values.min(axis=0)
    vmax = values.max(axis=0)
    norm = (values - vmin) / (vmax - vmin + 1e-12)

    for i, is_min in enumerate(minimize):
        if is_min:
            norm[:, i] = 1.0 - norm[:, i]

    return norm


def to_minimization_array(
    points: np.ndarray,
    objectives: list[str],
    minimize: list[bool],
) -> np.ndarray:
    """
    Convert objective matrix to an equivalent minimization space.

    Parameters
    ----------
    points : np.ndarray
        Objective matrix of shape (N, M), where:
            N = number of solutions
            M = number of objectives
    objectives : list[str]
        Objective names (metadata / ordering reference).
    minimize : list[bool]
        Orientation vector:
            True  -> objective already minimized
            False -> objective is maximized (will be inverted)

    Returns
    -------
    np.ndarray
        Transformed matrix where all objectives follow
        'smaller is better' semantics.

    Transformation
    --------------
    For each maximization objective j:

        f'_j(x) = - f_j(x)

    This preserves Pareto dominance relations.
    """

    # --- Shape normalization ---
    points = np.asarray(points, dtype=float)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    n_obj_matrix = points.shape[1]
    n_obj_meta = len(objectives)

    if n_obj_matrix != n_obj_meta:
        raise ValueError(
            f"Points have {n_obj_matrix} objectives, "
            f"but metadata defines {n_obj_meta}"
        )

    if len(minimize) != n_obj_meta:
        raise ValueError(
            "`objectives` and `minimize` must have same length"
        )

    # --- Copy to avoid mutating original array ---
    out = points.copy()

    # --- Orientation transform ---
    for j, is_min in enumerate(minimize):
        if not is_min:
            out[:, j] *= -1.0

    return out


def compute_worst_point(
    pareto_per_gen: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    minimize: list[bool]
) -> list[float]:
    all_points = []

    for fronts in pareto_per_gen.values():
        for p in fronts:
            all_points.append(
                [p["objectives"][o] for o in objective_names]
            )
            
    all_points = to_minimization_array(np.array(all_points), objectives=objective_names, minimize=minimize)

    arr = np.array(all_points)
    return arr.max(axis=0).tolist()

def compute_gd(front: np.ndarray, ref_front: np.ndarray) -> float:
    if len(front) == 0 or len(ref_front) == 0:
        return float("inf")

    dists = []
    for p in front:
        dist = np.min(np.linalg.norm(ref_front - p, axis=1))
        dists.append(dist)

    return float(np.sqrt(np.mean(np.square(dists))))

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
        gen = ind.get("generation")
        rank = ind.get("rank")
        if rank is None:
            rank = 999
            ind["rank"] = rank
        counts[gen][rank] += 1

    generations = sorted(counts.keys())
    fronts = sorted({ind["rank"] for ind in population})

    x = np.arange(len(generations))
    bar_width = 1 / len(fronts)

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
    

def plot_front_per_generation_distribution(
    pareto_per_generation: dict[int, list[dict[str, Any]]],
    output_path: Path
):
    """
    Plot distribution of Pareto fronts per generation.

    The non-dominated sorting is computed independently
    inside each generation.
    """

    # ------------------------------------------------------------
    # Build counts[generation][front] = number of individuals
    # ------------------------------------------------------------
    counts = defaultdict(lambda: defaultdict(int))

    for gen, individuals in pareto_per_generation.items():

        # --------------------------------------------
        # Build local population (generation only)
        # --------------------------------------------
        local_population = []

        for ind in individuals:
            local_population.append({
                "id": ind["simulation_id"],
                "generation": gen,
                "objectives": ind["objectives"]
            })

        if not local_population:
            continue

        # --------------------------------------------
        # Fast non-dominated sorting (intra-generation)
        # --------------------------------------------
        fronts = fast_nondominated_sort(local_population)

        for f_idx, front in enumerate(fronts):
            counts[gen][f_idx] = len(front)

    # ------------------------------------------------------------
    # Axes construction
    # ------------------------------------------------------------
    generations = sorted(counts.keys())

    all_fronts = sorted({
        f for gen_counts in counts.values()
        for f in gen_counts.keys()
    })

    if not generations or not all_fronts:
        print("[WARN] No data for front-per-generation distribution")
        return

    x = np.arange(len(generations))
    bar_width = 0.8 / len(all_fronts)

    colors = plt.cm.coolwarm(
        np.linspace(0.1, 0.9, len(all_fronts))
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 6))

    for idx_f, f in enumerate(all_fronts):

        values = [
            counts[g].get(f, 0)
            for g in generations
        ]

        ax.bar(
            x + idx_f * bar_width,
            values,
            width=bar_width,
            color=colors[idx_f],
            label=f"F{f}"
        )

    # ------------------------------------------------------------
    # Labels & layout
    # ------------------------------------------------------------
    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of Individuals")
    ax.set_title(
        "Pareto Front Distribution per Generation\n"
        "(Ranking computed intra-generation)"
    )

    ax.set_xticks(
        x + bar_width * (len(all_fronts) - 1) / 2
    )
    ax.set_xticklabels(generations)

    ax.legend(title="Pareto Fronts")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    
def plot_parallel_coordinates_pareto0(
    pareto_by_front: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    minimize: list[bool],
    output_path: Path
):
    front0 = pareto_by_front.get(0, [])
    if not front0:
        print("[WARN] Empty Pareto front 0 (parallel coordinates)")
        return

    ids = [p["id"] for p in front0]

    data = np.array([
        [p["objectives"][obj] for obj in objective_names]
        for p in front0
    ])

    norm = normalize_objectives(data, minimize)

    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(objective_names))

    cmap = colormaps["tab20"].resampled(len(front0))

    for i, row in enumerate(norm):
        ax.plot(
            x,
            row,
            color=cmap(i),
            alpha=0.8,
            linewidth=2,
            label=str(ids[i])
        )

    ax.set_xticks(x)
    ax.set_xticklabels(objective_names)
    ax.set_ylabel("Normalized objective value")
    ax.set_title("Pareto Front 0 — Parallel Coordinates")

    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax.legend(
        title="Simulation ID",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_radar_pareto0(
    pareto_by_front: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    minimize: list[bool],
    output_path: Path
):
    front0 = pareto_by_front.get(0, [])
    if not front0:
        print("[WARN] Empty Pareto front 0 (radar)")
        return

    ids = [p["id"] for p in front0]

    data = np.array([
        [p["objectives"][obj] for obj in objective_names]
        for p in front0
    ])

    norm = normalize_objectives(data, minimize)

    num_vars = len(objective_names)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, polar=True)

    cmap = colormaps["tab20"].resampled(len(front0))

    for i, row in enumerate(norm):
        values = np.concatenate([row, [row[0]]])
        ax.plot(
            angles,
            values,
            color=cmap(i),
            linewidth=2,
            alpha=0.9,
            label=str(ids[i])
        )
        ax.fill(
            angles,
            values,
            color=cmap(i),
            alpha=0.12
        )

    ax.set_thetagrids(
        angles[:-1] * 180 / np.pi,
        objective_names
    )

    ax.set_ylim(0, 1)
    ax.set_title("Pareto Front 0 — Radar Plot", pad=25)

    ax.legend(
        title="Simulation ID",
        loc="center left",
        bbox_to_anchor=(1.25, 0.5),
        fontsize=9
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hv_gd(
    generations: list[int],
    hv_values: list[float],
    gd_values: list[float],
    worst_point: list[float],
    output_path: Path
):
    fig, (ax_hv, ax_gd) = plt.subplots(
        1, 2,
        figsize=(15, 5),
        sharex=True
    )

    # -------------------------------
    # Hypervolume
    # -------------------------------
    ax_hv.plot(generations, hv_values, marker="o")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title(
        "Hypervolume Evolution\n"
        f"Reference point = {np.round(worst_point, 3).tolist()}"
    )
    ax_hv.grid(True)

    # -------------------------------
    # Generational Distance
    # -------------------------------
    ax_gd.plot(generations, gd_values, marker="s", color="tab:red")
    ax_gd.set_ylabel("Generational Distance")
    ax_gd.set_title("Generational Distance to Final Pareto Front")
    ax_gd.grid(True)

    for ax in (ax_hv, ax_gd):
        ax.set_xlabel("Generation")
        ax.set_xticks(generations)
        ax.set_xticklabels([str(g) for g in generations])

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_individual_lifetime(
    individuals_per_generation: dict[int, list[dict[str, Any]]],
    pareto_by_front: dict[int, list[dict]],
    output_path: Path
):
    """
    Plot individual survival across generations.

    Each horizontal bar represents the lifespan of an individual:
        birth_generation -> death_generation

    Color encodes the GLOBAL Pareto rank.
    """

    # ------------------------------------------------------------
    # Build presence map
    # ------------------------------------------------------------
    presence = defaultdict(list)

    for gen, individuals in individuals_per_generation.items():
        for ind in individuals:
            iid = ind["simulation_id"]
            presence[iid].append(gen)

    # ------------------------------------------------------------
    # Birth / death extraction
    # ------------------------------------------------------------
    lifetimes = []

    for iid, gens in presence.items():
        birth = min(gens)
        death = max(gens)
        lifetimes.append((iid, birth, death))

    # ------------------------------------------------------------
    # Global rank map
    # ------------------------------------------------------------
    rank_map = {}

    for rank, front in pareto_by_front.items():
        for ind in front:
            rank_map[ind["id"]] = rank

    # Default rank for missing (should not happen)
    max_rank = max(rank_map.values(), default=0)

    # ------------------------------------------------------------
    # Sorting individuals
    # Strategy: by birth → rank → death
    # ------------------------------------------------------------
    lifetimes.sort(
        key=lambda x: (
            x[1],                     # birth
            rank_map.get(x[0], max_rank),
            x[2]                      # death
        )
    )

    ids = [x[0] for x in lifetimes]
    births = [x[1] for x in lifetimes]
    deaths = [x[2] for x in lifetimes]
    durations = [d - b + 1 for b, d in zip(births, deaths)]

    # ------------------------------------------------------------
    # Color mapping by global Pareto front
    # ------------------------------------------------------------
    unique_ranks = sorted(set(rank_map.values()))
    cmap = plt.cm.coolwarm(
        np.linspace(0.1, 0.9, len(unique_ranks))
    )

    rank_to_color = {
        r: cmap[i]
        for i, r in enumerate(unique_ranks)
    }

    colors = [
        rank_to_color.get(
            rank_map.get(iid, max_rank),
            (0.5, 0.5, 0.5, 1.0)
        )
        for iid in ids
    ]

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    N = len(ids)

    HEIGHT_PER_INDIVIDUAL = 0.25
    MAX_HEIGHT_IN = 30

    height = min(MAX_HEIGHT_IN, max(6, N * HEIGHT_PER_INDIVIDUAL))

    fig, ax = plt.subplots(
        figsize=(18, height)
    )

    y_pos = np.arange(N)

    ax.barh(
        y_pos,
        durations,
        left=births,
        color=colors,
        edgecolor="black",
        alpha=0.9
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------
    ax.set_xlabel("Generation")
    ax.set_ylabel("Individual (Simulation ID)")
    ax.set_title(
        "Individual Lifetime Across Generations\n"
        "Color = Global Pareto Rank"
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ids, fontsize=8)

    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # ------------------------------------------------------------
    # Legend (Pareto fronts)
    # ------------------------------------------------------------
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=rank_to_color[r])
        for r in unique_ranks
    ]

    labels = [f"F{r}" for r in unique_ranks]

    ax.legend(
        handles,
        labels,
        title="Global Pareto Front",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_individual_lifetime_per_generation(
    individuals_per_generation: dict[int, list[dict[str, Any]]],
    pareto_by_front: dict[int, list[dict]],
    output_dir: Path
):
    """
    Generate one lifetime plot per generation.

    For each generation g:
        - individuals present in g are selected
        - survival is tracked forward
        - color = global Pareto rank
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Presence map
    # ------------------------------------------------------------
    presence = defaultdict(list)

    for gen, individuals in individuals_per_generation.items():
        for ind in individuals:
            presence[ind["simulation_id"]].append(gen)

    # ------------------------------------------------------------
    # Global rank map
    # ------------------------------------------------------------
    rank_map = {}

    for rank, front in pareto_by_front.items():
        for ind in front:
            rank_map[ind["id"]] = rank

    unique_ranks = sorted(rank_map.values())
    cmap = plt.cm.coolwarm(
        np.linspace(0.1, 0.9, len(unique_ranks))
    )

    rank_to_color = {
        r: cmap[i]
        for i, r in enumerate(unique_ranks)
    }

    generations = sorted(individuals_per_generation.keys())

    # ------------------------------------------------------------
    # Plot per generation
    # ------------------------------------------------------------
    for g in generations:

        individuals = individuals_per_generation[g]

        lifetimes = []

        for ind in individuals:
            iid = ind["simulation_id"]
            gens = presence[iid]

            birth = g
            death = max(t for t in gens if t >= g)

            lifetimes.append((iid, birth, death))

        if not lifetimes:
            continue

        # Sort by rank then survival
        lifetimes.sort(
            key=lambda x: (
                rank_map.get(x[0], 999),
                -(x[2] - x[1])
            )
        )

        ids = [x[0] for x in lifetimes]
        births = [x[1] for x in lifetimes]
        durations = [x[2] - x[1] + 1 for x in lifetimes]

        colors = [
            rank_to_color.get(
                rank_map.get(iid, 999),
                (0.5, 0.5, 0.5, 1.0)
            )
            for iid in ids
        ]

        # --------------------------------------------------------
        # Adaptive sizing
        # --------------------------------------------------------
        N = len(ids)

        HEIGHT_PER_INDIVIDUAL = 0.25
        MAX_HEIGHT_IN = 25

        height = min(
            MAX_HEIGHT_IN,
            max(5, N * HEIGHT_PER_INDIVIDUAL)
        )

        fig, ax = plt.subplots(figsize=(18, height))

        y_pos = np.arange(N)

        ax.barh(
            y_pos,
            durations,
            left=births,
            color=colors,
            edgecolor="black",
            alpha=0.9
        )

        ax.set_xlabel("Generation")
        ax.set_title(
            f"Individual Survival Starting at Generation {g}\n"
            "Color = Global Pareto Rank"
        )

        if N > 60:
            ax.set_yticks([])
        else:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ids, fontsize=8)

        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=rank_to_color[r])
            for r in unique_ranks
        ]

        labels = [f"F{r}" for r in unique_ranks]

        ax.legend(
            handles,
            labels,
            title="Global Pareto Front",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left"
        )

        plt.tight_layout()

        out_path = output_dir / f"lifetime_gen_{g}.png"

        plt.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"[OK] Lifetime plot generated for generation {g}")


def plot_last_generation_pareto_front(
    pareto_per_generation: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    output_path: Path
):
    """
    Plot ONLY the Pareto front of the last generation,
    following the EXACT visual pattern of plot_pareto_fronts().
    """

    if not pareto_per_generation:
        print("[WARN] No Pareto data available")
        return

    # ------------------------------------------------------------
    # Identify last generation
    # ------------------------------------------------------------
    last_gen = max(pareto_per_generation.keys())
    front = pareto_per_generation[last_gen]

    if not front:
        print(f"[WARN] Empty Pareto front at generation {last_gen}")
        return

    # ------------------------------------------------------------
    # Mimic structure of 'pareto_by_front'
    # Single front → index 0
    # ------------------------------------------------------------
    pareto_by_front = {0: front}

    fronts = [0]
    num_fronts = 1

    colors = plt.cm.coolwarm(
        np.linspace(0.1, 0.9, num_fronts)
    )

    # ------------------------------------------------------------
    # Figure layout (IDENTICAL)
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    pairs = [(1, 0), (1, 2), (0, 2)]

    # ------------------------------------------------------------
    # 2D projections
    # ------------------------------------------------------------
    for idx_pair, (i, j) in enumerate(pairs):

        ax = fig.add_subplot(gs[0, idx_pair])

        front_data = pareto_by_front[0]

        x = [
            p["objectives"][objective_names[i]]
            for p in front_data
        ]

        y = [
            p["objectives"][objective_names[j]]
            for p in front_data
        ]

        ax.scatter(
            x,
            y,
            color=colors[0],
            alpha=0.85
        )

        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.set_title(
            f"{objective_names[i]} vs {objective_names[j]}"
        )
        ax.grid(True)

    # ------------------------------------------------------------
    # 3D view A (XYZ)
    # ------------------------------------------------------------
    ax3d_a = fig.add_subplot(
        gs[1, 0],
        projection="3d"
    )

    xs = [
        p["objectives"][objective_names[0]]
        for p in front
    ]
    ys = [
        p["objectives"][objective_names[1]]
        for p in front
    ]
    zs = [
        p["objectives"][objective_names[2]]
        for p in front
    ]

    ax3d_a.scatter(
        xs,
        ys,
        zs,
        color=colors[0],
        alpha=0.85
    )

    ax3d_a.set_xlabel(objective_names[0])
    ax3d_a.set_ylabel(objective_names[1])
    ax3d_a.set_zlabel(objective_names[2])
    ax3d_a.set_title(
        f"Pareto Front — Last Generation (XYZ)\nG{last_gen}"
    )

    # ------------------------------------------------------------
    # 3D view B (swapped axes)
    # ------------------------------------------------------------
    ax3d_b = fig.add_subplot(
        gs[1, 1],
        projection="3d"
    )

    xs = [
        p["objectives"][objective_names[1]]
        for p in front
    ]
    ys = [
        p["objectives"][objective_names[0]]
        for p in front
    ]
    zs = [
        p["objectives"][objective_names[2]]
        for p in front
    ]

    sc = ax3d_b.scatter(
        xs,
        ys,
        zs,
        color=colors[0],
        alpha=0.85
    )

    ax3d_b.set_xlabel(objective_names[1])
    ax3d_b.set_ylabel(objective_names[0])
    ax3d_b.set_zlabel(objective_names[2])

    ax3d_b.set_title(
        f"Pareto Front — Last Generation (swapped)\nG{last_gen}"
    )

    # Legend (same pattern)
    ax3d_b.legend(
        [sc],
        [f"F0 (Gen {last_gen})"],
        loc="center left",
        bbox_to_anchor=(1.15, 0.5),
        title="Fronts"
    )

    # ------------------------------------------------------------
    # Layout & save
    # ------------------------------------------------------------
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[OK] Last generation Pareto front plotted → {output_path}"
    )


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
    parser.add_argument("--keep-the-files", default=False)

    parser.add_argument(
        "--objectives",
        nargs=3,
        default=("latency", "energy", "throughput"),
        metavar=("X", "Y", "Z")
    )

    parser.add_argument(
        "--minimize",
        nargs=3,
        default=("True", "True", "False"),
        metavar=("X", "Y", "Z")
    )
    
    args = parser.parse_args()
    
    args.minimize = [s.lower() == "true" for s in args.minimize]

    if not args.api_key:
        raise SystemExit("Missing API key")

    session = build_session(args.api_key)

    individuals_per_gen = get_individuals_per_generation_api(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid
    )

    population = build_population_from_api(individuals_per_gen)
    fronts = fast_nondominated_sort(population, args.objectives, args.minimize)
    pareto_by_front = convert_pareto_by_front(fronts)


    pareto_plot = Path(f"pareto_fronts_{args.expid}.png")
    plot_pareto_fronts(
        pareto_by_front,
        tuple(args.objectives),
        pareto_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        pareto_plot,
        "pareto_fronts",
        "Pareto fronts (dominance layers)"
    )
    print("[OK] Pareto dominance analysis completed")
    

    lifetime_dir = Path(f"lifetimes_{args.expid}")
    plot_individual_lifetime_per_generation(
        individuals_per_generation=individuals_per_gen,
        pareto_by_front=pareto_by_front,
        output_dir=lifetime_dir
    )
    for file in lifetime_dir.glob("*.png"):
        upload_analysis_file_api(
            session,
            args.api_base,
            args.expid,
            file,
            file.name.replace(".png", ""),
            file.name.replace(".png", "")        
        )
    print("[OK] Lifetime per generation completed")
    
    
    lifetime_plot = Path(f"individual_lifetime_{args.expid}.png")
    plot_individual_lifetime(
        individuals_per_generation=individuals_per_gen,
        pareto_by_front=pareto_by_front,
        output_path=lifetime_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        lifetime_plot,
        "individual_lifetime",
        "Individual survival across generations (colored by global Pareto rank)"
    )
    print("[OK] Individual lifetime completed")

        
    pareto_per_gen = get_pareto_per_generation_api(
            session=session,
            api_base=args.api_base,
            experiment_id=args.expid,
        )
        
    generations = sorted(pareto_per_gen.keys())
    if not generations:
        raise SystemExit("No Pareto fronts found")
    
    
    last_front_plot = Path(
        f"pareto_last_generation_{args.expid}.png"
    )
    plot_last_generation_pareto_front(
        pareto_per_generation=pareto_per_gen,
        objective_names=tuple(args.objectives),
        output_path=last_front_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        last_front_plot,
        "pareto_last_generation",
        "Pareto front of the last generation"
    )

    # ------------------------------------------------------------
    # Worst reference point for hypervolume
    # ------------------------------------------------------------
    worst_point = compute_worst_point(
        pareto_per_gen,
        tuple(args.objectives),
        minimize=args.minimize
    )

    # ------------------------------------------------------------
    # HV and GD computation
    # ------------------------------------------------------------
    final_front = np.array([
        [p["objectives"][o] for o in args.objectives]
        for p in pareto_by_front[0]
    ])
    
    final_front = to_minimization_array(final_front, objectives=args.objectives, minimize=args.minimize)

    hv_values = []
    gd_values = []

    for gen in generations:
        front = pareto_per_gen[gen]

        points = np.array([
            [p["objectives"][o] for o in args.objectives]
            for p in front
        ])
        
        #points = to_minimization_array(points, objectives=args.objectives, minimize=args.minimize)

        print(f"points: {points}")
        print(f"worst_point: {worst_point}")
        hv_val = hv.hypervolume(points, worst_point)

        gd_val = compute_gd(
            points,
            final_front
        )

        hv_values.append(hv_val)
        gd_values.append(gd_val)

    hv_gd_plot = Path(f"hv_gd_{args.expid}.png")

    plot_hv_gd(
        generations=generations,
        hv_values=hv_values,
        gd_values=gd_values,
        worst_point=worst_point,
        output_path=hv_gd_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        hv_gd_plot,
        "hv_gd",
        "Hypervolume and generational distance evolution 2"
    )    
    print("[OK] Pareto HV and GD analysis completed")
    # ------------------------------------------------------------


    front_gen_plot = Path(f"pareto_fronts_per_generation_{args.expid}.png")
    plot_front_per_generation_distribution(
        pareto_per_generation=pareto_per_gen,
        output_path=front_gen_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        front_gen_plot,
        "pareto_fronts_per_generation",
        "Pareto fronts distribution computed intra-generation"
    )
    print("[OK] Pareto fronts per generation completed")


    parallel_plot = Path(f"pareto_parallel_{args.expid}.png")
    plot_parallel_coordinates_pareto0(
        pareto_by_front,
        tuple(args.objectives),
        args.minimize,
        parallel_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        parallel_plot,
        "pareto_parallel",
        "Pareto front 0 — parallel coordinates"
    )
    print("[OK] Pareto parallel coordenates analysis completed")

    radar_plot = Path(f"pareto_radar_{args.expid}.png")
    plot_radar_pareto0(
        pareto_by_front,
        tuple(args.objectives),
        args.minimize,
        radar_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        radar_plot,
        "pareto_radar",
        "Pareto front 0 — radar plot"
    )
    print("[OK] Pareto radar coordenates analysis completed")


    dist_plot = Path(f"pareto_distribution_{args.expid}.png")
    plot_generation_front_distribution(
        population,
        dist_plot
    )
    upload_analysis_file_api(
        session,
        args.api_base,
        args.expid,
        dist_plot,
        "pareto_distribution",
        "Distribution of individuals per front and generation"
    )        
    print("[OK] Distribution of individuals per front and generation completed")
    
    # Cleanup
    if args.keep_the_files == False:
        try:
            pareto_plot.unlink(missing_ok=True)
            dist_plot.unlink(missing_ok=True)
            parallel_plot.unlink(missing_ok=True)
            radar_plot.unlink(missing_ok=True)
            hv_gd_plot.unlink(missing_ok=True)
            lifetime_plot.unlink(missing_ok=True)
            front_gen_plot.unlink(missing_ok=True)
            for file in lifetime_dir.glob("*.png"):
                file.unlink(missing_ok=True)
            print("[OK] Temporary files removed")
        except Exception as ex:
            print(f"[WARN] Failed to remove temporary file: {ex}")

if __name__ == "__main__":
    main()
