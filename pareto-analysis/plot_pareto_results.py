"""
Pareto analysis for SimLab experiments.

Produces two figures, both consistent with what the SimLab web GUI shows:

  1. Pareto fronts (dominance layers) — every front of the global
     non-dominated sort, each with its own colour and marker size.
  2. Hypervolume (per generation and cumulative) and Generational Distance.

Consistency with the GUI
------------------------
The GUI does not compute these metrics in the browser: it reads them from
``GET /experiments/{id}/hv-gd`` (rest-api/api/endpoints/experiment.py) and it
colours points by ``individualRankMap`` (gui/.../pages/ExperimentDetail.vue).
This module mirrors both definitions exactly:

* penalized individuals (any |objective| >= 1e8) are dropped everywhere;
* the HV reference point is the per-axis worst value **in minimization space**
  (maximized objectives negated first), plus a ``5% + 1.0`` margin;
* HV of a generation is the hypervolume of that generation's own Pareto front,
  counting only points that strictly dominate the reference point;
* cumulative HV folds each generation's front into a running non-dominated set
  ("best so far"), and is therefore monotonically non-decreasing;
* GD is the *mean* nearest-neighbour distance from the generation's front to
  the experiment's stored Pareto front, in minimization space;
* front ranks come from a global non-dominated sort over the unique objective
  vectors of every non-penalized individual of every generation.
"""
import os
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import moocore

from lib.api import (
    build_session,
    get_generations_from_experiment,
    get_experiment_pareto_front,
    upload_analysis_file_api,
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
        for all i: a_i <= b_i   (min)   or   a_i >= b_i (max)
        exists  j: a_j <  b_j   (min)   or   a_j >  b_j (max)

    Matches ``_dominates`` in the rest-api hv-gd endpoint and ``dominates``
    in the GUI's nonDominatedSort.ts.
    """

    if len(objectives) != len(minimize):
        raise ValueError("`objectives` and `minimize` must have same length")

    strictly_better = False

    for obj, is_min in zip(objectives, minimize):
        va = a[obj]
        vb = b[obj]

        if is_min:
            if va > vb:
                return False
            if va < vb:
                strictly_better = True
        else:
            if va < vb:
                return False
            if va > vb:
                strictly_better = True

    return strictly_better


# ------------------------------------------------------------
# Fast non-dominated sorting
# ------------------------------------------------------------
def fast_nondominated_sort(
    population: list[dict[str, Any]],
    objectives: list[str] = ["latency", "energy", "throughput"],
    minimize: list[bool] = [True, True, False]
) -> list[list[dict[str, Any]]]:
    """
    NSGA-II fast non-dominated sort.

    Returns the fronts as lists of the input dicts, and stamps a ``rank`` key
    (0 = non-dominated) on every individual.  Points with identical objective
    vectors never dominate each other, so duplicates share a front — the same
    convention used by the GUI.
    """
    S: dict[Any, list[dict[str, Any]]] = {}
    n: dict[Any, int] = {}
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

    for p in population:
        if "rank" not in p:
            p["rank"] = i

    return fronts[:-1]


# ------------------------------------------------------------
# Minimization transform for hypervolume / GD
# ------------------------------------------------------------
def to_minimization_array(
    points: np.ndarray,
    objectives: list[str],
    minimize: list[bool],
) -> np.ndarray:
    """
    Convert an objective matrix to an equivalent minimization space.

    Parameters
    ----------
    points : np.ndarray
        Objective matrix of shape (N, M): N solutions, M objectives.
    objectives : list[str]
        Objective names (metadata / ordering reference).
    minimize : list[bool]
        True  -> objective already minimized
        False -> objective is maximized (will be negated)

    Returns
    -------
    np.ndarray
        Matrix where every objective follows 'smaller is better' semantics.

    For each maximization objective j:  f'_j(x) = -f_j(x), which preserves
    Pareto dominance relations.
    """

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
        raise ValueError("`objectives` and `minimize` must have same length")

    out = points.copy()

    for j, is_min in enumerate(minimize):
        if not is_min:
            out[:, j] *= -1.0

    return out


# ------------------------------------------------------------
# Worst point computation for the hypervolume reference
# ------------------------------------------------------------
def compute_worst_point(
    pareto_per_gen: dict[int, list[dict]],
    objective_names: tuple[str, ...],
    minimize: list[bool]
) -> list[float]:
    """
    Per-axis worst (maximum) value over every individual, in minimization space.

    Maximized objectives are negated *before* taking the maximum: computing the
    worst value in raw space would place the reference on the wrong side of a
    maximized axis, inflating HV by a large constant and masking its growth.
    This mirrors the ``all_min`` / ``worst`` block of the hv-gd endpoint.

    Raises
    ------
    ValueError
        If there is no individual to derive the reference from.
    """
    all_points = []

    for fronts in pareto_per_gen.values():
        for p in fronts:
            all_points.append([p["objectives"][o] for o in objective_names])

    if not all_points:
        raise ValueError("cannot compute a worst point: no individuals available")

    all_points = to_minimization_array(
        np.array(all_points, dtype=float), objectives=list(objective_names), minimize=minimize
    )

    return all_points.max(axis=0).tolist()


# ------------------------------------------------------------
# Generational Distance
# ------------------------------------------------------------
def compute_gd(front: np.ndarray, ref_front: np.ndarray) -> float:
    """Generational Distance between ``front`` and ``ref_front``.

    For each point p in ``front``, let d(p) be its Euclidean distance to the
    nearest point of ``ref_front``.  This returns the arithmetic mean:

        GD = (1/N) * sum_p d(p)

    This is the p=1 formulation, and it is the one the SimLab API serves to the
    GUI (``dist.min(axis=1).mean()`` in the hv-gd endpoint), so the numbers the
    tool prints match the numbers the GUI shows.  Both inputs must be in the
    same minimization objective space.  Returns +inf if either set is empty.
    """
    if len(front) == 0 or len(ref_front) == 0:
        return float("inf")

    dist = np.sqrt(((np.asarray(front, dtype=float)[:, None, :]
                     - np.asarray(ref_front, dtype=float)[None, :, :]) ** 2).sum(axis=2))

    return float(dist.min(axis=1).mean())


# ------------------------------------------------------------
# Non-dominated filter in minimization space
# ------------------------------------------------------------
def nondominated_rows_min(rows: list[list[float]]) -> list[list[float]]:
    """
    Keep only the non-dominated rows of a minimization-space point set.

    Mirrors ``_pareto_front(..., [True]*n_obj)`` in the hv-gd endpoint.
    """
    n = len(rows)
    if n <= 1:
        return [list(r) for r in rows]

    arr = np.asarray(rows, dtype=float)
    keep: list[list[float]] = []

    for i in range(n):
        # j dominates i  <=>  all(arr[j] <= arr[i]) and any(arr[j] < arr[i])
        le = np.all(arr <= arr[i], axis=1)
        lt = np.any(arr < arr[i], axis=1)
        dominated = le & lt
        dominated[i] = False
        if not dominated.any():
            keep.append(list(arr[i]))

    return keep


def generation_front_min(
    individuals: list[dict],
    objectives: list[str],
    minimize: list[bool],
) -> list[list[float]]:
    """
    Pareto front of one generation, deduplicated, in minimization space.

    Equivalent to the endpoint's ``_pareto_front`` + dedup block: the front is
    computed in the original space with the real orientations, then mapped to
    minimization space and deduplicated by exact objective tuple.
    """
    if not individuals:
        return []

    raw = np.array(
        [[ind["objectives"][o] for o in objectives] for ind in individuals],
        dtype=float,
    )
    pts_min = to_minimization_array(raw, objectives=objectives, minimize=minimize)

    # Dominance in minimization space is identical to dominance in the original
    # space with the real orientations, so a single filter suffices here.
    front_rows = nondominated_rows_min(pts_min.tolist())

    seen: set[tuple] = set()
    out: list[list[float]] = []
    for row in front_rows:
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            out.append(row)

    return out


# ------------------------------------------------------------
# Convergence metrics: HV, cumulative HV and GD
# ------------------------------------------------------------
def compute_convergence_metrics(
    individuals_per_gen: dict[int, list[dict]],
    objectives: list[str],
    minimize: list[bool],
    hv_ref: list[float],
    reference_front_min: np.ndarray,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """
    Compute per-generation HV, cumulative HV and GD.

    Returns
    -------
    (generations, hv, hv_cumulative, gd)
        ``generations`` is sorted ascending; the three metric lists are aligned
        with it index by index.  ``gd`` uses NaN for generations with no
        feasible individual so the curve shows a gap instead of a fake zero.

    The cumulative curve folds each generation's front into a running
    non-dominated set rather than re-sorting the whole population: a point that
    is off its own generation's front is dominated within that generation too,
    so it can never join the accumulated front.  Cost is O(G * front^2).
    """
    generations = sorted(individuals_per_gen.keys())

    hv_values: list[float] = []
    hv_cumulative: list[float] = []
    gd_values: list[float] = []

    hv_ref_arr = np.asarray(hv_ref, dtype=float)
    acc_seen: set[tuple] = set()        # dedup keys of the running front
    acc_rows: list[list[float]] = []    # running non-dominated set (min-space)
    last_cum_hv = 0.0

    for gen in generations:
        front_rows = generation_front_min(
            individuals_per_gen[gen], objectives=objectives, minimize=minimize
        )

        if not front_rows:
            hv_values.append(0.0)
            hv_cumulative.append(last_cum_hv)   # an empty generation adds nothing
            gd_values.append(float("nan"))
            continue

        pts_min = np.asarray(front_rows, dtype=float)

        # Only points that strictly dominate the reference contribute to HV;
        # moocore requires every point passed to it to dominate ``ref``.
        dominating = pts_min[np.all(pts_min < hv_ref_arr, axis=1)]
        hv_val = float(moocore.hypervolume(dominating, ref=hv_ref)) if len(dominating) else 0.0

        # Cumulative ("best so far") front.  Duplicates never dominate each
        # other, so without the dedup key the accumulator would keep one copy
        # per generation a point reappears in — same hypervolume, unbounded
        # growth.  Deduplicating keeps the running set minimal.
        for row in front_rows:
            key = tuple(row)
            if key not in acc_seen:
                acc_seen.add(key)
                acc_rows.append(row)
        acc_rows = nondominated_rows_min(acc_rows)
        acc_seen = {tuple(r) for r in acc_rows}
        acc_arr = np.asarray(acc_rows, dtype=float)
        acc_dom = acc_arr[np.all(acc_arr < hv_ref_arr, axis=1)]
        cum_hv = float(moocore.hypervolume(acc_dom, ref=hv_ref)) if len(acc_dom) else 0.0
        last_cum_hv = cum_hv

        hv_values.append(hv_val)
        hv_cumulative.append(cum_hv)
        gd_values.append(compute_gd(pts_min, reference_front_min))

    return generations, hv_values, hv_cumulative, gd_values


# ------------------------------------------------------------
# Plot: Pareto fronts (dominance layers)
# ------------------------------------------------------------
# Colours and marker sizes are taken verbatim from the web GUI's 3D Pareto
# chart (gui/.../components/charts/ParetoFront3DChart.vue), so a front is the
# same colour here and there.  The GUI labels ranks 1-indexed and merges
# everything from rank 5 down into a single grey "Other" group; both
# conventions are reproduced.
_GUI_RANK_PALETTE = (
    "#3b82f6",   # Front 1 — blue
    "#10b981",   # Front 2 — emerald
    "#f59e0b",   # Front 3 — amber
    "#8b5cf6",   # Front 4 — violet
    "#f43f5e",   # Front 5 — rose
    "#94a3b8",   # Other   — gray
)
# GUI RANK_SIZES_3D: ECharts symbolSize, i.e. marker DIAMETERS in px on a chart
# a few hundred px wide.  matplotlib's scatter ``s`` is an AREA in points² on a
# 300-dpi figure, so the diameters are rescaled to this canvas and squared —
# the 6:5:5:4:4:3 proportions between fronts are preserved exactly.
_GUI_RANK_SIZES = (6, 5, 5, 4, 4, 3)
_MARKER_DIAMETER_SCALE = 1.47
# GUI opacity: 0.9 for a labelled front, 0.4 for the "Other" bucket.
_GUI_RANK_ALPHA = (0.9, 0.9, 0.9, 0.9, 0.9, 0.4)
# Ranks at or beyond this index collapse into "Other", as in the GUI.
_MAX_LABELED_RANKS = 5


def _gui_rank_groups(
    pareto_by_front: dict[int, list[dict]],
) -> list[tuple[int, str, list[dict]]]:
    """
    Bucket dominance layers the way the GUI does.

    Ranks 0..4 stay separate and keep the GUI's 1-indexed labels ("Front 1" is
    the non-dominated set); every deeper rank is merged into one "Other" group.
    No individual is dropped — deep layers are grouped, not discarded.

    Returns a list of ``(bucket_index, label, points)``, ordered best-first.
    """
    groups: list[tuple[int, str, list[dict]]] = []

    for rank in range(_MAX_LABELED_RANKS):
        points = pareto_by_front.get(rank, [])
        if points:
            groups.append((rank, f"Front {rank + 1}", points))

    other = [
        point
        for rank, points in sorted(pareto_by_front.items())
        if rank >= _MAX_LABELED_RANKS
        for point in points
    ]
    if other:
        groups.append((_MAX_LABELED_RANKS, "Other", other))

    return groups


def plot_pareto_fronts(
    pareto_by_front: dict[int, list[dict]],
    objective_names: tuple[str, ...],
    output_path: Path,
):
    """
    Scatter every dominance layer of the global non-dominated sort.

    Colour and marker size per front come from the web GUI's palette, so the
    figure and the GUI agree on what each front looks like.  Fronts are drawn
    worst-first so that front 1 (the non-dominated set) is never hidden
    underneath a deeper layer.

    ``pareto_by_front`` maps rank -> individuals; ``objective_names`` must hold
    exactly three objectives (three 2D projections plus one 3D view).
    """
    if len(objective_names) != 3:
        raise ValueError(f"expected exactly 3 objectives, got {len(objective_names)}")

    groups = _gui_rank_groups(pareto_by_front)
    if not groups:
        raise ValueError("no non-empty front to plot")

    def _style(bucket: int) -> tuple[str, float, float]:
        color = _GUI_RANK_PALETTE[bucket]
        size = (_GUI_RANK_SIZES[bucket] * _MARKER_DIAMETER_SCALE) ** 2
        return color, size, _GUI_RANK_ALPHA[bucket]

    def _values(points: list[dict], axis: int) -> list[float]:
        return [p["objectives"][objective_names[axis]] for p in points]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    # Draw worst-first so the non-dominated set ends up on top everywhere.
    draw_order = list(range(len(groups)))[::-1]

    # ---------- 2D projections: all three axis pairs ----------
    pairs = [(0, 1), (0, 2), (1, 2)]

    for idx_pair, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, idx_pair])

        for g in draw_order:
            bucket, _, points = groups[g]
            color, size, alpha = _style(bucket)
            ax.scatter(
                _values(points, i),
                _values(points, j),
                color=color,
                s=size,
                alpha=alpha,
                edgecolors="none",
                zorder=len(groups) - g,
            )

        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.set_title(f"{objective_names[i]} vs {objective_names[j]}")
        ax.grid(True, alpha=0.3)

    # ---------- 3D view ----------
    ax3d = fig.add_subplot(gs[1, 0:2], projection="3d")

    for g in draw_order:
        bucket, _, points = groups[g]
        color, size, alpha = _style(bucket)
        ax3d.scatter(
            _values(points, 0),
            _values(points, 1),
            _values(points, 2),
            color=color,
            s=size,
            alpha=alpha,
            edgecolors="none",
            depthshade=False,
        )

    ax3d.set_xlabel(objective_names[0])
    ax3d.set_ylabel(objective_names[1])
    ax3d.set_zlabel(objective_names[2])
    ax3d.set_title("Pareto fronts (3D)")

    # ---------- Legend ----------
    ax_legend = fig.add_subplot(gs[1, 2])
    ax_legend.axis("off")

    total_points = sum(len(points) for _, _, points in groups)
    n_fronts = len(pareto_by_front)

    handles = []
    for bucket, label, points in groups:
        color, size, alpha = _style(bucket)
        handles.append(
            Line2D(
                [], [], linestyle="none", marker="o",
                markerfacecolor=color, markeredgecolor="none", alpha=alpha,
                # Line2D sizes are diameters in points, matching the diameter
                # the scatter above was built from.
                markersize=_GUI_RANK_SIZES[bucket] * _MARKER_DIAMETER_SCALE,
                label=f"{label}  (n = {len(points)})",
            )
        )

    ax_legend.legend(
        handles=handles,
        loc="center",
        title=f"Dominance layers\n{n_fronts} fronts · {total_points} unique solutions",
        frameon=True,
        labelspacing=1.1,
    )

    fig.suptitle("Pareto fronts (dominance layers)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------
# Plot: hypervolume (per generation + cumulative) and GD
# ------------------------------------------------------------
def plot_hv_gd(
    generations: list[int],
    hv_values: list[float],
    hv_cumulative: list[float],
    gd_values: list[float],
    worst_point: list[float],
    objective_names: tuple[str, ...],
    output_path: Path,
):
    """
    Left panel: hypervolume per generation and its cumulative (best-so-far)
    envelope, the same two series the GUI toggles between.  Both share the
    reference point printed in the title, so they are directly comparable and
    the cumulative curve always sits on or above the per-generation one.

    Right panel: generational distance to the experiment's stored Pareto front.
    """
    fig, (ax_hv, ax_gd) = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

    ref_txt = ", ".join(
        f"{name}={value:.3g}" for name, value in zip(objective_names, worst_point)
    )

    # ---------- Hypervolume ----------
    ax_hv.plot(
        generations, hv_values,
        marker="o", markersize=4, linewidth=1.5,
        color="tab:blue", label="Per generation",
    )
    ax_hv.plot(
        generations, hv_cumulative,
        marker="^", markersize=4, linewidth=1.8, linestyle="--",
        color="tab:green", label="Cumulative (best so far)",
    )
    ax_hv.fill_between(generations, hv_values, hv_cumulative, color="tab:green", alpha=0.08)
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title(f"Hypervolume evolution\nreference point (min-space): {ref_txt}")
    ax_hv.legend(loc="lower right", fontsize=9)
    ax_hv.grid(True, alpha=0.3)

    # ---------- Generational distance ----------
    ax_gd.plot(
        generations, gd_values,
        marker="s", markersize=4, linewidth=1.5, color="tab:red",
    )
    ax_gd.set_ylabel("Generational distance")
    ax_gd.set_title("Generational distance to the stored Pareto front")
    ax_gd.grid(True, alpha=0.3)

    # A long run would otherwise stamp one tick per generation and smear them
    # into an unreadable band.
    step = max(1, len(generations) // 15)
    ticks = generations[::step]

    for ax in (ax_hv, ax_gd):
        ax.set_xlabel("Generation")
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(g) for g in ticks])

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # ------------------------------------------------------------
    # ARGS
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Pareto front dominance analysis (API-based)"
    )

    parser.add_argument("--api-base", default="http://localhost:8000/api/v1")
    parser.add_argument("--api-key", default=os.getenv("SIMLAB_API_KEY", "api-password"))
    parser.add_argument("--expid", required=True)
    parser.add_argument("--keep-the-files", action="store_true", default=False)

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

    parser.add_argument(
        "--true-front-bench",
        choices=["DTLZ2", "ZDT1", "SCH1"],
        default=None,
        help=(
            "For synthetic experiments: measure HV and GD against the "
            "benchmark's analytical Pareto front (convergence to the true "
            "optimum) instead of the experiment's own empirical references."
        ),
    )
    parser.add_argument(
        "--true-front-m",
        type=int,
        default=None,
        help="Number of objectives for the analytical front (default: len(objectives)).",
    )

    args = parser.parse_args()

    args.objectives = list(args.objectives)
    args.minimize = [s.lower() == "true" for s in args.minimize]

    # ------------------------------------------------------------
    # INITIAL DATA FETCHING
    # ------------------------------------------------------------
    session = build_session(args.api_key)

    # Non-penalized individuals per generation (penalized filtered in lib/api.py)
    individuals_per_gen = get_generations_from_experiment(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
        label_objectives=args.objectives,
    )

    if not any(individuals_per_gen.values()):
        raise SystemExit(
            "[ERROR] No feasible individual found for this experiment — "
            "every individual is penalized or the experiment has no generations."
        )

    # Authoritative Pareto front stored by the engine — exactly what the GUI shows.
    # Items: {"objectives": {metric: value, ...}, "chromosome": {...}}
    stored_pareto = get_experiment_pareto_front(
        session=session,
        api_base=args.api_base,
        experiment_id=args.expid,
    )

    # ------------------------------------------------------------
    # GLOBAL DOMINANCE LAYERS — deduplicated by exact objectives
    # ------------------------------------------------------------
    # The same chromosome can appear in several generations under different
    # MongoDB _ids.  Keeping every copy would inflate the non-dominated front,
    # because duplicates never dominate each other.  Deduplicating here makes
    # the ranks match the GUI's individualRankMap, which sorts unique objective
    # vectors of all non-penalized individuals.
    seen_obj_keys: set[tuple] = set()
    dedup_population: list[dict] = []
    for gen in sorted(individuals_per_gen.keys()):
        for ind in individuals_per_gen[gen]:
            key = tuple(ind["objectives"][o] for o in args.objectives)
            if key not in seen_obj_keys:
                seen_obj_keys.add(key)
                dedup_population.append({
                    "id": ind["id"],
                    "generation": gen,
                    "objectives": ind["objectives"],
                })

    dedup_fronts = fast_nondominated_sort(dedup_population, args.objectives, args.minimize)
    pareto_by_front: dict[int, list[dict]] = {
        rank: front for rank, front in enumerate(dedup_fronts) if front
    }

    # ------------------------------------------------------------
    # Pareto fronts (dominance layers)
    # ------------------------------------------------------------
    pareto_plot = Path(f"pareto_fronts_{args.expid}.png")
    plot_pareto_fronts(
        pareto_by_front,
        tuple(args.objectives),
        pareto_plot,
    )
    upload_analysis_file_api(
        session, args.api_base, args.expid, pareto_plot,
        "pareto_fronts", "Pareto fronts (dominance layers)",
    )
    print(
        f"[OK] Pareto dominance analysis completed "
        f"({len(pareto_by_front)} fronts, {len(dedup_population)} unique solutions)"
    )

    # ------------------------------------------------------------
    # HV (per generation + cumulative) and GD
    # ------------------------------------------------------------
    # Synthetic experiments have a closed-form true Pareto front: use the
    # benchmark's analytical front as the GD reference (convergence to the real
    # optimum, not to the run's own final front) and its fixed nadir as the HV
    # reference (so HV is comparable across runs). WSN experiments keep the
    # empirical references (population-derived worst point + stored final front).
    if args.true_front_bench:
        from lib.true_fronts import sample_true_front, true_nadir
        m = args.true_front_m or len(args.objectives)
        true_front = sample_true_front(args.true_front_bench, m)
        reference_front_min = to_minimization_array(
            true_front, objectives=args.objectives, minimize=args.minimize
        )
        worst_point = [v * 1.1 for v in true_nadir(args.true_front_bench, m)]
    else:
        # Reference point: worst feasible values + margin (no penalty contamination).
        worst_point = compute_worst_point(
            individuals_per_gen,
            tuple(args.objectives),
            minimize=args.minimize,
        )
        worst_point = [coord + abs(coord) * 0.05 + 1.0 for coord in worst_point]

        if not stored_pareto:
            raise SystemExit(
                "[ERROR] The experiment has no stored pareto_front, which is the "
                "GD reference. Re-run the engine for this experiment, or pass "
                "--true-front-bench for a synthetic benchmark."
            )

        # GD reference: unique objectives from the stored (authoritative) front.
        stored_obj_matrix = np.array([
            [p["objectives"][o] for o in args.objectives]
            for p in stored_pareto
        ], dtype=float)
        stored_obj_unique = np.unique(stored_obj_matrix, axis=0)
        reference_front_min = to_minimization_array(
            stored_obj_unique, objectives=args.objectives, minimize=args.minimize
        )

    generations, hv_values, hv_cumulative, gd_values = compute_convergence_metrics(
        individuals_per_gen=individuals_per_gen,
        objectives=args.objectives,
        minimize=args.minimize,
        hv_ref=worst_point,
        reference_front_min=reference_front_min,
    )

    hv_gd_plot = Path(f"hv_gd_{args.expid}.png")
    plot_hv_gd(
        generations=generations,
        hv_values=hv_values,
        hv_cumulative=hv_cumulative,
        gd_values=gd_values,
        worst_point=worst_point,
        objective_names=tuple(args.objectives),
        output_path=hv_gd_plot,
    )
    upload_analysis_file_api(
        session, args.api_base, args.expid, hv_gd_plot,
        "hv_gd", "Hypervolume (per generation and cumulative) and generational distance",
    )
    print("[OK] Pareto HV and GD analysis completed")

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    if not args.keep_the_files:
        try:
            pareto_plot.unlink(missing_ok=True)
            hv_gd_plot.unlink(missing_ok=True)
            print("[OK] Temporary files removed")
        except OSError as ex:
            print(f"[WARN] Failed to remove temporary file: {ex}")


if __name__ == "__main__":
    main()
