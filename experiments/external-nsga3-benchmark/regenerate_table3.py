"""
Regenerates Table 3 of the SimLab paper — DTLZ2 quality metrics for three
NSGA-III implementations integrated as benchmarks:

  * nsga3_func        — SimLab native implementation (mo-engine/lib/nsga/)
  * nsga3_deap_func   — DEAP's selNSGA3
  * nsga3_pymoo_func  — pymoo's NSGA3

Metrics: Hypervolume (HV), Generational Distance (GD), Inverted GD (IGD),
Coverage C(A, B) where A is the implementation's Pareto front and B is
the reference front sampled from the analytical DTLZ2 surface.

This is the standalone counterpart of the SimLab platform's evaluation —
no MongoDB, no master-node, just three algorithms on the same problems
with shared metrics so the comparison is apples-to-apples.

Usage:
    python regenerate_table3.py [--seeds 5] [--pop-size 92] [--generations 400] \
                                [--out results/table3.csv]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import (
    coverage,
    dtlz2_reference_front,
    generational_distance,
    hypervolume,
    inverted_generational_distance,
    summary,
    wilcoxon_pvalue,
)
from run_native import run_native

# DEAP / pymoo are imported lazily so a partial install (e.g. native only)
# still produces a useful report.
def _safe_import_deap():
    try:
        from run_deap import run_deap  # noqa: F401
        return run_deap
    except Exception as e:  # pragma: no cover
        print(f"[warn] DEAP runner unavailable: {e}", file=sys.stderr)
        return None


def _safe_import_pymoo():
    try:
        from run_pymoo import run_pymoo  # noqa: F401
        return run_pymoo
    except Exception as e:  # pragma: no cover
        print(f"[warn] pymoo runner unavailable: {e}", file=sys.stderr)
        return None


# (M, k) grid from Table 3 of the SimLab paper.
TABLE3_GRID: list[tuple[int, int]] = [
    (2, 1),
    (3, 2),
    (4, 3),
    (4, 4),
    (5, 4),
    (6, 5),
    (6, 14),
]


def evaluate_front(front: np.ndarray, M: int) -> dict[str, float]:
    ref = dtlz2_reference_front(M, n_points=2000)
    # Reference point for HV: slightly beyond the worst observed value per dim
    worst = front.max(axis=0) + 0.1
    return {
        "hv": hypervolume(front, worst),
        "gd": generational_distance(front, ref),
        "igd": inverted_generational_distance(front, ref),
        "coverage": coverage(front, ref),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of independent seeds per (method, M, k) cell (default: 5)")
    parser.add_argument("--pop-size", type=int, default=92)
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--divisions", type=int, default=12)
    parser.add_argument("--methods", nargs="+",
                        default=["native", "deap", "pymoo"],
                        choices=["native", "deap", "pymoo"])
    parser.add_argument("--out", default=str(HERE / "results" / "table3.csv"))
    args = parser.parse_args()

    runners: dict[str, callable] = {}
    if "native" in args.methods:
        runners["nsga3_func"] = run_native
    if "deap" in args.methods:
        deap_runner = _safe_import_deap()
        if deap_runner is not None:
            runners["nsga3_deap_func"] = deap_runner
    if "pymoo" in args.methods:
        pymoo_runner = _safe_import_pymoo()
        if pymoo_runner is not None:
            runners["nsga3_pymoo_func"] = pymoo_runner

    if not runners:
        print("No runners available — install deap / pymoo or use --methods native", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Statistical significance is reported against the SimLab native method.
    BASELINE = "nsga3_func"
    KEYS = ("hv", "gd", "igd", "coverage")

    rows: list[dict] = []
    for M, k in TABLE3_GRID:
        cell_rows: list[dict] = []
        cell_igd: dict[str, list[float]] = {}   # per-seed IGD for the paired test
        for method, runner in runners.items():
            per_seed: dict[str, list[float]] = {key: [] for key in KEYS}
            t0 = time.time()
            for seed_offset in range(args.seeds):
                front = runner(
                    M=M, k=k,
                    pop_size=args.pop_size,
                    n_generations=args.generations,
                    divisions=args.divisions,
                    seed=42 + seed_offset,
                )
                m = evaluate_front(np.asarray(front), M)
                for key in KEYS:
                    per_seed[key].append(m[key])
            elapsed = time.time() - t0
            cell_igd[method] = per_seed["igd"]
            s = {key: summary(per_seed[key]) for key in KEYS}
            cell_rows.append({
                "M": M, "k": k, "method": method,
                "mean_hv": round(s["hv"]["mean"], 3), "std_hv": round(s["hv"]["std"], 3),
                "mean_gd": round(s["gd"]["mean"], 4), "std_gd": round(s["gd"]["std"], 4),
                "mean_igd": round(s["igd"]["mean"], 4), "std_igd": round(s["igd"]["std"], 4),
                "mean_coverage": round(s["coverage"]["mean"], 3),
                "std_coverage": round(s["coverage"]["std"], 3),
                "elapsed_s": round(elapsed, 1),
                "n_seeds": args.seeds,
            })
            print(f"({M},{k}) {method:18s} "
                  f"HV={s['hv']['mean']:.3f}±{s['hv']['std']:.3f}  "
                  f"GD={s['gd']['mean']:.4f}  "
                  f"IGD={s['igd']['mean']:.4f}±{s['igd']['std']:.4f}  "
                  f"Cov={s['coverage']['mean']:.3f}  ({elapsed:.1f}s)")

        # Paired Wilcoxon (per-seed IGD) of each method vs the native baseline.
        for row in cell_rows:
            if BASELINE in cell_igd and row["method"] != BASELINE:
                p = wilcoxon_pvalue(cell_igd[row["method"]], cell_igd[BASELINE])
                row["igd_p_vs_native"] = round(p, 4) if p == p else float("nan")
            else:
                row["igd_p_vs_native"] = float("nan")
        rows.extend(cell_rows)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[regenerate] wrote {out_path}")


if __name__ == "__main__":
    main()
