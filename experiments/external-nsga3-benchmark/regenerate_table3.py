"""
Regenerates Table 3 of the SimLab paper — DTLZ2 quality metrics for three
NSGA-III implementations integrated as benchmarks:

  * nsga3_func        — SimLab native implementation (mo-engine/lib/nsga/)
  * nsga3_deap_func   — DEAP's selNSGA3
  * nsga3_pymoo_func  — pymoo's NSGA3

Metrics, on the non-dominated set of each final population: Hypervolume (HV),
Generational Distance (GD), Inverted GD (IGD) and IGD+, against the
analytical DTLZ2 front. Each method is compared with the native one by the
two-sided Mann–Whitney U test on the per-run IGD, with the Â12 effect size.

This is the standalone counterpart of the SimLab platform's evaluation —
no MongoDB, no master-node, just three algorithms on the same problems
with shared metrics so the comparison is apples-to-apples.

Usage:
    python regenerate_table3.py [--seeds 5] [--generations 400] \
                                [--pop-size N] [--divisions P] \
                                [--out results/table3.csv]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import (
    a12_lower,
    dtlz2_reference_front,
    generational_distance,
    hypervolume,
    igd_plus,
    inverted_generational_distance,
    mann_whitney_pvalue,
    non_dominated,
    summary,
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

#: Das & Dennis divisions per number of objectives. M = 3 and M = 5 are the
#: settings of Deb & Jain (2014) for DTLZ; the others are chosen the same way,
#: for a number of directions H of the same order. A single setting (12
#: divisions, 92 individuals) gave 455 to 6188 directions for M >= 4, more than
#: the population, and NSGA-III's niching then has a direction per individual
#: at best.
DIVISIONS: dict[int, int] = {2: 99, 3: 12, 4: 7, 5: 6, 6: 4}


def reference_directions(M: int, divisions: int) -> int:
    """H, the number of Das & Dennis directions: C(p + M − 1, M − 1)."""
    return math.comb(divisions + M - 1, M - 1)


def population_for(M: int, divisions: int) -> int:
    """Deb & Jain's rule: the smallest multiple of four above H."""
    return (reference_directions(M, divisions) // 4 + 1) * 4


#: HV reference point: 1.1 × the DTLZ2 nadir (1, …, 1), the same for every
#: method and seed, as the platform's /hv-gd. One derived from each run's own
#: worst point made HV incomparable between the rows of the table.
HV_REFERENCE = 1.1


def evaluate_front(population: np.ndarray, M: int) -> dict[str, float]:
    """Indicators of the non-dominated set of a final population."""
    front = non_dominated(population)
    ref = dtlz2_reference_front(M, n_points=2000)
    return {
        "hv": hypervolume(front, np.full(M, HV_REFERENCE)),
        "gd": generational_distance(front, ref),
        "igd": inverted_generational_distance(front, ref),
        "igd_plus": igd_plus(front, ref),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of independent seeds per (method, M, k) cell (default: 5)")
    parser.add_argument("--generations", type=int, default=400)
    parser.add_argument("--pop-size", type=int, default=None,
                        help="One population size for every M (default: Deb & Jain's rule per M)")
    parser.add_argument("--divisions", type=int, default=None,
                        help="One number of divisions for every M (default: DIVISIONS per M)")
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

    # Each method is compared with the SimLab native one.
    BASELINE = "nsga3_func"
    KEYS = ("hv", "gd", "igd", "igd_plus")

    rows: list[dict] = []
    for M, k in TABLE3_GRID:
        divisions = args.divisions if args.divisions is not None else DIVISIONS[M]
        pop_size = args.pop_size if args.pop_size is not None else population_for(M, divisions)
        cell_rows: list[dict] = []
        cell_igd: dict[str, list[float]] = {}   # per-run IGD for the comparison
        for method, runner in runners.items():
            per_seed: dict[str, list[float]] = {key: [] for key in KEYS}
            t0 = time.time()
            for seed_offset in range(args.seeds):
                front = runner(
                    M=M, k=k,
                    pop_size=pop_size,
                    n_generations=args.generations,
                    divisions=divisions,
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
                "divisions": divisions,
                "directions": reference_directions(M, divisions),
                "pop_size": pop_size,
                "mean_hv": round(s["hv"]["mean"], 3), "std_hv": round(s["hv"]["std"], 3),
                "mean_gd": round(s["gd"]["mean"], 4), "std_gd": round(s["gd"]["std"], 4),
                "mean_igd": round(s["igd"]["mean"], 4), "std_igd": round(s["igd"]["std"], 4),
                "mean_igd_plus": round(s["igd_plus"]["mean"], 4),
                "std_igd_plus": round(s["igd_plus"]["std"], 4),
                "elapsed_s": round(elapsed, 1),
                "n_seeds": args.seeds,
            })
            print(f"({M},{k}) {method:18s} "
                  f"HV={s['hv']['mean']:.3f}±{s['hv']['std']:.3f}  "
                  f"GD={s['gd']['mean']:.4f}  "
                  f"IGD={s['igd']['mean']:.4f}±{s['igd']['std']:.4f}  "
                  f"IGD+={s['igd_plus']['mean']:.4f}  ({elapsed:.1f}s)")

        # Unpaired: the runs of two methods share nothing but a seed number.
        for row in cell_rows:
            if BASELINE in cell_igd and row["method"] != BASELINE:
                p = mann_whitney_pvalue(cell_igd[row["method"]], cell_igd[BASELINE])
                a12 = a12_lower(cell_igd[row["method"]], cell_igd[BASELINE])
                row["igd_p_vs_native"] = round(p, 4) if p == p else float("nan")
                row["igd_a12_vs_native"] = round(a12, 3) if a12 == a12 else float("nan")
            else:
                row["igd_p_vs_native"] = float("nan")
                row["igd_a12_vs_native"] = float("nan")
        rows.extend(cell_rows)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n[regenerate] wrote {out_path}")


if __name__ == "__main__":
    main()
