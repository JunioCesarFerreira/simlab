# External NSGA-III Benchmark

This directory contains the **standalone scripts** meant to regenerate **Table 3**
of the SimLab paper — the DTLZ2 quality comparison between three NSGA-III
implementations:

| Method label in Table 3 | Source |
|---|---|
| `nsga3_func`        | SimLab native: [mo-engine/lib/nsga/](../../mo-engine/lib/nsga/) |
| `nsga3_deap_func`   | [DEAP `selNSGA3`](https://deap.readthedocs.io/) |
| `nsga3_pymoo_func`  | [pymoo `NSGA3`](https://pymoo.org/algorithms/moo/nsga3.html) |

The scripts here are **independent of the SimLab runtime** — there is no
MongoDB, no master-node, no Cooja. They evaluate NSGA-III on the analytical
DTLZ2 benchmark on a clean Python environment.

> **The single command is currently broken**: `regenerate_table3.py` does not
> import with the pinned DEAP. The three runners work on their own. See
> [Known issues](#known-issues).

## Why "external"?

DEAP and pymoo are runtime dependencies of the mo-engine, for the strategies
`nsga3_deap` and `nsga3_pymoo`, which replace only the environmental selection
([strategies](../../mo-engine/lib/strategy/README.md)). Here each library runs
its own algorithm, independently of the platform.

Keeping the benchmark separate also makes its purpose explicit: this is
**reproduction infrastructure for an academic table**, not part of the
deployed system.

## Quick start

```bash
cd experiments/external-nsga3-benchmark

# 1. Install dependencies into a dedicated venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Smoke-test each runner
python run_native.py
python run_deap.py
python run_pymoo.py

# 3. Regenerate the full Table 3 (fails today, see Known issues)
python regenerate_table3.py --seeds 5 --pop-size 92 --generations 400
# Output: results/table3.csv
```

## Files

| File | Role |
|---|---|
| `dtlz2.py`               | Shared DTLZ2 implementation (equivalent to `pylib/benchmarks.py`) |
| `metrics.py`             | HV / GD / IGD / Coverage; analytical reference front sampler |
| `run_native.py`          | SimLab native NSGA-III runner (re-uses `mo-engine/lib/nsga/`) |
| `run_deap.py`            | DEAP-based NSGA-III runner |
| `run_pymoo.py`           | pymoo-based NSGA-III runner |
| `regenerate_table3.py`   | Top-level: runs each method on Table 3's `(M, k)` grid with `--seeds` repetitions, writes `results/table3.csv` |
| `requirements.txt`       | Standalone dependency manifest |

## Reproducibility notes

* All runners share the **same** DTLZ2 implementation defined locally in
  `dtlz2.py`, equivalent to the platform's in
  [pylib/benchmarks.py](../../pylib/benchmarks.py).
* All runners share the **same** analytical reference front (uniform sample
  on the unit hypersphere in the first orthant — the DTLZ2 Pareto surface).
* The native runner re-uses the SimLab platform's actual
  `fast_nondominated_sort`, `niching_selection`, `sbx`, and `poly_mut`
  modules. The loop around them is **reimplemented**: SBX is applied per gene
  with probability 0.5 and the tournament breaks ties differently, so
  `nsga3_func` is not the code path of a platform experiment.
* Reference points (Das & Dennis): `divisions=12` here. The platform's
  default is 10.
* Default `(pop_size, n_generations) = (92, 400)` mirrors common
  evolutionary-MOO practice for DTLZ2 in the literature.

## Mapping to Table 3 of the paper

The CSV produced by `regenerate_table3.py` has one row per
`(M, k, method)` cell: `M`, `k`, `method`, the `mean_` and `std_` of `hv`,
`gd`, `igd` and `coverage`, `elapsed_s`, `n_seeds` and `igd_p_vs_native`. The
number of variables is `n = M − 1 + k`. The manuscript's layout is:

```
#Obj   k      Method               Mean HV   Mean GD   Mean IGD   Coverage
 2      1     nsga3_func           ...       ...       ...        ...
        1     nsga3_deap_func      ...       ...       ...        ...
        1     nsga3_pymoo_func     ...       ...       ...        ...
 3      2     nsga3_func           ...
...
```

The grid is:
`(M, k) ∈ {(2,1), (3,2), (4,3), (4,4), (5,4), (6,5), (6,14)}`
(see Table 3 of the manuscript, top to bottom).

## Caveats

* DEAP's `selNSGA3` and pymoo's `NSGA3` use slightly different default
  reference-direction layouts; we standardise on Das & Dennis with the same
  `divisions` parameter. They still differ in variation and in what they
  return (below). With 5 seeds, the smallest two-sided p of the exact paired
  Wilcoxon test is 0.0625: 5 seeds cannot show a difference at 0.05.
* HV reference point: each run uses its own `worst_observed + 0.1` per
  dimension, so HV is **not** comparable between methods, seeds or `M`.
* GD here is the RMS of the nearest distances in raw units, **not** the
  platform's normalised mean (p = 1).
* This benchmark is **implementation validation evidence**. It does not go
  through the platform's experiment schema, and it is **not** a claim about
  which library is "better".

## Known issues

* **`regenerate_table3.py` fails on import**: `metrics.py` imports
  `deap.tools._hypervolume`, which DEAP 1.4.4 does not have. It fails with
  `--methods native` too.
* **The runners do not share parameters.** In pymoo, `PM(prob=1/n)` is the
  probability per *individual* (the per-variable one keeps its default), while
  the other two use `1/n` per variable. The DEAP runner has no mating
  tournament.
* **They do not return the same set**: native and DEAP return the whole final
  population, pymoo only its non-dominated solutions.
* With a population of 92 and 12 divisions there are more reference directions
  than individuals for M ≥ 4 (455 to 6188); pymoo warns about it.
