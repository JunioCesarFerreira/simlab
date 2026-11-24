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

# 3. Regenerate the full Table 3
python regenerate_table3.py --seeds 30 --generations 400
# Output: results/table3.csv

# 4. Tests (pytest is not in requirements.txt)
python -m pytest -q test_benchmark.py
```

## Files

| File | Role |
|---|---|
| `dtlz2.py`               | Shared DTLZ2 implementation (equivalent to `pylib/benchmarks.py`) |
| `metrics.py`             | HV (exact, `moocore`) / GD / IGD / IGD+; non-dominated filter; Mann–Whitney and Â12; analytical reference front sampler |
| `run_native.py`          | SimLab native NSGA-III runner (re-uses `mo-engine/lib/nsga/`) |
| `run_deap.py`            | DEAP-based NSGA-III runner |
| `run_pymoo.py`           | pymoo-based NSGA-III runner |
| `regenerate_table3.py`   | Top-level: runs each method on Table 3's `(M, k)` grid with `--seeds` repetitions, writes `results/table3.csv` |
| `requirements.txt`       | Standalone dependency manifest |
| `test_benchmark.py`      | The runners vary alike and return their population; the metrics, the sizes per M; the table is written |

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
* The three runners share the variation around the environmental selection:
  binary tournament on non-domination rank for mating (pymoo's default for
  NSGA3 is a random pick, and DEAP's example has no mating selection), SBX
  with probability 0.9 per pair and 0.5 per variable (η = 20), polynomial
  mutation with probability 1/n per variable (η = 25), and no removal of
  duplicates. Each library keeps its own implementation of those operators
  and its own environmental selection.
* Each runner returns its final population. Every indicator is computed on the
  non-dominated set of that population, for all three methods.
* HV is exact (`moocore`, as the platform) with the reference point fixed at
  1.1 per objective, 1.1 × the DTLZ2 nadir, as the platform's `/hv-gd`. It is
  comparable between methods and seeds for the same `M`.
* Reference directions (Das & Dennis) and population depend on `M`, as in
  Deb & Jain (2014): the population is the smallest multiple of four above
  the number of directions H, so each direction can hold an individual.

  | M | divisions | H | population |
  |---|---|---|---|
  | 2 | 99 | 100 | 104 |
  | 3 | 12 | 91 | 92 |
  | 4 | 7 | 120 | 124 |
  | 5 | 6 | 210 | 212 |
  | 6 | 4 | 126 | 128 |

  M = 3 and M = 5 are Deb & Jain's settings for DTLZ; the others follow the same
  rule for an H of the same order. `--divisions` and `--pop-size` impose one
  value for every M. The platform's default is 10 divisions.
* 400 generations for every cell, at least what Deb & Jain use for DTLZ2 up to
  M = 5 (250 and 350).

## Mapping to Table 3 of the paper

The CSV produced by `regenerate_table3.py` has one row per
`(M, k, method)` cell: `M`, `k`, `method`, `divisions`, `directions`,
`pop_size`, the `mean_` and `std_` of `hv`, `gd`, `igd` and `igd_plus`,
`elapsed_s`, `n_seeds`, and against the native method on the per-run IGD,
`igd_p_vs_native` (two-sided Mann–Whitney U) and `igd_a12_vs_native` (the
probability that a run of the method has a lower IGD than a run of the native
one; 0.5 is no difference). The number of variables is `n = M − 1 + k`.

The manuscript's layout had a Coverage column: C(A, R) against the true front,
which is 0 by construction, since no feasible point dominates a Pareto-optimal
one. IGD+ takes its place:

```
#Obj   k      Method               Mean HV   Mean GD   Mean IGD   Mean IGD+
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
  `divisions` parameter.
* The comparison is unpaired: the runs of two methods share nothing but a seed
  number, since each library draws from its own generator. With 5 runs per
  method the smallest two-sided p of the exact test is 2/252 ≈ 0.008; the
  campaign uses 30. A p-value says whether a difference is detectable, Â12 how
  large it is; neither shows equivalence, which needs a margin declared
  beforehand (TOST).
* HV is not comparable between different `M`: the box it is measured in grows
  as 1.1^M.
* GD here is the RMS of the nearest distances in raw units, **not** the
  platform's normalised mean (p = 1).
* This benchmark is **implementation validation evidence**. It does not go
  through the platform's experiment schema, and it is **not** a claim about
  which library is "better".

## Known issues

None known.
