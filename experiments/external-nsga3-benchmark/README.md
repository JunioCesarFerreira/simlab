# External NSGA-III Benchmark

This directory contains the **standalone scripts** that regenerate **Table 3**
of the SimLab paper — the DTLZ2 quality comparison between three NSGA-III
implementations:

| Method label in Table 3 | Source |
|---|---|
| `nsga3_func`        | SimLab native: [mo-engine/lib/nsga/](../../mo-engine/lib/nsga/) |
| `nsga3_deap_func`   | [DEAP `selNSGA3`](https://deap.readthedocs.io/) |
| `nsga3_pymoo_func`  | [pymoo `NSGA3`](https://pymoo.org/algorithms/moo/nsga3.html) |

The scripts here are **independent of the SimLab runtime** — there is no
MongoDB, no master-node, no Cooja. They evaluate NSGA-III on the analytical
DTLZ2 benchmark so the Table 3 comparison is reproducible from a single
command on a clean Python environment.

## Why "external"?

DEAP and pymoo are heavyweight scientific stacks. Bringing them into
[mo-engine/requirements.txt](../../mo-engine/requirements.txt) would inflate
the production Docker image with dependencies the platform never uses at
runtime (the platform's runtime NSGA-III is the native version, see
[mo-engine/lib/strategy/nsga3.py](../../mo-engine/lib/strategy/nsga3.py)).

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
python regenerate_table3.py --seeds 5 --pop-size 92 --generations 400
# Output: results/table3.csv
```

## Files

| File | Role |
|---|---|
| `dtlz2.py`               | Shared DTLZ2 implementation (mirrors `master-node/lib/synthetic_data.py`) |
| `metrics.py`             | HV / GD / IGD / Coverage; analytical reference front sampler |
| `run_native.py`          | SimLab native NSGA-III runner (re-uses `mo-engine/lib/nsga/`) |
| `run_deap.py`            | DEAP-based NSGA-III runner |
| `run_pymoo.py`           | pymoo-based NSGA-III runner |
| `regenerate_table3.py`   | Top-level: runs each method on Table 3's `(M, k)` grid with `--seeds` repetitions, writes `results/table3.csv` |
| `requirements.txt`       | Standalone dependency manifest |

## Reproducibility notes

* All runners share the **same** DTLZ2 implementation defined locally in
  `dtlz2.py`, which is line-for-line equivalent to the one in
  [master-node/lib/synthetic_data.py:33-51](../../master-node/lib/synthetic_data.py#L33-L51).
* All runners share the **same** analytical reference front (uniform sample
  on the unit hypersphere in the first orthant — the DTLZ2 Pareto surface).
* The native runner re-uses the SimLab platform's actual
  `fast_nondominated_sort`, `niching_selection`, `sbx`, and `poly_mut`
  modules — so `nsga3_func` in Table 3 is the **same algorithmic code**
  that runs in production.
* Reference points (Das & Dennis): `divisions=12` matches the default
  used by the platform unless overridden in the experiment payload.
* Default `(pop_size, n_generations) = (92, 400)` mirrors common
  evolutionary-MOO practice for DTLZ2 in the literature.

## Mapping to Table 3 of the paper

The CSV produced by `regenerate_table3.py` has one row per
`(M, k, method)` cell with columns `mean_hv`, `mean_gd`, `mean_igd`,
`coverage`. Compare directly against the manuscript:

```
#Obj   #Var   Method               Mean HV   Mean GD   Mean IGD   Coverage
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
  `divisions` parameter, but results can still differ at the third decimal
  place due to internal RNG handling. Always report `--seeds` ≥ 5 to smooth
  out these effects.
* HV reference point: each run uses `worst_observed + 0.1` per dimension —
  reasonable for DTLZ2 in `[0, 1]`-scaled spaces but **not** comparable
  across `M` (Hypervolume scales with dim by definition).
* This benchmark is for **architectural validation evidence** (the platform
  successfully integrates three NSGA-III implementations under the same
  experiment schema). It is **not** a claim about which library is "better".
