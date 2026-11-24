# Pareto Analysis Tool

Command-line tools that fetch an experiment's results from the SimLab API and
measure them offline:

| Script | Output |
|---|---|
| `plot_pareto_results.py` | Two PNG files, uploaded to the experiment. Exactly 3 objectives |
| `compute_hv_gd.py` | The indicator series as JSON on stdout. Any number of objectives |

## How It Works

1. **Data collection** — fetches all generations and their individuals via the
   SimLab REST API. Individuals without a full objective vector, and those the
   evaluation record marks as invalid (see [Filtering](#filtering)), are dropped.
2. **Pareto sorting** — the unique objective vectors of all generations are
   sorted into dominance layers.
3. **Plot generation** — `plot_pareto_results.py` writes, in the working
   directory:
   - `pareto_fronts_<id>.png` — the dominance layers in the three objectives;
   - `hv_gd_<id>.png` — Hypervolume (per generation and cumulative),
     Generational Distance (GD) and Inverted Generational Distance (IGD / IGD+)
     over generations.
4. **Upload** — each PNG is attached to the experiment record via
   `PATCH /experiments/{id}/analysis-file`, then deleted unless
   `--keep-the-files` is given.

Each objective is a minimisation or maximisation target, set by `--minimize`.

## Setup

From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The scripts go through the reverse proxy and verify TLS against
`nginx/certs/simlab.crt` when it exists.

| Variable | Default | Description |
|---|---|---|
| `SIMLAB_API_BASE` | `https://localhost/api/v1` | API base URL |
| `SIMLAB_API_KEY` | `simlab-api-key42` | API key, Compose's default |
| `SIMLAB_CA_BUNDLE` | the proxy's certificate, if present | CA bundle |
| `SIMLAB_TLS_VERIFY` | `true` | `false` skips verification |

## CLI Arguments

| Argument | `plot_pareto_results.py` | `compute_hv_gd.py` | Description |
|---|---|---|---|
| `--expid` | required | required | Experiment ObjectId |
| `--api-key` | `SIMLAB_API_KEY` | `SIMLAB_API_KEY` | REST API key |
| `--api-base` | `SIMLAB_API_BASE` | `SIMLAB_API_BASE` | Base URL of the SimLab API |
| `--objectives` | `latency energy throughput` | required | Objective names, looked up by name among those the experiment declares; any order, or a subset. An unknown name stops the tool |
| `--minimize` | `True True False` | required | One boolean per objective: `True` minimises, `False` maximises |
| `--true-front-bench` | as `/hv-gd` | as `/hv-gd` | `DTLZ2`, `ZDT1` or `SCH1`: measure against the analytical front. By default, the experiment's benchmark when it is synthetic, every objective is minimised and all are requested |
| `--population` | `survivors` | `survivors` | The set each generation is measured on, as `/hv-gd`: `survivors` (P_t; the offspring where a generation recorded none), `offspring`, or `archive` |
| `--true-front-m` | number of objectives | number of objectives | Objectives of the analytical front |
| `--raw-distances` | off | off | Do not normalise GD, IGD and IGD+ |
| `--keep-the-files` | off | n/a | Keep the local PNG files after upload |

## Examples

```bash
export SIMLAB_API_KEY=simlab-api-key42

# plots of a finished experiment
.venv/bin/python plot_pareto_results.py --expid <experiment_id> \
    --objectives latency power_mw throughput --minimize True True False

# indicators of a synthetic experiment, against the true front
.venv/bin/python compute_hv_gd.py --expid <experiment_id> \
    --objectives f1 f2 f3 --minimize True True True \
    --true-front-bench DTLZ2 --true-front-m 3

# another server
SIMLAB_API_BASE=https://<server>/api/v1 SIMLAB_CA_BUNDLE=<certificate> \
    .venv/bin/python plot_pareto_results.py --expid <experiment_id> \
    --objectives <a> <b> <c> --minimize <True|False> <True|False> <True|False>
```

[`util/run_pareto_analysis.py`](../util/README.md#run_pareto_analysispy--run-pareto-analysis)
runs the plots for every finished experiment, taking the objectives from each
document.

## Convergence indicators

`lib/metrics.py` defines GD, IGD and IGD+ against a reference front. It is a
standalone mirror of the API's canonical `pylib/moo_metrics.py` — the CLIs must
run with no `pylib` on the path — and `tests/test_metrics_parity.py` is what
keeps the two from drifting apart. **Any change to one must be made in the other.**

* **GD** — mean distance from each front point to its nearest reference point
  (the p=1 form, matching `moocore`, `pymoo` and jMetal). Measures convergence.
  For a known benchmark it is the closed-form distance to the true front.
* **IGD** — mean distance from each *reference* point to its nearest front
  point. Measures convergence *and* spread: a front collapsed onto one corner
  of the optimum scores a perfect GD and a poor IGD.
* **IGD+** — the weakly Pareto-compliant variant (Ishibuchi et al., 2015).

The reference is the experiment's stored final front (self-reference), or the
analytical front with `--true-front-bench`. Distances are normalised by the
reference's ideal-nadir range (the theoretical one for a benchmark), so
objectives of different magnitudes contribute comparably.

Hypervolume is always in raw units — it carries its own reference point:
`1.1 × nadir` for a known benchmark, otherwise the run's worst valid value plus
5% and 1 per objective, which is not comparable across runs. The definitions
are discussed in
[SYNTHETIC_MODE.md §7](../docs/markdown/SYNTHETIC_MODE.md#7-convergence-metrics-against-the-true-front).

### Filtering

A vector with a non-finite value is always dropped. Otherwise the evidence
stored with it decides (`completeness` on an individual, `evaluation` on a
front item); only records from before that evidence existed fall back to the
magnitude rule (any |objective| ≥ 1e8 is a penalty).

The reference front also loses duplicate and dominated rows. This matters most
for IGD, which averages *over* the reference — a single penalised row at 1e9
would drag the indicator to 1e9. GD hides the same contamination, because it
minimises over the reference instead.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q
```

## Known issues

None known.
