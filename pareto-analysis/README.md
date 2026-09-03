# Pareto Analysis Tool

A command-line tool that fetches experiment results from the SimLab API, performs multi-objective Pareto analysis, generates visualisation plots, and uploads them back to the experiment.

## How It Works

1. **Data collection** — fetches all generations and individuals for a given experiment via the SimLab REST API.
2. **Pareto sorting** — applies fast non-dominated sorting to classify individuals into Pareto fronts across the configured objectives.
3. **Plot generation** — produces multiple PNG files:
   - Pareto front scatter plots (per pair of objectives)
   - Individual distribution per front and per generation
   - Global population distribution
   - Parallel coordinates chart
   - Radar chart
   - Hypervolume (per generation and cumulative), Generational Distance (GD) and
     Inverted Generational Distance (IGD / IGD+) evolution over generations
4. **Upload** — each generated PNG is attached to the experiment record via `PATCH /experiments/{id}/analysis-file`.

Objectives can be individually configured as minimisation or maximisation targets via CLI flags.

## Convergence indicators

`lib/metrics.py` defines GD, IGD and IGD+ against a reference front. It is a
standalone mirror of the API's canonical `pylib/moo_metrics.py` — the CLIs must
run with no `pylib` on the path — and `tests/test_metrics_parity.py` is what
keeps the two from drifting apart. **Any change to one must be made in the other.**

* **GD** — mean distance from each front point to its nearest reference point
  (the p=1 form, matching `moocore`, `pymoo` and jMetal). Measures convergence.
* **IGD** — mean distance from each *reference* point to its nearest front
  point. Measures convergence *and* spread: a front collapsed onto one corner
  of the optimum scores a perfect GD and a poor IGD.
* **IGD+** — the weakly Pareto-compliant variant (Ishibuchi et al., 2015).

The reference front is sanitised before use: penalised rows (any |objective| ≥
1e8), duplicates and dominated rows are dropped. This matters most for IGD,
which averages *over* the reference — a single penalised row at 1e9 would drag
the indicator to 1e9. GD hides the same contamination, because it minimises
over the reference instead.

Distances are normalised by the reference front's ideal-nadir range by default,
so objectives of different magnitudes contribute comparably; pass
`--raw-distances` for the unnormalised values. Hypervolume is always in raw
units — it carries its own reference point.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--expid` | *(required)* | Experiment ObjectId |
| `--api-key` | *(required)* | REST API key |
| `--api-base` | `http://localhost:8198/api/v1` | Base URL of the SimLab API |
| `--objectives` | `throughput delay pdr` | Space-separated list of objective names |
| `--minimize` | `delay` | Objectives to minimise (rest are maximised) |
| `--keep-the-files` | `False` | Keep local PNG files after upload |

## Commands Examples

**Localhost**

```bash
py plot_pareto_results.py --expid 69c677b93b925d1dafcb4151 --api-key api-password --keep-the-files True
```

**Server**

Replece `secret` with your api key.

```bash
python3 plot_pareto_results.py --expid 69b97e5c911986d1242e5a7e --api-key secret --api-base http://localhost:8198/api/v1
```


**Local/Server**

```bash
py plot_pareto_results.py --expid 69b97e5c911986d1242e5a7e --keep-the-files --api-key secret --api-base http://andromeda.lasdpc.icmc.usp.br/:8198/api/v1
```

**host linux**
```bash
cd /home/github/simlab/pareto-analysis
./.venv/bin/python plot_pareto_results.py --expid 6a92d3018d392109abdffaba \
    --objectives latency energy throughput --minimize True True False
```