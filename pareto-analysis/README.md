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
   - Hypervolume and Generational Distance (GD) evolution over generations
4. **Upload** — each generated PNG is attached to the experiment record via `PATCH /experiments/{id}/analysis-file`.

Objectives can be individually configured as minimisation or maximisation targets via CLI flags.

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
py plot_pareto_results.py --expid 69c4096f8f366d49ef753a50 --api-key api-password --keep-the-files True
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
