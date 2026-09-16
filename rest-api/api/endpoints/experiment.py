import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import moocore

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from bson import errors as bson_errors
from tempfile import NamedTemporaryFile
from pydantic import BaseModel

from bson import ObjectId

from pylib import benchmarks, moo_metrics
from pylib.db import MongoRepository
from pylib.db.models.enums import EnumStatus
from api.dependencies import get_factory
from api.metrics_cache import hv_gd_cache
from api.domain.experiment import ExperimentDto, ExperimentFullDto, ExperimentInfoDto
from api.mappers.experiment import (
    experiment_from_mongo,
    experiment_full_from_mongo,
    experiment_info_from_mongo,
    experiment_to_mongo,
    _runtime_metrics_from_mongo,
)
from api.mappers.generation import generation_from_mongo

router = APIRouter()


@router.post("/", response_model=str)
def create_experiment(
    experiment: ExperimentDto,
    factory: MongoRepository = Depends(get_factory)
) -> str:
    """Create a new experiment. Returns the generated experiment_id."""
    try:
        doc = experiment_to_mongo(experiment)
        return str(factory.experiment_repo.insert(doc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=list[ExperimentInfoDto])
def get_all_experiments(
    factory: MongoRepository = Depends(get_factory)
) -> list[ExperimentInfoDto]:
    """Lightweight listing of every experiment, status included."""
    try:
        docs = factory.experiment_repo.find_all_info()
        return [experiment_info_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-status/{status}", response_model=list[ExperimentInfoDto])
def get_experiments_by_status(
    status: str,
    factory: MongoRepository = Depends(get_factory)
) -> list[ExperimentInfoDto]:
    """Retrieve all experiments with a given status."""
    try:
        docs = factory.experiment_repo.find_by_status(status)
        return [experiment_info_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}/full", response_model=ExperimentFullDto)
def get_experiment_full(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> ExperimentFullDto:
    """Retrieve an experiment with all its generations and individuals fully embedded."""
    try:
        doc = factory.experiment_repo.get(experiment_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Experiment not found")
        gens = factory.generation_repo.find_by_experiment(ObjectId(experiment_id))
        sims_by_individual = factory.simulation_repo.find_ids_grouped_by_individual(
            ObjectId(experiment_id)
        )
        individuals = factory.individual_repo.find_grouped_by_experiment(ObjectId(experiment_id))
        generations = [
            generation_from_mongo(
                g,
                individuals.get(g["_id"], []),
                sims_by_individual,
            )
            for g in gens
        ]
        return experiment_full_from_mongo(doc, generations)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{experiment_id}", response_model=ExperimentDto)
def get_experiment(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> ExperimentDto:
    """Retrieve a single experiment by its ObjectId."""
    try:
        doc = factory.experiment_repo.get(experiment_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return experiment_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{experiment_id}", response_model=bool)
def update_experiment(
    experiment_id: str,
    updates: dict,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Partially update an experiment using $set semantics."""
    try:
        return factory.experiment_repo.update(experiment_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{experiment_id}", response_model=bool)
def delete_experiment(
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Delete an experiment and all its associated data (cascade)."""
    try:
        res = factory.experiment_repo.delete(experiment_id)
        if isinstance(res, dict):
            return res.get("deleted_experiments", 0) == 1
        return bool(res)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{experiment_id}/status", response_model=bool)
def update_experiment_status(
    experiment_id: str,
    new_status: str,
    factory: MongoRepository = Depends(get_factory)
) -> bool:
    """Update only the status field of an experiment."""
    valid = {s.value for s in EnumStatus}
    if new_status not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status {new_status!r}; expected one of {sorted(valid)}",
        )
    try:
        factory.experiment_repo.update_status(experiment_id, new_status)
        return True
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{experiment_id}/analysis-file", response_model=str)
async def attach_analysis_file(
    experiment_id: str,
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    factory: MongoRepository = Depends(get_factory)
) -> str:
    """Upload and attach an analysis file to an experiment. Returns the GridFS file_id."""
    try:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp.flush()
            tmp_path = tmp.name
        oid = factory.experiment_repo.add_analysis_file_to_experiment(
            experiment_id, description, tmp_path, name
        )
        return str(oid)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Runtime (computational) telemetry ────────────────────────────────────────

def _downsample_points(points: list[list[float]], max_points: int) -> list[list[float]]:
    """Bucket-average a [[ts, value], ...] series down to at most max_points.

    Keeps the chart payload small; exact extremes remain available in the
    stored summary and in the raw GridFS artifact.
    """
    n = len(points)
    if n <= max_points:
        return points
    bucket = -(-n // max_points)  # ceil division
    out: list[list[float]] = []
    for i in range(0, n, bucket):
        chunk = points[i:i + bucket]
        out.append([
            sum(p[0] for p in chunk) / len(chunk),
            sum(p[1] for p in chunk) / len(chunk),
        ])
    return out


@router.get("/{experiment_id}/runtime-metrics")
def get_experiment_runtime_metrics(
    experiment_id: str,
    max_points: int = Query(1000, ge=10, le=20000),
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """Full runtime-metrics time series of an experiment.

    Loads the raw telemetry artifact from GridFS, reconstructs every series
    and downsamples each one to at most ``max_points`` samples. The summary
    block is returned by GET /experiments/{id}; this endpoint exists so the
    heavy series are only transferred on demand.
    """
    from pylib.telemetry.artifact import deserialize_samples

    doc = factory.experiment_repo.get(experiment_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Experiment not found")

    rm: dict = doc.get("runtime_metrics") or {}
    if not rm:
        raise HTTPException(
            status_code=404,
            detail="Runtime metrics not available for this experiment",
        )

    base = {
        "status": rm.get("status"),
        "started_at": rm.get("started_at"),
        "finished_at": rm.get("finished_at"),
        "summary": rm.get("summary") or {},
        "series": [],
        "downsampled": False,
        "total_samples": 0,
    }

    artifact: dict = rm.get("artifact") or {}
    file_id = artifact.get("file_id")
    if rm.get("status") != "completed" or not file_id:
        # collecting / no_data / failed — nothing to plot yet
        return base

    try:
        data = factory.fs_handler.read_file_content(str(file_id))
        samples = deserialize_samples(data, artifact.get("content_type", ""))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read telemetry artifact: {e}"
        )

    # Rebuild the original series: one per (metric, scope, label-set)
    grouped: dict[tuple, dict] = {}
    for s in samples:
        labels: dict = s.get("labels") or {}
        key = (s["metric"], s["scope"], json.dumps(labels, sort_keys=True))
        entry = grouped.setdefault(key, {
            "metric": s["metric"],
            "scope": s["scope"],
            "unit": s.get("unit", ""),
            "labels": labels,
            "name": labels.get("name") or s["scope"],
            "points": [],
        })
        entry["points"].append([float(s["timestamp"]), float(s["value"])])

    downsampled = False
    series = []
    for entry in grouped.values():
        entry["points"].sort(key=lambda p: p[0])
        reduced = _downsample_points(entry["points"], max_points)
        downsampled = downsampled or len(reduced) < len(entry["points"])
        series.append({**entry, "points": reduced})
    series.sort(key=lambda e: (e["metric"], e["scope"], e["name"]))

    return {
        **base,
        "series": series,
        "downsampled": downsampled,
        "total_samples": len(samples),
    }


def _as_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Drop tz info so an override matches the naive timestamps SimLab stores.

    Experiment start/end times are persisted as naive local datetimes; a
    tz-aware override (e.g. an ISO string ending in 'Z') is converted to the
    same local wall-clock so the Prometheus query window stays consistent.
    """
    if isinstance(dt, datetime) and dt.tzinfo is not None:
        return dt.astimezone().replace(tzinfo=None)
    return dt


class RuntimeMetricsCollectRequest(BaseModel):
    """Optional [start, end] override for a manual collection.

    When omitted, the experiment's own start_time/end_time are used. Times are
    interpreted in the server's local time, matching stored experiment
    timestamps.
    """
    start: Optional[datetime] = None
    end: Optional[datetime] = None


@router.post("/{experiment_id}/runtime-metrics/collect")
def collect_experiment_runtime_metrics(
    experiment_id: str,
    body: Optional[RuntimeMetricsCollectRequest] = None,
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """Manually (re)collect runtime metrics from Prometheus for an experiment.

    Overwrites any existing block. Intended as the operator-facing retry when
    the automatic collection found Prometheus unreachable, failed, or captured
    no data: monitoring is back, so query the ``[start, end]`` window again —
    optionally corrected via the request body. Returns the resulting collection
    ``status`` and the refreshed ``runtime_metrics`` summary block.
    """
    from pylib.telemetry.collector import collect_and_store

    try:
        doc = factory.experiment_repo.get(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    if not doc:
        raise HTTPException(status_code=404, detail="Experiment not found")

    body = body or RuntimeMetricsCollectRequest()
    started_at = _as_naive(body.start or doc.get("start_time"))
    finished_at = _as_naive(body.end or doc.get("end_time") or datetime.now())

    if not started_at:
        raise HTTPException(
            status_code=400,
            detail="Experiment has no start_time; provide an explicit 'start' to collect.",
        )
    if finished_at <= started_at:
        raise HTTPException(status_code=400, detail="'end' must be after 'start'.")

    current_status = (doc.get("runtime_metrics") or {}).get("status")
    if current_status == "collecting":
        raise HTTPException(
            status_code=409,
            detail="A telemetry collection is already in progress for this experiment.",
        )

    try:
        status = collect_and_store(
            factory, experiment_id, started_at, finished_at, force=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Runtime metrics collection failed: {e}"
        )

    rm = factory.experiment_repo.get_runtime_metrics(experiment_id)
    return {"status": status, "runtime_metrics": _runtime_metrics_from_mongo(rm)}


class ParetoPlotRequest(BaseModel):
    objectives: list[str]
    minimize: list[bool]


# Sibling of rest-api/, resolved from this file rather than from the working
# directory or a build-time absolute path: the same expression yields
# /app/pareto-analysis in the container image and <repo>/pareto-analysis in a
# checkout, the two layouts the API runs in.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_PARETO_SCRIPT = os.getenv(
    "SIMLAB_PARETO_SCRIPT",
    os.path.join(_REPO_ROOT, "pareto-analysis", "plot_pareto_results.py"),
)
# sys.executable is the interpreter already serving the API, so it exists by
# construction and carries the dependencies installed alongside the API.
_PARETO_PYTHON = os.getenv("SIMLAB_PARETO_PYTHON", sys.executable)


@router.post("/{experiment_id}/plot-pareto")
def plot_pareto_results(
    experiment_id: str,
    body: ParetoPlotRequest,
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """Run pareto analysis script and upload resulting plots to the experiment."""
    try:
        doc = factory.experiment_repo.get(experiment_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Experiment not found")
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")

    if len(body.objectives) < 3 or len(body.minimize) < 3:
        raise HTTPException(status_code=422, detail="objectives and minimize must each have at least 3 items")

    if not os.path.isfile(_PARETO_SCRIPT):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Pareto analysis script not found at {_PARETO_SCRIPT}. "
                "Set SIMLAB_PARETO_SCRIPT to its location, or rebuild the API "
                "image so pareto-analysis/ ships with it."
            ),
        )

    api_key = os.getenv("SIMLAB_API_KEY", "api-password")
    minimize_strs = [str(m) for m in body.minimize[:3]]

    cmd = [
        _PARETO_PYTHON, _PARETO_SCRIPT,
        "--expid", experiment_id,
        "--objectives", *body.objectives[:3],
        "--minimize", *minimize_strs,
        # The script reads generations and uploads the plots back over HTTP, so
        # it needs a URL for this same API; the default is the port it serves on.
        "--api-base", os.getenv("SIMLAB_API_BASE", "http://localhost:8000/api/v1"),
        "--api-key", api_key,
    ]

    # Synthetic experiments have a closed-form Pareto front: measure HV/GD
    # against the benchmark's analytical front instead of the run's own
    # empirical references (mirrors the /hv-gd endpoint).
    syn = (((doc.get("parameters") or {}).get("simulation") or {}).get("synthetic") or {})
    bench = str(syn.get("bench") or "").upper()
    if syn.get("enabled") and bench in ("DTLZ2", "ZDT1", "SCH1") and all(body.minimize[:3]):
        cmd += ["--true-front-bench", bench, "--true-front-m", str(len(body.objectives[:3]))]

    # The script renders to PNG with no display attached, and matplotlib needs a
    # writable config dir; neither is guaranteed by the API's own environment.
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(_PARETO_SCRIPT),
            env=env,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script failed:\n{result.stderr}")
        return {"status": "ok", "output": result.stdout}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Pareto analysis timed out (10 min limit)")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── HV/GD inline helpers ──────────────────────────────────────────────────────

_PENALTY_THRESHOLD = moo_metrics.PENALTY_THRESHOLD


def _is_penalized(objs: list[float]) -> bool:
    return any(abs(v) >= _PENALTY_THRESHOLD for v in objs)


def _front_rows(objs_list: list[list[float]], minimize: list[bool]) -> list[list[float]]:
    """Deduplicated non-dominated subset, returned in minimization space."""
    if not objs_list:
        return []
    points = np.asarray(objs_list, dtype=float)
    points *= np.where(minimize, 1.0, -1.0)
    return moocore.filter_dominated(points).tolist()


def _empty_hv_gd() -> dict:
    """Response shape when there is nothing to measure — same keys, empty series."""
    return {
        "generations": [],
        "hv": [],
        "hv_cumulative": [],
        "gd": [],
        "igd": [],
        "igd_plus": [],
        "reference": None,
        "reference_size": 0,
        "normalized": False,
        "worst_point": {},
        "population": None,
        "population_source": None,
        "population_sources": [],
        "gd_method": None,
        "gd_formula": None,
        "normalization": None,
    }


@router.get("/{experiment_id}/hv-gd")
def get_hv_gd(
    experiment_id: str,
    objectives: list[str] = Query(...),
    minimize: list[str] = Query(...),
    population: str = Query(
        "survivors",
        pattern="^(survivors|offspring|archive)$",
        description=(
            "Which set each generation is measured on. 'survivors' is the "
            "population environmental selection kept (P_t) — what the algorithm "
            "carries forward and what the reference notebooks plot. 'offspring' "
            "is the children evaluated in that generation (Q_t): it swings with "
            "each batch and can drop while the search still holds a better "
            "parent. 'archive' is the non-dominated set of everything seen so "
            "far, which is monotone in HV by construction. Generations without "
            "a persisted survivor set fall back to 'offspring'; the response "
            "reports each generation's set in 'population_sources'; "
            "'population_source' is 'mixed' when the series uses both sets."
        ),
    ),
    normalize: bool = Query(
        True,
        description=(
            "Normalise GD/IGD/IGD+ by the reference front's ideal-nadir range, so "
            "objectives on different scales contribute comparably. Hypervolume is "
            "unaffected — it keeps its own reference point in raw units."
        ),
    ),
    factory: MongoRepository = Depends(get_factory),
    include_cumulative: bool = True,
) -> dict:
    """Compute exact indicators. Set include_cumulative=false to omit the extra
    archive HV series (hv_cumulative=[]), avoiding its cost for population plots.
    The selected population's HV/GD/IGD/IGD+ remain unchanged.
    """
    try:
        exp_oid = ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")

    n_obj = len(objectives)
    if n_obj < 2 or len(minimize) != n_obj:
        raise HTTPException(
            status_code=422,
            detail="objectives and minimize must have the same length (≥ 2)",
        )

    doc = factory.experiment_repo.get(experiment_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Experiment not found")

    gens = factory.generation_repo.find_by_experiment(exp_oid)
    individuals = factory.individual_repo.find_grouped_by_experiment(
        exp_oid, objectives_only=True,
    )
    # Read current inputs before looking up the cache: objective updates within
    # an existing generation and newly persisted survivor sets must invalidate it.
    # Chromosomes, simulations and unrelated experiment metadata are excluded.
    inputs = [
        objectives, minimize, population, normalize, include_cumulative,
        (doc.get("parameters") or {}).get("objectives"),
        ((doc.get("parameters") or {}).get("simulation") or {}).get("synthetic"),
        [p.get("objectives") for p in doc.get("pareto_front") or []],
        [[g["index"], g.get("survivors"),
          [[i.get("individual_id"), i.get("objectives")]
           for i in individuals.get(g["_id"], [])]] for g in gens],
    ]
    return hv_gd_cache.get_or_compute(
        inputs,
        lambda: _compute_hv_gd(doc, gens, individuals, objectives, minimize, population,
                               normalize, include_cumulative),
    )


def _compute_hv_gd(
    doc: dict, gens: list[dict], individuals: dict,
    objectives: list[str], minimize: list[str], population: str, normalize: bool,
    include_cumulative: bool = True,
) -> dict:
    n_obj = len(objectives)
    stored_pf: list[dict] = doc.get("pareto_front") or []
    minimize_bools = [m.lower() == "true" for m in minimize]

    # Individuals store their objectives as a POSITIONAL list, in the order the
    # experiment declared them. Reading the first n_obj entries instead of
    # looking the names up silently mismatches the moment the request reorders
    # or subsets the objectives: asking for [f2, f1] used to compare f2 against
    # the f1 reference, which turned a GD of 0 into 11.31.
    declared = ((doc.get("parameters") or {}).get("objectives") or [])
    canonical = [str(o.get("metric_name")) for o in declared if o.get("metric_name")]
    if canonical:
        unknown = [o for o in objectives if o not in canonical]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown objective(s) {unknown}; this experiment declares "
                    f"{canonical}."
                ),
            )
        objective_columns = [canonical.index(o) for o in objectives]
    else:
        # Experiments written before objectives were declared on the document
        # have nothing to resolve against; positional order is all there is.
        objective_columns = list(range(n_obj))

    # ── Resolve the projected individuals and survivor sets ──────────────────
    individuals_per_gen: dict[int, list[list[float]]] = {}
    # Survivors are chromosome hashes, and one may have been evaluated in an
    # older generation, so they are resolved against the whole experiment.
    objectives_by_hash: dict[str, list[float]] = {}
    survivor_hashes_per_gen: dict[int, list[str]] = {}
    for gen in gens:
        gen_idx: int = gen["index"]
        valid: list[list[float]] = []
        for ind in individuals.get(gen["_id"], []):
            raw = ind.get("objectives") or []
            if len(raw) <= max(objective_columns):
                continue
            objs = [float(raw[i]) for i in objective_columns]
            if _is_penalized(objs) or not np.isfinite(objs).all():
                continue
            valid.append(objs)
            ind_hash = ind.get("individual_id")
            if ind_hash:
                objectives_by_hash[ind_hash] = objs
        individuals_per_gen[gen_idx] = valid
        stored_survivors = gen.get("survivors")
        if stored_survivors is not None:
            survivor_hashes_per_gen[gen_idx] = list(stored_survivors)

    if not individuals_per_gen:
        return _empty_hv_gd()

    # Legacy generations, incomplete generations and failed metadata writes can
    # leave P_t unavailable. Fall back only for those generations and report
    # their source: a mixed series is not a homogeneous survivor trajectory.
    generations_sorted = sorted(individuals_per_gen)
    population_sources = [
        "offspring" if population == "survivors" and g not in survivor_hashes_per_gen
        else population for g in generations_sorted
    ]
    sources = set(population_sources)
    population_source = population_sources[0] if len(sources) == 1 else "mixed"
    needs_archive = include_cumulative or population == "archive"

    # ── Reference front (GD/IGD/IGD+) + HV reference point ───────────────────
    # Synthetic experiments have a closed-form true Pareto front: use it as the
    # GD/IGD reference (measuring convergence to the real optimum, not to the
    # run's own final front) and a FIXED nadir as the HV reference so HV is
    # comparable across runs of the same benchmark. WSN experiments keep the
    # empirical references (own stored front + population-derived worst point).
    syn = (((doc.get("parameters") or {}).get("simulation") or {}).get("synthetic") or {})
    bench = syn.get("bench")
    # A benchmark's front is defined for the FULL objective set it was run with.
    # The front of a projection onto fewer objectives is not the front of the
    # smaller benchmark — DTLZ2 with M=3 read on two axes is not DTLZ2 M=2 — so
    # a subset request falls back to the empirical reference. A permutation is
    # fine: the analytical front is reordered to match below.
    covers_all_objectives = bool(canonical) and sorted(objectives) == sorted(canonical)
    is_synthetic = (
        bool(syn.get("enabled"))
        and bool(bench)
        and all(minimize_bools)
        and (covers_all_objectives or not canonical)
    )

    reference_kind = "final_front"
    reference_front = None
    hv_ref: list[float] = []
    # Theoretical ideal-nadir range, when the benchmark provides one. Used both
    # to normalise the indicators (instead of the sampled reference's extremes,
    # which depend on how that sample was drawn) and to decide whether GD can
    # take the exact route below.
    analytical_bounds: tuple[np.ndarray, np.ndarray] | None = None
    if is_synthetic:
        try:
            # Built in the experiment's declared order, then reordered to the
            # request. ZDT1 is not symmetric in its objectives, so a permuted
            # request needs a permuted front, not the same one.
            columns = objective_columns if canonical else list(range(n_obj))
            reference_front = benchmarks.true_front(bench, n_obj)[:, columns]
            nadir = benchmarks.nadir(bench, n_obj)
            ideal = benchmarks.ideal(bench, n_obj)
            hv_ref = [nadir[i] * 1.1 for i in columns]
            analytical_bounds = (
                np.array([ideal[i] for i in columns], dtype=float),
                np.array([nadir[i] for i in columns], dtype=float),
            )
            reference_kind = "true_front"
        except (ValueError, IndexError):
            is_synthetic = False  # unknown benchmark → fall back to empirical
            analytical_bounds = None

    if not is_synthetic:
        # Reference point in MINIMIZATION space, consistent with pts_min below.
        # Max objectives must be negated first: computing the worst value in raw
        # space would place the reference on the wrong side of a maximized axis,
        # inflating HV by a large constant and masking its per-generation growth.
        all_min = [
            [o[i] if minimize_bools[i] else -o[i] for i in range(n_obj)]
            for v in individuals_per_gen.values() for o in v
        ]
        # Every individual penalised (or none feasible) leaves nothing to derive
        # a reference point from; max() on the empty set used to raise a 500.
        if not all_min:
            return _empty_hv_gd()
        worst = [max(row[i] for row in all_min) for i in range(n_obj)]
        hv_ref = [v + abs(v) * 0.05 + 1.0 for v in worst]
        # The stored front is the empirical reference. Only this branch needs
        # it, so a synthetic run with none still gets its analytical series —
        # the early return here used to deny them to every experiment alike.
        if not stored_pf:
            return _empty_hv_gd()
        ref_min_rows: list[list[float]] = []
        for p in stored_pf:
            objs_dict: dict = p.get("objectives") or {}
            # A row missing an objective is dropped rather than defaulted: a
            # missing value read as 0.0 would look optimal on a minimised axis
            # and pull the whole reference front towards the origin.
            if any(o not in objs_dict for o in objectives):
                continue
            ref_min_rows.append([
                float(objs_dict[o]) if minimize_bools[i] else -float(objs_dict[o])
                for i, o in enumerate(objectives)
            ])
        reference_front = ref_min_rows

    # Penalised, duplicate and dominated rows are stripped before the front is
    # used as a reference. IGD averages over the reference, so a single
    # penalised row (≥ 1e8) would dominate the mean outright — GD never exposed
    # this because it minimises over the reference instead.
    reference_front = moo_metrics.sanitize_reference_front(reference_front)
    if reference_front.size == 0:
        return _empty_hv_gd()

    hv_ref_arr = np.array(hv_ref, dtype=float)

    # GD is the mean distance from each front point to the true front. With a
    # sampled reference that mean cannot go below the sample's fill distance —
    # points sitting EXACTLY on the DTLZ2 front score 0.19 at M=6 against the
    # 500-point reference, and still 0.05 against 200 000 points, because the
    # fill distance of a (M-1)-dimensional manifold shrinks only as
    # N**(-1/(M-1)). Where the distance has a closed form, use it: the same
    # points then score ~5e-17. IGD and IGD+ average over the reference set
    # itself, so they keep the sampled front and its floor.
    gd_scale: float | None = None
    if analytical_bounds is not None:
        scale = moo_metrics.analytical_scale(*analytical_bounds)
        # The closed forms are Euclidean in raw space, so only an isotropic
        # range carries through exactly. All current benchmarks have one.
        if np.allclose(scale, scale[0]):
            gd_scale = float(scale[0]) if normalize else 1.0
    gd_method = "analytical" if gd_scale is not None else "reference_front"

    distance_reference = reference_front
    if normalize:
        if analytical_bounds is None:
            distance_ideal, distance_scale = moo_metrics.normalization_bounds(reference_front)
        else:
            distance_ideal = analytical_bounds[0]
            distance_scale = moo_metrics.analytical_scale(*analytical_bounds)
        distance_reference = moo_metrics.normalize(reference_front, distance_ideal, distance_scale)

    # ── Per-generation HV / GD / IGD / IGD+ ──────────────────────────────────
    # GD, IGD and IGD+ are three readings of the same comparison and are cheap
    # once the reference front is in hand, so all three are returned: GD alone
    # rewards a population that converged onto a corner of the front, IGD adds
    # the spread requirement, and IGD+ is the Pareto-compliant variant.
    #
    # Two HV curves are returned:
    #   • hv            — the Pareto front of the SELECTED population (see the
    #     ``population`` parameter) at each generation.
    #   • hv_cumulative — the front of every individual seen up to and including
    #     the generation ("best-so-far"). It is monotonically non-decreasing,
    #     independent of ``population``, and built incrementally: the running
    #     non-dominated set is folded with each generation's offspring front
    #     using moocore's native dominance filter. With ``population=archive`` the two
    #     curves coincide by construction.
    hv_values: list[float] = []
    hv_cumulative: list[float] = []
    gd_values: list[float | None] = []
    igd_values: list[float | None] = []
    igd_plus_values: list[float | None] = []

    acc_seen: set[tuple] = set()        # dedup keys of the running front
    acc_rows: list[list[float]] = []    # running non-dominated set (min-space)
    last_cum_hv = 0.0

    for gen_idx, generation_source in zip(generations_sorted, population_sources):
        # The archive always folds in the OFFSPRING, whichever set is reported:
        # it is the set of everything evaluated, and survivors are a subset of
        # earlier offspring. A point off its own generation's front is dominated
        # within that generation too, so it can never join the accumulated front
        # — merging the front alone keeps the set minimal.
        offspring_rows = (
            _front_rows(individuals_per_gen.get(gen_idx, []), minimize_bools)
            if needs_archive or generation_source == "offspring" else []
        )
        new_rows = (
            [row for row in offspring_rows if tuple(row) not in acc_seen] if needs_archive else []
        )
        archive_changed = False
        if new_rows:
            merged = moocore.filter_dominated(np.asarray(acc_rows + new_rows, dtype=float)).tolist()
            merged_keys = {tuple(r) for r in merged}
            archive_changed = merged_keys != acc_seen
            acc_rows, acc_seen = merged, merged_keys

        if generation_source == "survivors":
            survivor_objs = [
                objectives_by_hash[h]
                for h in survivor_hashes_per_gen.get(gen_idx, [])
                if h in objectives_by_hash
            ]
            pts_min_rows = _front_rows(survivor_objs, minimize_bools)
        elif generation_source == "archive":
            pts_min_rows = acc_rows
        else:
            pts_min_rows = offspring_rows

        cum_hv = last_cum_hv
        if archive_changed:
            acc_arr = np.asarray(acc_rows, dtype=float)
            acc_dom = acc_arr[np.all(acc_arr < hv_ref_arr, axis=1)]
            cum_hv = float(moocore.hypervolume(acc_dom, ref=hv_ref)) if len(acc_dom) else 0.0
        last_cum_hv = cum_hv

        if not pts_min_rows:
            hv_values.append(0.0)
            hv_cumulative.append(cum_hv)
            gd_values.append(None)
            igd_values.append(None)
            igd_plus_values.append(None)
            continue

        pts_min = np.array(pts_min_rows, dtype=float)

        # HV: only points that strictly dominate the (fixed) reference contribute.
        dominating = pts_min[np.all(pts_min < hv_ref_arr, axis=1)]
        hv_val = cum_hv if generation_source == "archive" else (
            float(moocore.hypervolume(dominating, ref=hv_ref)) if len(dominating) else 0.0
        )

        hv_values.append(hv_val)
        hv_cumulative.append(cum_hv)
        distance_points = (
            moo_metrics.normalize(pts_min, distance_ideal, distance_scale) if normalize else pts_min
        )
        if gd_scale is not None:
            # The sampled reference follows the request order, but the analytical
            # function is defined in benchmark order. Undo the permutation.
            benchmark_points = pts_min[:, np.argsort(objective_columns)]
            gd_values.append(
                moo_metrics.gd_analytical(
                    benchmarks.front_distance(bench, benchmark_points, n_obj), scale=gd_scale
                )
            )
        else:
            gd_values.append(
                moo_metrics.gd(distance_points, distance_reference, normalized=False)
            )
        igd_values.append(
            moo_metrics.igd(
                distance_points, distance_reference, normalized=False
            )
        )
        igd_plus_values.append(
            moo_metrics.igd_plus(
                distance_points, distance_reference, normalized=False
            )
        )

    return {
        "generations": generations_sorted,
        "hv": hv_values,
        "hv_cumulative": hv_cumulative if include_cumulative else [],
        "gd": gd_values,
        "igd": igd_values,
        "igd_plus": igd_plus_values,
        "reference": reference_kind,
        "reference_size": int(len(reference_front)),
        "normalized": bool(normalize),
        "worst_point": dict(zip(objectives, hv_ref)),
        "population": population,
        "population_source": population_source,
        "population_sources": population_sources,
        # What the numbers mean, so a plot can label itself and two runs can be
        # compared knowingly. "analytical" GD is the exact distance to the true
        # front; "reference_front" is the mean nearest-neighbour distance to the
        # reference, which carries that sample's discretisation floor.
        "gd_method": gd_method,
        "gd_formula": "mean of each front point's distance to the true front (p=1)",
        "normalization": "analytical ideal-nadir range" if analytical_bounds is not None
                         else "reference front ideal-nadir range",
    }
