import json
import os
import subprocess

import numpy as np
import moocore

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from bson import errors as bson_errors
from tempfile import NamedTemporaryFile
from pydantic import BaseModel

from bson import ObjectId

from pylib.db import MongoRepository
from api.dependencies import get_factory
from api.domain.experiment import ExperimentDto, ExperimentFullDto, ExperimentInfoDto
from api.mappers.experiment import (
    experiment_from_mongo,
    experiment_full_from_mongo,
    experiment_info_from_mongo,
    experiment_to_mongo,
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
        generations = [
            generation_from_mongo(
                g,
                factory.individual_repo.find_by_generation(g["_id"]),
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
    try:
        factory.experiment_repo.update_status(experiment_id, new_status)
        return True
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


class ParetoPlotRequest(BaseModel):
    objectives: list[str]
    minimize: list[bool]


_PARETO_SCRIPT = os.getenv(
    "SIMLAB_PARETO_SCRIPT",
    "/home/junio/github/simlab/pareto-analysis/plot_pareto_results.py",
)
_PARETO_PYTHON = os.getenv(
    "SIMLAB_PARETO_PYTHON",
    "/home/junio/github/simlab/mo-engine/.venv/bin/python",
)


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

    api_key = os.getenv("SIMLAB_API_KEY", "api-password")
    minimize_strs = [str(m) for m in body.minimize[:3]]

    cmd = [
        _PARETO_PYTHON, _PARETO_SCRIPT,
        "--expid", experiment_id,
        "--objectives", *body.objectives[:3],
        "--minimize", *minimize_strs,
        "--api-base", "http://localhost:8000/api/v1",
        "--api-key", api_key,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(_PARETO_SCRIPT),
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

_PENALTY_THRESHOLD = 1.0e8


def _is_penalized(objs: list[float]) -> bool:
    return any(abs(v) >= _PENALTY_THRESHOLD for v in objs)


def _dominates(a: list[float], b: list[float], minimize: list[bool]) -> bool:
    """True if a dominates b (at least as good everywhere, strictly better somewhere)."""
    better = False
    for ai, bi, m in zip(a, b, minimize):
        if (m and ai > bi) or (not m and ai < bi):
            return False
        if (m and ai < bi) or (not m and ai > bi):
            better = True
    return better


def _pareto_front(objs_list: list[list[float]], minimize: list[bool]) -> list[int]:
    """Returns indices of the non-dominated (rank-0) individuals."""
    n = len(objs_list)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if _dominates(objs_list[j], objs_list[i], minimize):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


@router.get("/{experiment_id}/hv-gd")
def get_hv_gd(
    experiment_id: str,
    objectives: list[str] = Query(...),
    minimize: list[str] = Query(...),
    factory: MongoRepository = Depends(get_factory),
) -> dict:
    """Compute hypervolume and generational distance per generation."""
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

    stored_pf: list[dict] = doc.get("pareto_front") or []
    if not stored_pf:
        return {"generations": [], "hv": [], "gd": [], "worst_point": {}}

    minimize_bools = [m.lower() == "true" for m in minimize]

    # ── Fetch generations + individuals directly from DB ─────────────────────
    gens = factory.generation_repo.find_by_experiment(exp_oid)

    individuals_per_gen: dict[int, list[list[float]]] = {}
    for gen in gens:
        gen_idx: int = gen["index"]
        individuals = factory.individual_repo.find_by_generation(gen["_id"])
        valid: list[list[float]] = []
        for ind in individuals:
            raw = ind.get("objectives") or []
            if len(raw) < n_obj:
                continue
            objs = [float(raw[i]) for i in range(n_obj)]
            if _is_penalized(objs):
                continue
            valid.append(objs)
        individuals_per_gen[gen_idx] = valid

    if not individuals_per_gen:
        return {"generations": [], "hv": [], "gd": [], "worst_point": {}}

    # ── Worst feasible point → HV reference (original space) ─────────────────
    all_objs = [o for v in individuals_per_gen.values() for o in v]
    worst = [
        max(o[i] for o in all_objs) if minimize_bools[i]
        else min(o[i] for o in all_objs)
        for i in range(n_obj)
    ]
    worst_ref = [v + abs(v) * 0.05 + 1.0 for v in worst]

    # ── GD reference: stored Pareto front in minimization space ──────────────
    seen_ref: set[tuple] = set()
    ref_min_rows: list[list[float]] = []
    for p in stored_pf:
        objs_dict: dict = p.get("objectives") or {}
        row_min = tuple(
            float(objs_dict.get(o, 0.0)) if minimize_bools[i]
            else -float(objs_dict.get(o, 0.0))
            for i, o in enumerate(objectives)
        )
        if row_min not in seen_ref:
            seen_ref.add(row_min)
            ref_min_rows.append(list(row_min))
    final_front = np.array(ref_min_rows, dtype=float)

    # ── Per-generation HV / GD ────────────────────────────────────────────────
    generations_sorted = sorted(individuals_per_gen.keys())
    hv_values: list[float] = []
    gd_values: list[float | None] = []

    for gen_idx in generations_sorted:
        pop_objs = individuals_per_gen[gen_idx]
        if not pop_objs:
            hv_values.append(0.0)
            gd_values.append(None)
            continue

        front_idx = _pareto_front(pop_objs, minimize_bools)
        if not front_idx:
            hv_values.append(0.0)
            gd_values.append(None)
            continue

        # Minimization space + dedup by objective tuple
        seen_pts: set[tuple] = set()
        pts_min_rows: list[list[float]] = []
        for i in front_idx:
            key = tuple(
                pop_objs[i][j] if minimize_bools[j] else -pop_objs[i][j]
                for j in range(n_obj)
            )
            if key not in seen_pts:
                seen_pts.add(key)
                pts_min_rows.append(list(key))
        pts_min = np.array(pts_min_rows, dtype=float)

        hv_val = float(moocore.hypervolume(pts_min, ref=worst_ref))
        diffs = pts_min[:, np.newaxis, :] - final_front[np.newaxis, :, :]
        gd_val = float(np.sqrt((diffs ** 2).sum(axis=2)).min(axis=1).mean())

        hv_values.append(hv_val)
        gd_values.append(gd_val)

    return {
        "generations": generations_sorted,
        "hv": hv_values,
        "gd": gd_values,
        "worst_point": dict(zip(objectives, worst_ref)),
    }
