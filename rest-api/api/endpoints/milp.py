import math
import os
from datetime import datetime

from bson import ObjectId, errors as bson_errors
from fastapi import APIRouter, Depends, HTTPException

from pylib.config.milp_models import MILP_ALLOWED_BACKENDS, MILP_MODEL_SPECS
from pylib.db import EnumStatus, MongoRepository

from api.dependencies import get_factory
from api.domain.milp import (
    MilpEngineStatusDto,
    MilpModelDto,
    MilpModelInfoDto,
    MilpSweepCreateDto,
    MilpSweepDto,
    MilpSweepInfoDto,
)
from api.mappers.milp import (
    milp_engine_status_from_mongo,
    milp_model_from_spec,
    milp_model_info_from_spec,
    milp_sweep_from_mongo,
    milp_sweep_info_from_mongo,
)

router = APIRouter()

# Guardrail against accidental combinatorial explosions in the sweep grid
MAX_COMBINATIONS = int(os.getenv("MILP_MAX_COMBINATIONS", "10000"))


# ---------------------------------------------------------------- model catalog

@router.get("/models", response_model=list[MilpModelInfoDto])
def list_models() -> list[MilpModelInfoDto]:
    """List the available MILP models (static catalog)."""
    return [milp_model_info_from_spec(spec) for spec in MILP_MODEL_SPECS.values()]


@router.get("/models/{model_key}", response_model=MilpModelDto)
def get_model(model_key: str) -> MilpModelDto:
    """Full model detail: parameter schema, solver defaults and formulation."""
    spec = MILP_MODEL_SPECS.get(model_key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown MILP model '{model_key}'")
    return milp_model_from_spec(spec)


# --------------------------------------------------------------- engine status

@router.get("/status", response_model=MilpEngineStatusDto)
def get_engine_status(
    factory: MongoRepository = Depends(get_factory),
) -> MilpEngineStatusDto:
    """Solver/license status last published by the milp-engine."""
    try:
        doc = factory.milp_sweep_repo.get_engine_status()
        return milp_engine_status_from_mongo(doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------- sweeps

def _validate_sweep(dto: MilpSweepCreateDto) -> dict:
    """Validate a sweep request against the model catalog; returns the spec."""
    spec = MILP_MODEL_SPECS.get(dto.model_key)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown MILP model '{dto.model_key}'")

    param_specs = {p["name"]: p for p in spec["parameters"]}
    unknown = (set(dto.parameter_grid) | set(dto.fixed_parameters)) - set(param_specs)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown parameters for model '{dto.model_key}': {sorted(unknown)}",
        )
    overlap = set(dto.parameter_grid) & set(dto.fixed_parameters)
    if overlap:
        raise HTTPException(
            status_code=400, detail=f"Parameters both swept and fixed: {sorted(overlap)}"
        )

    total = 1
    for name, values in dto.parameter_grid.items():
        if not values:
            raise HTTPException(
                status_code=400, detail=f"Empty value list for swept parameter '{name}'"
            )
        if len(values) > 1 and not param_specs[name]["sweepable"]:
            raise HTTPException(
                status_code=400, detail=f"Parameter '{name}' is not sweepable"
            )
        if any(not math.isfinite(v) for v in values):
            raise HTTPException(
                status_code=400, detail=f"Non-finite value for parameter '{name}'"
            )
        total *= len(values)
    if total > MAX_COMBINATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Grid has {total} combinations; maximum is {MAX_COMBINATIONS}",
        )

    problem_name = dto.problem.get("name")
    if problem_name != spec["problem_key"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{dto.model_key}' expects a '{spec['problem_key']}' problem, "
                f"got '{problem_name}'"
            ),
        )

    if dto.solver.backend not in MILP_ALLOWED_BACKENDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown solver backend '{dto.solver.backend}'. "
                   f"Allowed: {list(MILP_ALLOWED_BACKENDS)}",
        )
    return spec


@router.post("/sweeps", response_model=str)
def create_sweep(
    dto: MilpSweepCreateDto,
    factory: MongoRepository = Depends(get_factory),
) -> str:
    """Create a MILP parameter sweep. The milp-engine picks it up via change
    stream and, once solved, creates the batch experiment. Returns the sweep id."""
    _validate_sweep(dto)

    total = 1
    for values in dto.parameter_grid.values():
        total *= len(values)

    try:
        problem_oid = ObjectId(dto.problem_id) if dto.problem_id else None
        campaign_oid = ObjectId(dto.campaign_id) if dto.campaign_id else None
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid problem_id or campaign_id")

    doc = {
        "name": dto.name,
        "model_key": dto.model_key,
        "problem": dto.problem,
        "problem_id": problem_oid,
        "parameter_grid": dto.parameter_grid,
        "fixed_parameters": dto.fixed_parameters,
        "solver": dto.solver.model_dump(),
        "batch_options": dto.batch_options,
        "status": EnumStatus.WAITING,
        "system_message": None,
        "cancel_requested": False,
        "created_time": datetime.now(),
        "start_time": None,
        "end_time": None,
        "progress": {
            "total_combos": total,
            "done": 0,
            "solved": 0,
            "infeasible": 0,
            "unique_genotypes": 0,
        },
        "checkpoint": {"next_index": 0, "genotypes": []},
        "solutions": [],
        "experiment_id": None,
        "campaign_id": campaign_oid,
    }
    try:
        return str(factory.milp_sweep_repo.insert(doc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sweeps", response_model=list[MilpSweepInfoDto])
def list_sweeps(
    factory: MongoRepository = Depends(get_factory),
) -> list[MilpSweepInfoDto]:
    """All sweeps with summary progress (solutions excluded)."""
    try:
        docs = factory.milp_sweep_repo.find_all()
        return [milp_sweep_info_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sweeps/{sweep_id}", response_model=MilpSweepDto)
def get_sweep(
    sweep_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> MilpSweepDto:
    """Full sweep detail, including per-combination solutions."""
    try:
        doc = factory.milp_sweep_repo.get(sweep_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Sweep not found")
        return milp_sweep_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sweeps/{sweep_id}/cancel", response_model=bool)
def cancel_sweep(
    sweep_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Request cooperative cancellation of a Waiting/Running sweep."""
    try:
        doc = factory.milp_sweep_repo.get(sweep_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Sweep not found")
        result = factory.milp_sweep_repo.request_cancel(sweep_id)
        if not result:
            raise HTTPException(
                status_code=409,
                detail=f"Sweep in status '{doc.get('status')}' cannot be cancelled",
            )
        return True
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sweeps/{sweep_id}", response_model=bool)
def delete_sweep(
    sweep_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Delete a sweep. Running sweeps must be cancelled first."""
    try:
        doc = factory.milp_sweep_repo.get(sweep_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Sweep not found")
        if doc.get("status") == EnumStatus.RUNNING:
            raise HTTPException(
                status_code=409, detail="Cannot delete a running sweep; cancel it first"
            )
        return factory.milp_sweep_repo.delete(sweep_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
