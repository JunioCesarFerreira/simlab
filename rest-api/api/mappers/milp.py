from datetime import datetime
from typing import Any, Optional

from api.domain.milp import (
    MilpEngineStatusDto,
    MilpModelDto,
    MilpModelInfoDto,
    MilpParamSpecDto,
    MilpSolverConfigDto,
    MilpSweepDto,
    MilpSweepInfoDto,
    MilpSweepProgressDto,
)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def milp_model_info_from_spec(spec: dict[str, Any]) -> MilpModelInfoDto:
    return MilpModelInfoDto(
        key=spec["key"],
        problem_key=spec["problem_key"],
        title=spec["title"],
        description=spec["description"],
    )


def milp_model_from_spec(spec: dict[str, Any]) -> MilpModelDto:
    return MilpModelDto(
        key=spec["key"],
        problem_key=spec["problem_key"],
        title=spec["title"],
        description=spec["description"],
        parameters=[MilpParamSpecDto(**p) for p in spec["parameters"]],
        solver_defaults=spec["solver_defaults"],
        formulation=spec["formulation"],
    )


def _sweep_common(doc: dict[str, Any]) -> dict[str, Any]:
    progress = doc.get("progress") or {}
    return {
        "id": str(doc["_id"]),
        "name": doc.get("name", ""),
        "model_key": doc.get("model_key", ""),
        "status": doc.get("status"),
        "system_message": doc.get("system_message"),
        "created_time": _iso(doc.get("created_time")),
        "start_time": _iso(doc.get("start_time")),
        "end_time": _iso(doc.get("end_time")),
        "progress": MilpSweepProgressDto(**progress),
        "experiment_id": str(doc["experiment_id"]) if doc.get("experiment_id") else None,
        "campaign_id": str(doc["campaign_id"]) if doc.get("campaign_id") else None,
        "cancel_requested": bool(doc.get("cancel_requested", False)),
    }


def milp_sweep_info_from_mongo(doc: dict[str, Any]) -> MilpSweepInfoDto:
    return MilpSweepInfoDto(**_sweep_common(doc))


def milp_sweep_from_mongo(doc: dict[str, Any]) -> MilpSweepDto:
    return MilpSweepDto(
        **_sweep_common(doc),
        problem=doc.get("problem") or {},
        problem_id=str(doc["problem_id"]) if doc.get("problem_id") else None,
        parameter_grid=doc.get("parameter_grid") or {},
        fixed_parameters=doc.get("fixed_parameters") or {},
        solver=MilpSolverConfigDto(**(doc.get("solver") or {})),
        batch_options=doc.get("batch_options") or {},
        solutions=doc.get("solutions") or [],
    )


def milp_engine_status_from_mongo(doc: Optional[dict[str, Any]]) -> MilpEngineStatusDto:
    if not doc:
        return MilpEngineStatusDto(status="unknown")
    return MilpEngineStatusDto(
        status=doc.get("status", "unknown"),
        solver=doc.get("solver"),
        gurobi_license=doc.get("gurobi_license"),
        available_backends=doc.get("available_backends", []),
        updated_time=_iso(doc.get("updated_time")),
    )
