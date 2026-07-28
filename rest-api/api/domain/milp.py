from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- model catalog

class MilpParamSpecDto(BaseModel):
    name: str
    description: str
    default: float
    sweepable: bool


class MilpModelInfoDto(BaseModel):
    key: str
    problem_key: str
    title: str
    description: str


class MilpModelDto(MilpModelInfoDto):
    parameters: list[MilpParamSpecDto]
    solver_defaults: dict[str, float]
    formulation: str


# --------------------------------------------------------------------- sweeps

class MilpSolverConfigDto(BaseModel):
    backend: str = "gurobi"
    time_limit_s: float = 300.0
    mip_gap: float = 0.01
    # When the requested backend is unavailable (e.g. no Gurobi license),
    # let the engine fall back to another available backend instead of failing.
    allow_fallback: bool = True


class MilpSweepCreateDto(BaseModel):
    name: str
    model_key: str
    # Exported problem JSON — same shape the GUI sends in experiment
    # parameters.problem (problem drafts are a GUI-internal format).
    problem: dict[str, Any]
    problem_id: Optional[str] = None  # provenance: problems collection draft
    parameter_grid: dict[str, list[float]]
    fixed_parameters: dict[str, float] = Field(default_factory=dict)
    solver: MilpSolverConfigDto = Field(default_factory=MilpSolverConfigDto)
    # Passed through to the generated batch experiment (simulation config,
    # objectives, source_repository_options, data_conversion_config, ...).
    batch_options: dict[str, Any] = Field(default_factory=dict)
    campaign_id: Optional[str] = None


class MilpSweepProgressDto(BaseModel):
    total_combos: int = 0
    done: int = 0
    solved: int = 0
    infeasible: int = 0
    unique_genotypes: int = 0


class MilpSweepInfoDto(BaseModel):
    id: str
    name: str
    model_key: str
    status: Optional[str] = None
    system_message: Optional[str] = None
    created_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress: MilpSweepProgressDto = Field(default_factory=MilpSweepProgressDto)
    experiment_id: Optional[str] = None
    campaign_id: Optional[str] = None
    cancel_requested: bool = False


class MilpSweepDto(MilpSweepInfoDto):
    problem: dict[str, Any] = Field(default_factory=dict)
    problem_id: Optional[str] = None
    parameter_grid: dict[str, list[float]] = Field(default_factory=dict)
    fixed_parameters: dict[str, float] = Field(default_factory=dict)
    solver: MilpSolverConfigDto = Field(default_factory=MilpSolverConfigDto)
    batch_options: dict[str, Any] = Field(default_factory=dict)
    # One entry per solved grid combination (params, status, genotype, ...)
    solutions: list[dict[str, Any]] = Field(default_factory=list)


class MilpEngineStatusDto(BaseModel):
    status: str = "unknown"  # online | offline | unknown
    solver: Optional[str] = None
    gurobi_license: Optional[str] = None
    available_backends: list[str] = Field(default_factory=list)
    updated_time: Optional[str] = None
