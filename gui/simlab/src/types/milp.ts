// MILP model catalog and parameter sweeps (mirrors rest-api/api/domain/milp.py)

export interface MilpParamSpecDto {
  name: string;
  description: string;
  default: number;
  sweepable: boolean;
}

export interface MilpModelInfoDto {
  key: string;
  problem_key: string;
  title: string;
  description: string;
}

export interface MilpModelDto extends MilpModelInfoDto {
  parameters: MilpParamSpecDto[];
  solver_defaults: Record<string, number>;
  formulation: string;
}

export interface MilpSolverConfigDto {
  backend: string;
  time_limit_s: number;
  mip_gap: number;
  allow_fallback: boolean;
}

export interface MilpSweepProgressDto {
  total_combos: number;
  done: number;
  solved: number;
  infeasible: number;
  unique_genotypes: number;
}

export interface MilpSweepInfoDto {
  id: string;
  name: string;
  model_key: string;
  status: string | null;
  system_message: string | null;
  created_time: string | null;
  start_time: string | null;
  end_time: string | null;
  progress: MilpSweepProgressDto;
  experiment_id: string | null;
  campaign_id: string | null;
  cancel_requested: boolean;
}

export interface MilpSolutionRecord {
  index: number;
  params: Record<string, number>;
  status: string;
  genotype: string | null;
  mask: number[] | null;
  n_installed: number | null;
  obj_value: number | null;
  mip_gap: number | null;
  runtime_s: number;
  is_duplicate: boolean;
  message: string;
}

export interface MilpSweepDto extends MilpSweepInfoDto {
  problem: Record<string, unknown>;
  problem_id: string | null;
  parameter_grid: Record<string, number[]>;
  fixed_parameters: Record<string, number>;
  solver: MilpSolverConfigDto;
  batch_options: Record<string, unknown>;
  solutions: MilpSolutionRecord[];
}

export interface MilpSweepCreatePayload {
  name: string;
  model_key: string;
  problem: Record<string, unknown>;
  problem_id?: string | null;
  parameter_grid: Record<string, number[]>;
  fixed_parameters: Record<string, number>;
  solver: MilpSolverConfigDto;
  batch_options: Record<string, unknown>;
  campaign_id?: string | null;
}

export interface MilpEngineStatusDto {
  status: string; // online | offline | unknown
  solver: string | null;
  gurobi_license: string | null;
  available_backends: string[];
  updated_time: string | null;
}
