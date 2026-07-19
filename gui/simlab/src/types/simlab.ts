/* -------------------------------------------------------
 * Penalty (infeasible individuals)
 * Objectives above this threshold are penalty values injected
 * by the engine when a chromosome violates a hard constraint
 * (e.g. trajectory coverage in P2).  These individuals must
 * never appear on the Pareto front and should be visually
 * distinguished from feasible ones in the UI.
 * ----------------------------------------------------- */

export const PENALTY_THRESHOLD = 1e8;

/** Returns true when at least one objective is a penalty value. */
export function isPenalized(objectives: number[]): boolean {
  return objectives.some((v) => v >= PENALTY_THRESHOLD);
}

/* -------------------------------------------------------
 * Primitivos
 * ----------------------------------------------------- */

export type ID = string;
export type ISODateTime = string;
export type FloatMap = Record<string, number>;
export type JsonValue = string | number | boolean | null | { [k: string]: JsonValue } | JsonValue[];
export type JsonObject = Record<string, JsonValue>;

/* -------------------------------------------------------
 * Status
 * ----------------------------------------------------- */

export type ExperimentStatus = "Waiting" | "Running" | "Done" | "Error";
export type GenerationStatus = "Waiting" | "Running" | "Done" | "Error";
export type SimulationStatus = "Waiting" | "Running" | "Done" | "Error";

/* -------------------------------------------------------
 * Parameters (alinhado com pylib/db/models/experiment.py)
 * ----------------------------------------------------- */

export interface ObjectiveItem {
  metric_name: string;
  goal: "min" | "max" | string;
}

export interface MetricItem {
  name: string;
  kind: string;
  column: string;
  q?: number;
  scale?: number;
}

export interface DataConversionConfigDto {
  node_col: string;
  time_col: string;
  metrics: MetricItem[];
}

/**
 * Algorithm parameters as built by the launch wizards. All optional because
 * strategies differ (random_search has no evolutionary operators, only NSGA-III
 * has divisions) and problems consume different keys (see
 * lib/problemCapabilities.ts); the index signature tolerates extras.
 */
export interface AlgorithmParamsDto {
  population_size?: number;
  number_of_generations?: number;
  random_seed?: number;
  divisions?: number;
  prob_cx?: number;
  prob_mt?: number;
  per_gene_prob?: number;
  crossover_method?: string;       // P1 only
  eta_cx?: number;                 // P0/P1: SBX distribution index
  eta_mt?: number;                 // P0/P1: polynomial mutation index
  pm_tau?: number;                 // P4: tau mutation probability
  sigma_tau?: number;              // P4: tau mutation std. dev.
  apply_coverage_repair?: boolean; // P1/P2
  repair_coverage_budget?: number; // P1/P2
  /** @deprecated the engine ignores these; kept for reading old experiments */
  selection_method?: string;
  /** @deprecated the engine ignores these; kept for reading old experiments */
  mutation_method?: string;
  // `unknown` (not JsonValue): the recursive JsonValue union inside an index
  // signature blows up Vue's deep ref-unwrapping type instantiation (TS2589).
  [k: string]: unknown;
}

export interface SyntheticConfigDto {
  enabled: boolean;
  bench?: string;
  noise_std?: number;
}

export interface SimulationParamsDto {
  duration?: number;
  random_seeds?: number[];
  random_seeds_count?: number;
  synthetic?: SyntheticConfigDto;
  // See AlgorithmParamsDto for why this is `unknown`.
  [k: string]: unknown;
}

export interface ParametersDto {
  strategy: string;
  algorithm: AlgorithmParamsDto;
  simulation: SimulationParamsDto;
  problem: JsonObject;
  objectives: ObjectiveItem[];
}

/* -------------------------------------------------------
 * Pareto front
 * ----------------------------------------------------- */

export interface ParetoFrontItemDto {
  chromosome: JsonObject;
  objectives: FloatMap; // { metric_name: value }
}

/* -------------------------------------------------------
 * Runtime (computational) telemetry
 * Summary comes embedded in ExperimentDto; the full series
 * are fetched on demand via GET /experiments/{id}/runtime-metrics.
 * ----------------------------------------------------- */

export type RuntimeMetricsStatus =
  | "collecting"
  | "completed"
  | "no_data"
  | "failed";

export interface RuntimeMetricsSummaryDto {
  duration_seconds?: number;
  cpu?: { average_percent: number; maximum_percent: number };
  memory?: { average_bytes: number; maximum_bytes: number };
}

export interface RuntimeMetricsArtifactDto {
  storage: string;
  file_id: ID;
  filename: string;
  content_type: string;
  compression: string;
  size_bytes: number;
  sha256: string;
  schema_version: number;
}

export interface RuntimeMetricsDto {
  status: RuntimeMetricsStatus;
  started_at?: ISODateTime | null;
  finished_at?: ISODateTime | null;
  collection_finished_at?: ISODateTime | null;
  summary?: RuntimeMetricsSummaryDto;
  artifact?: RuntimeMetricsArtifactDto;
  error?: string;
}

export interface RuntimeMetricsSeriesDto {
  metric: string; // "cpu_percent" | "memory_bytes" | future metrics
  scope: string; // "aggregate" | "container"
  unit: string;
  labels: Record<string, string>;
  name: string; // container name, or the scope for aggregates
  points: [number, number][]; // [epoch seconds, value]
}

export interface RuntimeMetricsSeriesResponseDto {
  status: RuntimeMetricsStatus;
  started_at?: ISODateTime | null;
  finished_at?: ISODateTime | null;
  summary: RuntimeMetricsSummaryDto;
  series: RuntimeMetricsSeriesDto[];
  downsampled: boolean;
  total_samples: number;
}

/* -------------------------------------------------------
 * Individual (dentro de uma geração)
 * ----------------------------------------------------- */

export interface IndividualDto {
  id: ID;
  individual_id: string;       // hash do cromossomo
  chromosome: JsonObject;
  objectives: number[];        // indexed; usar parameters.objectives[i].metric_name para nome
  topology_picture_id: ID | null;
  simulations_ids?: ID[];
}

/* -------------------------------------------------------
 * Generation
 * ----------------------------------------------------- */

export interface GenerationDto {
  id: ID;
  experiment_id: ID;
  index: number;
  status: GenerationStatus;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;
  population: IndividualDto[];
}

/* -------------------------------------------------------
 * Experiment (base e info)
 * ----------------------------------------------------- */

export interface ExperimentInfoDto {
  id: ID;
  name: string;
  system_message?: string | null;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;
  status: ExperimentStatus; // adicionado client-side (vem do endpoint by-status)
  is_synthetic?: boolean;
  synthetic_bench?: string | null;
}

export interface ExperimentDto {
  id: ID;
  name: string;
  status: ExperimentStatus;
  system_message?: string | null;
  created_time?: ISODateTime | null;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;
  parameters: ParametersDto;
  source_repository_options: Record<string, ID>;
  data_conversion_config: DataConversionConfigDto;
  pareto_front?: ParetoFrontItemDto[] | null;
  analysis_files?: Record<string, ID>;
  runtime_metrics?: RuntimeMetricsDto | null;
}

export interface ExperimentFullDto extends ExperimentDto {
  generations: GenerationDto[];
}

/* -------------------------------------------------------
 * Simulation
 * ----------------------------------------------------- */

export interface SimulationDto {
  id: ID;
  experiment_id: ID;
  generation_id: ID;
  individual_id: string;
  status: SimulationStatus;
  system_message?: string | null;
  random_seed: number;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;
  parameters: JsonObject;
  pos_file_id?: ID | null;
  csc_file_id?: ID | null;
  source_repository_id?: ID | null;
  log_cooja_id?: ID | null;
  runtime_log_id?: ID | null;
  csv_log_id?: ID | null;
  network_metrics: FloatMap;
}

/* -------------------------------------------------------
 * Campaign
 * ----------------------------------------------------- */

export interface CampaignInfoDto {
  id: ID;
  name: string;
  description: string;
  created_time?: ISODateTime | null;
  experiment_count: number;
}

export interface CampaignDto {
  id: ID;
  name: string;
  description: string;
  created_time?: ISODateTime | null;
  experiment_ids: ID[];
}

export interface CampaignFullDto {
  id: ID;
  name: string;
  description: string;
  created_time?: ISODateTime | null;
  experiments: ExperimentDto[];
}

/* -------------------------------------------------------
 * Experiment creation
 * The POST /experiments/ endpoint expects the full experiment
 * shape with id: null and server-controlled fields pre-set.
 * ----------------------------------------------------- */

export interface ExperimentCreateDto {
  id: null;
  name: string;
  status: "Waiting";
  system_message: null;
  created_time: ISODateTime | null;
  start_time: null;
  end_time: null;
  generations: [];
  parameters: ParametersDto;
  source_repository_options: Record<string, ID>;
  data_conversion_config: DataConversionConfigDto;
  pareto_front: null;
}

/* -------------------------------------------------------
 * Source repository
 * ----------------------------------------------------- */

export interface SourceFileDto {
  id: ID;
  file_name: string;
}

export interface SourceRepositoryDto {
  id: ID;
  name: string;
  description: string;
  source_files: SourceFileDto[];
}
