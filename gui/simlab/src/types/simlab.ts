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

export interface ParametersDto {
  strategy: string;
  algorithm: JsonObject;
  simulation: JsonObject;
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
