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
 * Primitives
 * ----------------------------------------------------- */

export type ID = string;
export type ISODateTime = string;
export type FloatMap = Record<string, number>;
export type JsonValue = string | number | boolean | null | { [k: string]: JsonValue } | JsonValue[];
export type JsonObject = Record<string, JsonValue>;

/* -------------------------------------------------------
 * Status
 * ----------------------------------------------------- */

export type ExperimentStatus = "Waiting" | "Running" | "Done" | "Error" | "Cancelled";
export type GenerationStatus = "Waiting" | "Running" | "Done" | "Error";
export type SimulationStatus = "Waiting" | "Running" | "Done" | "Error";

/* -------------------------------------------------------
 * Parameters (aligned with pylib/db/models/experiment.py)
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
  crossover_method?: string; // P1 only
  eta_cx?: number; // P0/P1: SBX distribution index
  eta_mt?: number; // P0/P1: polynomial mutation index
  pm_tau?: number; // P4: tau mutation probability
  sigma_tau?: number; // P4: tau mutation std. dev.
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
  | "failed"
  | "unavailable";

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
 * Firmware snapshot (traceability)
 * Captured when the experiment starts: an immutable copy of
 * every source file, owned by the experiment alone.
 * ----------------------------------------------------- */

export type FirmwareSnapshotStatus = "captured" | "partial" | "skipped" | "failed" | "capturing";

export interface FirmwareFileDto {
  file_name: string;
  /** GridFS id of the copy owned by the experiment. Absent for missing files. */
  file_id?: ID;
  /** GridFS id in the shared source repository. */
  origin_file_id?: ID;
  size_bytes?: number;
  sha256?: string;
}

export interface FirmwareRepositorySnapshotDto {
  option_keys: string[];
  source_repository_id: ID;
  name: string;
  description: string;
  files: FirmwareFileDto[];
  missing_files: FirmwareFileDto[];
}

export interface FirmwareSnapshotDto {
  status: FirmwareSnapshotStatus;
  captured_at?: ISODateTime | null;
  schema_version?: number;
  repositories: FirmwareRepositorySnapshotDto[];
  /** Why nothing was captured (skipped). */
  reason?: string;
  /** Why the capture failed. */
  error?: string;
}

/* -------------------------------------------------------
 * Individual (within a generation)
 * ----------------------------------------------------- */

/**
 * What an individual's objectives rest on.
 *
 * Published because the objectives alone cannot say: a mean over one replica
 * and a mean over five look identical. `state` is the short answer and the
 * counts are the evidence; `objective_used` says whether the experiment's
 * configured policy let this evaluation reach the algorithm.
 */
/**
 * One Cooja container's share of an experiment's simulations.
 *
 * `share` is a percentage of the **placed** simulations, not of every
 * simulation: a run that never reached a container belongs to no worker, and
 * those are counted in `unplaced` instead.
 */
export interface PlacementWorkerDto {
  resource_id?: string | null;
  resource_name: string;
  total: number;
  done: number;
  failed: number;
  active: number;
  share: number;
  /**
   * Mean `end_time - start_time` of the runs that finished Done, in seconds;
   * `null` when none did. Failed runs stop at arbitrary points and are left out.
   */
  mean_duration_sec?: number | null;
}

export interface PlacementHostDto {
  host_id?: string | null;
  host_name: string;
  total: number;
  done: number;
  failed: number;
  active: number;
  share: number;
  /** Weighted by run across the host's workers, not a mean of their means. */
  mean_duration_sec?: number | null;
  workers: PlacementWorkerDto[];
}

/**
 * Where an experiment's simulations actually ran.
 *
 * Counted per simulation, so a replica that was repeated counts on both
 * containers that ran it — the question is how much work each machine did,
 * not how many distinct replicas an evaluation rests on.
 */
export interface PlacementDistributionDto {
  total: number;
  placed: number;
  unplaced: number;
  hosts: PlacementHostDto[];
}

export interface CompletenessDto {
  /**
   * `analytical` and `infeasible` exist so that `unknown` keeps meaning one
   * thing — *the plan was not recorded*. Neither of those had replicas to
   * count in the first place.
   */
  state: "complete" | "partial" | "pending" | "invalid" | "unknown" | "analytical" | "infeasible";
  expected?: number;
  observed?: number;
  failed?: number;
  pending?: number;
  /** Planned seeds with no simulation at all — nothing to repeat. */
  absent?: number;
  expected_seeds?: number[];
  observed_seeds?: number[];
  missing_seeds?: number[];
  failed_seeds?: number[];
  /**
   * The holes no repeat can fill: `absent_seeds` were never created, so there
   * is no run to clone, and `thin_metrics` are missing from a replica that did
   * finish. Both are decided by the policy on the first pass.
   */
  absent_seeds?: number[];
  thin_metrics?: string[];
  /** Observations per metric: two objectives can rest on different counts. */
  per_metric?: Record<string, number>;
  attempts?: number;
  objective_used?: boolean;
  policy?: string;
  /**
   * Carried over from an earlier evaluation of the same genome, with that
   * evaluation's decision intact. A reused individual has no simulations in
   * its own generation to count.
   */
  reused?: boolean;
  source?: string;
  /**
   * Why the state is what it is, in one phrase — published for every state.
   * `partial` cannot be read without it: a generation whose every planned seed
   * finished can be partial because one result came back without a metric.
   */
  reason?: string;
}

export interface IndividualDto {
  id: ID;
  individual_id: string; // chromosome hash
  chromosome: JsonObject;
  objectives: number[]; // indexed; use parameters.objectives[i].metric_name for the name
  topology_picture_id: ID | null;
  simulations_ids?: ID[];
  /**
   * `null` when the evaluation predates the record, **not** when it is
   * complete. The two are different states and the panel says which.
   */
  completeness?: CompletenessDto | null;
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
 * Experiment (base and info)
 * ----------------------------------------------------- */

export interface ExperimentInfoDto {
  id: ID;
  name: string;
  system_message?: string | null;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;
  status: ExperimentStatus; // added client-side (comes from the by-status endpoint)
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
  firmware_snapshot?: FirmwareSnapshotDto | null;
}

export interface ExperimentFullDto extends ExperimentDto {
  generations: GenerationDto[];
}

/* -------------------------------------------------------
 * Simulation
 * ----------------------------------------------------- */

/** RPL DODAG tree reconstructed from the simulation log. */
export interface DodagTree {
  root: string | null;
  /** child address -> preferred-parent address (final edge). */
  edges: Record<string, string>;
  /** node address -> hop depth from the root. */
  depth: Record<string, number>;
  /** nodes that never reached the root. */
  disconnected: string[];
}

/** DODAG stable-convergence summary (from firmware parent-switch events). */
export interface DodagConvergence {
  converged: boolean;
  convergence_time_ms: number | null;
  n_nodes: number;
  expected_nodes?: number | null;
  total_parent_switches: number;
  parent_switches_per_node: Record<string, number>;
  uses_global_clock: boolean;
  stability_verified: boolean;
}

/** Approximate, first-contact based DODAG formation (works on legacy logs). */
export interface DodagApprox {
  n_nodes_joined: number;
  expected_nodes?: number | null;
  all_joined: boolean;
  approx_first_join_ms: number | null;
  approx_last_join_ms: number | null;
  approx_formation_time_ms: number | null;
  approx_max_depth: number | null;
  approx_mean_depth: number | null;
  per_node: Record<string, { approx_join_ms: number; approx_depth: number | null }>;
}

export interface DodagInfo {
  deployed_nodes: number;
  approx: DodagApprox;
  tree: DodagTree;
  convergence: DodagConvergence;
}

/**
 * Where one simulation ran.
 *
 * Recorded by the master-node when it reserves the container, before anything
 * starts, so a finished simulation still says where it executed even after
 * the host registration is edited or removed — which is why the names travel
 * alongside the ids.
 *
 * `null` on a simulation that was never placed, and on those recorded before
 * the master-node began writing this down: that is different from one that
 * ran somewhere unknown.
 */
export interface PlacementDto {
  host_id?: ID | null;
  host_name: string;
  resource_id?: ID | null;
  resource_name: string;
  /** Docker's id, when the worker was discovered rather than configured. */
  container_id?: string | null;
  /** The placement attempt; two attempts ran in different conditions. */
  attempt_id?: string | null;
  assigned_at?: ISODateTime | null;
  remote_pid?: number | null;
}

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
  dodag?: DodagInfo | null;
  placement?: PlacementDto | null;
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
