import type { ISODateTime } from "./simlab";

/**
 * Scalability benchmarks, as `GET /scalability` publishes them.
 *
 * A benchmark is an experiment run by the engine's `usl_benchmark` strategy:
 * one simulation, in waves of `N` identical copies, for several `N`. The
 * analysis is computed on demand from the simulations, so it reflects what
 * has run so far — a running benchmark already has a curve.
 */

/** The two throughput definitions the API fits. */
export type UslMetric = "batch" | "per_simulation";

export type FitStatus = "ok" | "amdahl_only" | "insufficient_data";

export interface BenchmarkWorkload {
  genome_hash?: string | null;
  source?: "generated" | "provided" | null;
  seed?: number | null;
}

export interface BenchmarkSummary {
  id: string;
  name: string;
  status: string;
  system_message: string;
  created_time?: ISODateTime | null;
  start_time?: ISODateTime | null;
  end_time?: ISODateTime | null;
  problem?: string | null;
  duration?: number | null;
  workload?: BenchmarkWorkload | null;
  levels: number[];
  repetitions: number;
  waves: number;
  simulations: number;
  waves_finished?: number;
}

export interface WaveMeasurement {
  generation_id: string;
  wave: number | null;
  level: number;
  repetition: number | null;
  status: string;
  simulations: number;
  done: number;
  failed: number;
  /** Simulations per host in this wave — why a level can vary between
   *  repetitions on a pool whose hosts differ in speed. */
  hosts: Record<string, number>;
  makespan_sec: number | null;
  mean_held_sec: number | null;
  throughput: Record<UslMetric, number | null>;
  valid: boolean;
  reason: string;
}

export interface LevelSummary {
  level: number;
  mean: number;
  std: number;
  count: number;
  min: number;
  max: number;
}

export interface CurvePoint {
  n: number;
  x: number;
  /** λN — linear scaling, for reference. */
  ideal?: number;
}

export interface UslFit {
  status: FitStatus;
  lam: number;
  sigma: number;
  kappa: number;
  r2: number | null;
  rmse: number | null;
  n_points: number;
  n_levels: number;
  n_peak: number | null;
  n_peak_int: number | null;
  x_peak: number | null;
  x_asymptote: number | null;
  peak_within_tested: boolean;
  levels: LevelSummary[];
  curve: CurvePoint[];
}

/**
 * One physical machine's own curve, from the same waves.
 *
 * In a wave every worker runs one simulation and they start together, so a
 * machine holding `k` of them ran `k` at once; its throughput at `k` is
 * `k / mean held` over its own runs. That curve is about one set of CPUs —
 * what the USL describes — unlike the pool-wide one on a mixed pool.
 */
export interface MachineFit {
  machine: string;
  hosts: string[];
  /** The most simulations it ran at once in any measured wave. */
  max_tested: number;
  fit: Omit<UslFit, "levels" | "curve" | "peak_within_tested">;
  levels: LevelSummary[];
  curve: CurvePoint[];
  peak_within_tested: boolean;
  /** The peak, when the measurements reach it — never an extrapolation. */
  recommended_limit: number | null;
  best_measured: { level: number; throughput: number } | null;
  /** The limit set on it now, if any. */
  limit: number | null;
  /** One simulation's throughput, sim/min. */
  unit_throughput: number;
  /** That, relative to the fastest machine (1 = fastest). */
  relative_speed: number | null;
}

export interface BenchmarkAnalysis {
  unit: string;
  metrics: UslMetric[];
  default_metric: UslMetric;
  planned_levels: number[];
  waves: WaveMeasurement[];
  valid_waves: number;
  fits: Record<UslMetric, UslFit>;
  machines?: MachineFit[];
  /** Machines differ in speed enough that the pool-wide curve mixes speed with scaling. */
  heterogeneous_pool?: boolean;
}

export interface BenchmarkDetail extends BenchmarkSummary {
  analysis: BenchmarkAnalysis;
}

export interface BenchmarkCreateInput {
  name: string;
  /** The exported problem configuration (`exportProblem(draft).problem`). */
  problem: Record<string, unknown>;
  source_repository_options: Record<string, string>;
  levels: number[];
  repetitions: number;
  duration: number;
  seed?: number;
  random_seed?: number;
  chromosome?: Record<string, unknown> | null;
  data_conversion_config?: Record<string, unknown>;
}

export interface BenchmarkCreated {
  id: string;
  levels: number[];
  repetitions: number;
  waves: number;
  simulations: number;
}
