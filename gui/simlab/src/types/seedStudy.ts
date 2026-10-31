/** What the seed study routes accept and return. Mirrors `pylib/seed_study.py`. */
import type { ObjectiveItem, SeedAggregatorSpec } from "./simlab";

export interface StudyTargets {
  confidence: number;
  relative_error: number;
  reliability: number;
}

export type StudySeeds =
  | { mode: "generated"; count: number; master_seed?: number }
  | { mode: "imported"; values: number[]; file_name?: string };

export type StudyPanel =
  | { source: "generated"; size: number; random_seed?: number }
  | { source: "provided"; random_seed?: number };

export interface StudyCreateInput {
  name: string;
  problem: Record<string, unknown>;
  source_repository_options: Record<string, string>;
  duration: number;
  seeds: StudySeeds;
  panel: StudyPanel;
  chromosomes?: Record<string, unknown>[] | null;
  objectives: ObjectiveItem[];
  targets: StudyTargets;
  control_replicates: number;
  data_conversion_config: Record<string, unknown>;
}

export interface SeedDiagnostics {
  low16_collisions: number;
  consecutive_pairs: number;
  guard: number;
}

export interface StudyPlan {
  seeds: number;
  chromosomes: number;
  control: number;
  simulations: number;
}

export interface StudyCreated extends StudyPlan {
  id: string;
  seed_diagnostics: SeedDiagnostics;
}

export interface StudySummary extends StudyPlan {
  id: string;
  name: string;
  status: string;
  system_message: string;
  created_time: string | null;
  start_time: string | null;
  end_time: string | null;
  problem: string | null;
  duration: number | null;
  panel_source: string | null;
  objectives: ObjectiveItem[];
  targets: StudyTargets;
  finished?: number;
}

export interface StudyWarning {
  code: string;
  metric: string | null;
  message: string;
}

export interface Precision {
  value: number | null;
  stein: number | null;
  empirical: number | null;
  method: "t" | "empirical" | "none";
  status: string;
  range?: [number | null, number | null];
}

export interface Description {
  n: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  sd: number | null;
  cv: number | null;
  median: number | null;
  q1: number | null;
  q3: number | null;
  iqr: number | null;
  p5: number | null;
  p95: number | null;
  skewness: number | null;
  excess_kurtosis: number | null;
}

export interface ChromosomeRow extends Description {
  genome_hash: string;
  ci_mean_t: [number, number] | null;
  shapiro_p: number | null;
  shapiro_p_holm: number | null;
  n_star: Precision | null;
}

export interface Spread {
  p50: number;
  p90: number;
  max: number;
  min: number;
}

export interface PanelAnalysis {
  used: {
    chromosomes: number;
    seeds: number;
    dropped_chromosomes: number[];
    dropped_seeds: number[];
  };
  components: { individual: number; seed: number; residual: number };
  shares: { individual: number; seed: number; residual: number };
  reliability: { m: number; e_rho2: number | null; phi: number | null }[];
  m_star: {
    value: number | null;
    ci: [number | null, number | null] | null;
    method: "generalizability" | "rank_stability";
  };
  m_rank: number | null;
  rank_stability: { k: number; spearman: number }[];
  signal: boolean;
}

export interface StudyMetric {
  name: string;
  status: string;
  aliases: string[];
  is_objective: boolean;
  goal: string | null;
  aggregator: SeedAggregatorSpec;
  observed: number;
  planned: number;
  all_zero: boolean;
  by_chromosome: ChromosomeRow[];
  panel: PanelAnalysis | null;
  summary: {
    n_star: Spread | null;
    empirical: Spread | null;
    cv: Spread | null;
    statuses: string[];
    normality_rejected: number | null;
  };
}

export interface ReportRow {
  value: number | null;
  p90: number | null;
  max: number | null;
  method: "t" | "empirical" | "none";
  status: string;
}

export interface OptimiseRow {
  value: number | null;
  ci: [number | null, number | null] | null;
  method: "generalizability" | "rank_stability" | null;
}

export interface StudyAnalysis {
  analysis_version: number;
  targets: StudyTargets;
  shape: { chromosomes: number; seeds: number };
  metrics: StudyMetric[];
  warnings: StudyWarning[];
  front_stability: { k: number; jaccard: number }[] | null;
  recommendation: {
    report: {
      by_objective: Record<string, ReportRow>;
      joint: { value: number; confidence: number; objectives: string[] } | null;
    };
    optimise: { by_objective: Record<string, OptimiseRow>; max: number | null };
  };
}

export interface StudyDetail extends StudySummary {
  seed_list: number[];
  panel: string[];
  seed_diagnostics: SeedDiagnostics;
  determinism: {
    checked: boolean;
    identical: boolean;
    replicates: number;
    compared: number;
    divergent: { genome_hash: string; metric: string; values: number[] }[];
    divergent_total: number;
  };
  completeness: {
    planned: number;
    observed: number;
    pending: number;
    failed_seeds: { genome_hash: string; seed: number; message: string; kind: string }[];
    failed_total: number;
    failures: {
      attempts: number;
      failed: number;
      by_kind: { infrastructure: number; execution: number };
      rate_ci: [number, number] | null;
    };
  };
  provenance: {
    images: string[];
    firmware: string | null;
    seeds: Record<string, unknown> | null;
    panel: Record<string, unknown>;
  };
  top_warnings: StudyWarning[];
  analysis: StudyAnalysis | null;
}

export interface MetricValue {
  seed: number;
  order: number;
  value: number;
  simulation_id: string | null;
  flag: "tukey" | "mad" | "both" | null;
}

export interface MetricDetail {
  metric: string;
  genome_hash: string;
  aggregator: SeedAggregatorSpec;
  summary: Description;
  values: MetricValue[];
  histogram: { from: number; to: number; count: number }[];
  box: { low: number; q1: number; median: number; q3: number; high: number } | null;
  qq: [number, number][];
  running: { n: number; mean: number; low: number; high: number }[];
  precision: {
    curve: { k: number; p: number; expected: number }[];
    n_emp: number | null;
  } | null;
  n_star: Precision | null;
  intervals: {
    mean_t: [number, number] | null;
    mean_bca: [number, number] | null;
    sd_bca: [number, number] | null;
    median: [number, number] | null;
    p90_upper: number | null;
    p95_upper: number | null;
    p99_upper: number | null;
  };
}

export interface StudyEstimate {
  samples: number;
  mean_seconds: number | null;
}
