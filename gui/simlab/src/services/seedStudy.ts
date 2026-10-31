/**
 * Reading a seed study, and planning one.
 *
 * Pure, so what the page says can be tested on its own. The statistics are
 * computed and tested in `pylib/seed_study.py`; what is pinned here is what an
 * operator is told about them, starting with the two answers to "how many
 * seeds", which must never read as one.
 */
import type {
  ReportRow,
  StudyDetail,
  StudyMetric,
  StudySummary,
  StudyTargets,
} from "../types/seedStudy";
import type { ObjectiveItem } from "../types/simlab";

/** Mirrors the API's bounds, so the form refuses what the server would. */
export const MIN_SEEDS = 3;
export const MAX_SEEDS = 500;
export const MAX_CHROMOSOMES = 100;
export const MAX_SIMULATIONS = 5000;
export const MAX_CONTROL_REPLICATES = 5;
export const SEED_LIMIT = 2 ** 31 - 2 ** 16;
/** Below this panel size, `m*` swings too much between panels. */
export const MIN_PANEL = 10;
const LOW16 = 2 ** 16;
const GUARD = 64;

export const DEFAULT_SEEDS = 30;
export const DEFAULT_PANEL = 20;
export const DEFAULT_DURATION = 180;
export const DEFAULT_CONTROL_REPLICATES = 2;
export const DEFAULT_TARGETS: StudyTargets = {
  confidence: 0.95,
  relative_error: 0.05,
  reliability: 0.9,
};

export function defaultObjectives(): ObjectiveItem[] {
  return [
    { metric_name: "latency", goal: "min", aggregator: "mean" },
    { metric_name: "energy", goal: "min", aggregator: "mean" },
    { metric_name: "throughput", goal: "max", aggregator: "mean" },
  ];
}

export interface StudyCost {
  main: number;
  control: number;
  total: number;
  overLimit: boolean;
}

export function studyCost(seeds: number, chromosomes: number, replicates: number): StudyCost {
  const main = seeds * chromosomes;
  const control = chromosomes * replicates;
  return { main, control, total: main + control, overLimit: main > MAX_SIMULATIONS };
}

/**
 * Hours for `total` simulations on `pool` workers, when there is history to say so.
 *
 * In whole rounds of the pool: five simulations on twelve workers still take
 * as long as one simulation does.
 */
export function estimateHours(
  total: number,
  meanSeconds: number | null | undefined,
  pool: number,
): number | null {
  if (!meanSeconds || pool < 1) return null;
  return (Math.ceil(total / pool) * meanSeconds) / 3600;
}

export function formatHours(hours: number): string {
  if (hours < 1) return `${Math.max(1, Math.round(hours * 60))} min`;
  return `${hours.toFixed(1)} h`;
}

export interface SeedCheck {
  errors: string[];
  lowCollisions: number;
  closePairs: number;
}

/**
 * What an imported list looks like before it is sent.
 *
 * A repeat or a value out of range is an error, as it is on the server. Seeds
 * that are close in their low 16 bits are only a warning: the list may come
 * from an earlier experiment that someone wants to reproduce as it was.
 */
export function checkSeeds(seeds: number[]): SeedCheck {
  const errors: string[] = [];
  if (seeds.length < MIN_SEEDS || seeds.length > MAX_SEEDS) {
    errors.push(`A study takes ${MIN_SEEDS} to ${MAX_SEEDS} seeds, this list has ${seeds.length}.`);
  }
  if (new Set(seeds).size !== seeds.length) errors.push("The list repeats a seed.");
  if (seeds.some((seed) => seed < 1 || seed >= SEED_LIMIT)) {
    errors.push(`Seeds must be between 1 and ${SEED_LIMIT - 1}.`);
  }
  let lowCollisions = 0;
  let closePairs = 0;
  const low = seeds.map((seed) => seed % LOW16);
  for (let i = 0; i < low.length; i++) {
    for (let j = i + 1; j < low.length; j++) {
      const apart = Math.abs(low[i]! - low[j]!);
      const distance = Math.min(apart, LOW16 - apart);
      if (distance === 0) lowCollisions++;
      else if (distance < GUARD) closePairs++;
    }
  }
  return { errors, lowCollisions, closePairs };
}

export function progress(study: Pick<StudySummary, "finished" | "simulations">): number {
  return study.simulations ? Math.min(1, (study.finished ?? 0) / study.simulations) : 0;
}

/** A number at a precision that suits it, without an exponent for everyday sizes. */
export function formatValue(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const size = Math.abs(value);
  if (size !== 0 && (size >= 1e7 || size < 1e-3)) return value.toExponential(2);
  if (size >= 1000) return Math.round(value).toLocaleString("en-US");
  return Number(value.toPrecision(4)).toString();
}

export function formatInterval(interval: [number, number] | null | undefined): string {
  return interval ? `[${formatValue(interval[0])}, ${formatValue(interval[1])}]` : "—";
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export type Tone = "ok" | "warn" | "bad" | "muted";

export const METRIC_STATUS: Record<string, { label: string; tone: Tone; hint: string }> = {
  ok: { label: "ok", tone: "ok", hint: "Usable." },
  extrapolated: {
    label: "extrapolated",
    tone: "warn",
    hint: "n* is past the seeds that ran, so it projects the dispersion of this sample.",
  },
  indicative: {
    label: "indicative",
    tone: "warn",
    hint: "n* is more than three times the seeds that ran, or fewer than 10 ran: take it as an order of magnitude.",
  },
  constant: {
    label: "constant",
    tone: "muted",
    hint: "Does not change with the seed, so seeds add nothing to it.",
  },
  overflow_suspect: {
    label: "measurement defect",
    tone: "bad",
    hint: "Values near 2⁶³: an unsigned subtraction wrapped around. No number of seeds fixes a defective measurement.",
  },
  near_zero_mean: {
    label: "near zero",
    tone: "warn",
    hint: "The interval of the mean contains zero, so a relative error has no meaning.",
  },
  not_applicable: {
    label: "no precision target",
    tone: "muted",
    hint: "The expected extreme moves with the number of seeds, so there is no fixed value to be close to.",
  },
  beyond_sample: {
    label: "beyond the sample",
    tone: "warn",
    hint: "Resampling did not reach the target within the seeds that ran.",
  },
  too_few: { label: "too few", tone: "muted", hint: "Not enough seeds yet." },
  no_data: { label: "no data", tone: "muted", hint: "Nothing measured yet." },
};

/** Most study-level warnings undermine the result. This one is only a caution. */
export function warningTone(code: string): Tone {
  return code === "seed_structure" ? "warn" : "bad";
}

export function statusInfo(status: string): { label: string; tone: Tone; hint: string } {
  return METRIC_STATUS[status] ?? { label: status, tone: "muted", hint: "" };
}

/** The status a row of the table shows: the metric's own, else the one of its n*. */
export function metricStatus(metric: StudyMetric, report?: ReportRow): string {
  if (metric.status !== "ok") return metric.status;
  if (report) return report.status;
  for (const status of ["extrapolated", "near_zero_mean", "indicative"]) {
    if (metric.summary.statuses.includes(status)) return status;
  }
  return "ok";
}

/** The reading of Shapiro–Wilk. A diagnostic: it selects no method. */
export function normalityLabel(rejected: number | null | undefined): string {
  if (rejected === null || rejected === undefined) return "—";
  if (rejected === 0) return "not rejected";
  return rejected >= 0.5 ? "rejected" : "doubtful";
}

/** The scope every recommendation is valid in, said next to the number. */
export function scopeSentence(detail: StudyDetail): string {
  const image = detail.provenance.images.join(", ");
  return (
    `Valid for ${detail.problem ?? "this instance"}, ${detail.duration} min simulated` +
    (image ? `, image ${image}` : "") +
    "."
  );
}

/** What to do with the study, in one line. */
export function headline(detail: StudyDetail): string {
  const analysis = detail.analysis;
  if (!analysis) return "No result yet.";
  const seeds = analysis.recommendation.optimise.max;
  if (seeds !== null) return `Use ${seeds} seed${seeds === 1 ? "" : "s"} to optimise.`;
  if (analysis.shape.chromosomes < 2) {
    return "One solution measures its own noise. Ranking solutions needs a panel.";
  }
  return "The panel gives no number of seeds to rank these solutions.";
}

/** The absolute half-widths of a request, as the route reads them. */
export function absoluteParam(absolute?: Record<string, number>): string | undefined {
  const parts = Object.entries(absolute ?? {}).map(([metric, value]) => `${metric}:${value}`);
  return parts.length ? parts.join(",") : undefined;
}

/** Metrics whose mean is too close to zero for a relative error to mean anything. */
export function nearZeroMetrics(metrics: StudyMetric[]): string[] {
  return metrics
    .filter((metric) => metric.summary.statuses.includes("near_zero_mean"))
    .map((metric) => metric.name);
}

export function chromosomeLabel(hash: string, index: number): string {
  return `#${index + 1} · ${hash.slice(0, 8)}`;
}

export function aggregatorLabel(spec: StudyMetric["aggregator"]): string {
  if (typeof spec === "string") return spec;
  return spec.kind === "trimmed_mean" ? `trimmed mean (${spec.trim ?? 0})` : (spec.kind ?? "mean");
}
