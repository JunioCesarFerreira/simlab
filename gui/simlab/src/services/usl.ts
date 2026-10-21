/**
 * Reading a Universal Scalability Law fit, and planning the benchmark behind it.
 *
 *   X(N) = λN / (1 + σ(N − 1) + κN(N − 1))
 *
 * `σ` is contention — the share of the work that serialises on something
 * shared. `κ` is coherency — the cost of keeping parallel units consistent,
 * which grows with the number of *pairs*. With `κ = 0` throughput only
 * approaches `λ/σ`; any `κ > 0` makes it peak at `N* = √((1 − σ)/κ)` and then
 * fall.
 *
 * Pure, so the readings the page depends on — what to call a coefficient,
 * when a peak is real, what a benchmark will cost — can be tested on their
 * own.
 */
import type { BenchmarkSummary, UslFit, UslMetric, WaveMeasurement } from "../types/scalability";
import type { OverviewDto } from "../types/infra";

/** Mirrors the API's bounds, so the form refuses what the server would. */
export const MAX_LEVEL = 256;
export const MAX_LEVELS = 32;
export const MAX_REPETITIONS = 10;
export const DURATION_RANGE: [number, number] = [10, 3600];

export const METRIC_LABELS: Record<UslMetric, { label: string; hint: string }> = {
  batch: {
    label: "Batch",
    hint:
      "N ÷ wave makespan, from the first placement to the last result. What a " +
      "batch of N actually costs: it includes dispatch, transfers and the " +
      "slowest container of the wave.",
  },
  per_simulation: {
    label: "Per simulation",
    hint:
      "N ÷ mean time a container was held (Little's law). Insensitive to one " +
      "straggler, so it isolates how much each simulation slows down as " +
      "concurrency grows.",
  },
};

/**
 * How many simulations the pool can run at once right now.
 *
 * Summed over the hosts the dispatcher will place on — **not** the overview's
 * totals, which count every host so the page can show capacity that is
 * merely out of contact. A host that is disabled, withdrawing or offline can
 * still list free containers; proposing them as concurrency levels would plan
 * waves that queue, and a queued wave measures the pool full rather than at
 * the concurrency it was asked for.
 *
 * `active` is included: a container busy with something else now is part of
 * the pool the benchmark will run on once it is released.
 */
export function schedulablePool(overview: Pick<OverviewDto, "hosts"> | null | undefined): number {
  if (!overview) return 0;
  return overview.hosts
    .filter((host) => host.schedulable)
    .reduce((sum, host) => sum + (host.free ?? 0) + (host.active ?? 0), 0);
}

/**
 * Concurrency levels worth measuring on a pool of `available` containers.
 *
 * Powers of two, then the full pool. Doubling spaces the points evenly on
 * the scale where contention shows; the full pool is the level an operator
 * actually runs at, and the one where coherency — if there is any — is
 * largest.
 */
export function proposeLevels(available: number): number[] {
  const top = Math.max(1, Math.min(MAX_LEVEL, Math.floor(available)));
  const levels: number[] = [];
  for (let n = 1; n < top; n *= 2) levels.push(n);
  levels.push(top);
  return levels;
}

export type ParsedLevels = { ok: true; levels: number[] } | { ok: false; error: string };

/** `"1, 2, 4 8"` → `[1, 2, 4, 8]`, sorted and without repeats. */
export function parseLevels(text: string): ParsedLevels {
  const tokens = text
    .split(/[\s,;]+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (tokens.length === 0) return { ok: false, error: "Give at least one level." };
  const values: number[] = [];
  for (const token of tokens) {
    if (!/^\d+$/.test(token)) {
      return { ok: false, error: `“${token}” is not a whole number.` };
    }
    const n = Number(token);
    if (n < 1 || n > MAX_LEVEL) {
      return { ok: false, error: `Levels must be between 1 and ${MAX_LEVEL}.` };
    }
    values.push(n);
  }
  const levels = [...new Set(values)].sort((a, b) => a - b);
  if (levels.length > MAX_LEVELS) {
    return { ok: false, error: `At most ${MAX_LEVELS} distinct levels.` };
  }
  return { ok: true, levels };
}

export interface BenchmarkCost {
  waves: number;
  simulations: number;
  /** Levels larger than the pool: their waves queue, so they measure the
   *  pool at full occupancy rather than at that concurrency. */
  beyondPool: number[];
}

export function benchmarkCost(
  levels: number[],
  repetitions: number,
  available: number,
): BenchmarkCost {
  const reps = Math.max(0, Math.floor(repetitions));
  return {
    waves: levels.length * reps,
    simulations: levels.reduce((sum, n) => sum + n, 0) * reps,
    beyondPool: available > 0 ? levels.filter((n) => n > available) : [],
  };
}

/** Throughput, in the API's unit (simulations per minute). */
export function formatThroughput(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

/**
 * A coefficient, readable across the eight orders of magnitude κ spans.
 * Exact zero is shown as zero: it means the fit found none, which is a
 * different statement from "very small".
 */
export function formatCoefficient(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  if (Math.abs(value) >= 0.01) return value.toFixed(4);
  return value.toExponential(2);
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value < 60) return `${value.toFixed(1)} s`;
  const minutes = Math.floor(value / 60);
  return `${minutes} min ${Math.round(value - minutes * 60)} s`;
}

export type Verdict = "linear" | "contention" | "retrograde" | "unknown";

/**
 * The shape of the curve, in one word.
 *
 * Thresholds are about what an operator can act on, not about statistics: a
 * contention below 1% costs under 10% of throughput even at a hundred
 * containers, and a coherency whose peak sits past the tested range is a
 * property of a pool nobody has.
 */
export function verdict(fit: UslFit | null | undefined): Verdict {
  if (!fit || fit.status === "insufficient_data") return "unknown";
  if (fit.n_peak !== null && fit.peak_within_tested) return "retrograde";
  if (fit.sigma >= 0.01 || fit.kappa > 0) return "contention";
  return "linear";
}

/** What the fit means, in sentences, ordered by what to act on first. */
export function interpret(fit: UslFit | null | undefined, testedMax: number): string[] {
  if (!fit || fit.status === "insufficient_data") {
    return ["Not enough finished waves to fit a curve yet — at least two levels are needed."];
  }
  const lines: string[] = [];
  if (fit.status === "amdahl_only") {
    lines.push(
      "Only two levels were measured, so coherency (κ) cannot be told apart " +
        "from contention and is held at zero. Add a third level to detect a peak.",
    );
  }
  lines.push(
    `Contention σ = ${formatCoefficient(fit.sigma)}: about ${(fit.sigma * 100).toFixed(1)}% ` +
      "of each simulation's cost behaves as if it were serialised.",
  );
  if (fit.n_peak !== null && fit.x_peak !== null && fit.n_peak_int !== null) {
    const where = fit.peak_within_tested ? "within" : "beyond";
    lines.push(
      `Coherency κ = ${formatCoefficient(fit.kappa)}: throughput peaks at N* ≈ ` +
        `${fit.n_peak.toFixed(1)} (best integer: ${fit.n_peak_int}), at ` +
        `${formatThroughput(fit.x_peak)} simulations/min — ${where} the tested range` +
        (fit.peak_within_tested
          ? ". Adding containers past that point lowers throughput."
          : `, which only reached ${testedMax}. The peak is extrapolated.`),
    );
  } else if (fit.x_asymptote !== null) {
    lines.push(
      `No coherency cost detected: throughput keeps rising and approaches ` +
        `${formatThroughput(fit.x_asymptote)} simulations/min (λ/σ) without falling.`,
    );
  } else {
    lines.push("No contention and no coherency cost detected: scaling is linear.");
  }
  const efficiency = parallelEfficiency(fit, testedMax);
  if (efficiency !== null) {
    lines.push(
      `At ${testedMax} concurrent simulations the pool delivers ` +
        `${(efficiency * 100).toFixed(0)}% of linear scaling.`,
    );
  }
  if (fit.r2 !== null && fit.r2 < 0.8) {
    lines.push(
      `The fit is loose (R² = ${fit.r2.toFixed(2)}): the waves vary more than ` +
        "the model explains. More repetitions, or a quieter pool, would tighten it.",
    );
  }
  return lines;
}

/** X(N) ÷ λN at a given concurrency, from the model. */
export function parallelEfficiency(fit: UslFit, n: number): number | null {
  if (fit.status === "insufficient_data" || fit.lam <= 0 || n < 1) return null;
  const x = usl(n, fit);
  return x / (fit.lam * n);
}

/** The model itself, for drawing and for tests. */
export function usl(n: number, fit: Pick<UslFit, "lam" | "sigma" | "kappa">): number {
  return (fit.lam * n) / (1 + fit.sigma * (n - 1) + fit.kappa * n * (n - 1));
}

export interface ChartData {
  /** One point per valid wave — every repetition, so the spread is visible. */
  measured: Array<[number, number]>;
  /** Mean ± one standard deviation per level. */
  means: Array<{ n: number; mean: number; low: number; high: number }>;
  model: Array<[number, number]>;
  ideal: Array<[number, number]>;
  peak: { n: number; x: number; withinTested: boolean } | null;
  testedMax: number;
}

/** Everything the chart draws, derived from the API's answer. */
export function chartData(
  fit: UslFit | null | undefined,
  waves: WaveMeasurement[],
  metric: UslMetric,
): ChartData {
  const measured: Array<[number, number]> = [];
  for (const wave of waves) {
    const value = wave.throughput[metric];
    if (wave.valid && value) measured.push([wave.level, value]);
  }
  const testedMax = Math.max(1, ...waves.map((w) => w.level));
  if (!fit || fit.status === "insufficient_data") {
    return { measured, means: [], model: [], ideal: [], peak: null, testedMax };
  }
  return {
    measured,
    means: fit.levels.map((row) => ({
      n: row.level,
      mean: row.mean,
      low: Math.max(0, row.mean - row.std),
      high: row.mean + row.std,
    })),
    model: fit.curve.map((p) => [p.n, p.x]),
    ideal: fit.curve.map((p) => [p.n, p.ideal ?? fit.lam * p.n]),
    peak:
      fit.n_peak_int !== null && fit.x_peak !== null
        ? { n: fit.n_peak_int, x: fit.x_peak, withinTested: fit.peak_within_tested }
        : null,
    testedMax,
  };
}

/**
 * Where the y-axis stops: a little above the highest measured or modelled
 * throughput, rounded to a readable step.
 *
 * Not above the linear reference. λN grows without bound, and scaling the
 * axis to it squashed the measured curve against the floor — in exactly the
 * case the chart exists for, a pool that degrades, the peak became a ripple
 * at the bottom. The reference line is clipped at the top instead, which says
 * the same thing: linear scaling would be far above here.
 */
export function yAxisMax(data: Pick<ChartData, "measured" | "means" | "model">): number {
  const highest = Math.max(
    0,
    ...data.measured.map((p) => p[1]),
    ...data.means.map((m) => m.high),
    ...data.model.map((p) => p[1]),
  );
  if (highest <= 0) return 1;
  const target = highest * 1.25;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * magnitude >= target) ?? 10;
  return step * magnitude;
}

/** Progress of a benchmark, as a fraction of its waves. */
export function progress(summary: Pick<BenchmarkSummary, "waves" | "waves_finished">): number {
  if (!summary.waves) return 0;
  return Math.min(1, (summary.waves_finished ?? 0) / summary.waves);
}
