import type {
  ObjectiveItem,
  SeedAggregatorKind,
  SeedAggregatorSpec,
  SimulationParamsDto,
} from '../types/simlab'

export interface SeedAggregatorOption {
  kind: SeedAggregatorKind
  label: string
  hint: string
}

export const SEED_AGGREGATORS: readonly SeedAggregatorOption[] = [
  { kind: 'mean', label: 'Mean', hint: 'Arithmetic mean of the seeds. The engine default.' },
  { kind: 'median', label: 'Median', hint: 'Middle value of the seeds; one outlier seed does not move it.' },
  {
    kind: 'trimmed_mean',
    label: 'Trimmed mean',
    hint: 'Mean after dropping the lowest and highest values; the trim sets how many.',
  },
  { kind: 'min', label: 'Minimum', hint: 'Lowest value across the seeds.' },
  { kind: 'max', label: 'Maximum', hint: 'Highest value across the seeds.' },
]

export const DEFAULT_SEED_AGGREGATOR: SeedAggregatorKind = 'mean'
export const DEFAULT_TRIM = 0.1

export function aggregatorKind(spec: SeedAggregatorSpec | undefined): SeedAggregatorKind {
  if (spec === undefined) return DEFAULT_SEED_AGGREGATOR
  return typeof spec === 'string' ? spec : (spec.kind ?? DEFAULT_SEED_AGGREGATOR)
}

export function aggregatorTrim(spec: SeedAggregatorSpec | undefined): number {
  return typeof spec === 'object' && spec.trim !== undefined ? spec.trim : DEFAULT_TRIM
}

export function isValidTrim(trim: number): boolean {
  return Number.isFinite(trim) && trim >= 0 && trim < 0.5
}

export function isValidSeedAggregator(spec: SeedAggregatorSpec | undefined): boolean {
  return aggregatorKind(spec) !== 'trimmed_mean' || isValidTrim(aggregatorTrim(spec))
}

/** Values dropped from each end, ⌊trim·n⌋, as pylib/seed_aggregation.py computes it. */
export function trimmedPerSide(trim: number, seeds: number): number {
  return Math.floor(trim * seeds)
}

export function buildSeedAggregator(kind: SeedAggregatorKind, trim: number): SeedAggregatorSpec {
  return kind === 'trimmed_mean' ? { kind, trim } : kind
}

/** The objective's own aggregator, else the experiment-wide default; undefined means mean. */
export function effectiveSeedAggregator(
  objective: ObjectiveItem,
  simulation: SimulationParamsDto | undefined,
): SeedAggregatorSpec | undefined {
  return objective.aggregator ?? simulation?.aggregator
}

function aggregatorKey(spec: SeedAggregatorSpec | undefined): string {
  const kind = aggregatorKind(spec)
  return kind === 'trimmed_mean' ? `${kind}:${aggregatorTrim(spec)}` : kind
}

/**
 * Metrics listed in more than one objective with different aggregators. The
 * engine collects per-seed values per metric, so the API rejects these.
 */
export function conflictingAggregatorMetrics(objectives: ObjectiveItem[]): string[] {
  const seen = new Map<string, string>()
  const conflicts = new Set<string>()
  for (const o of objectives) {
    const metric = o.metric_name.trim()
    if (!metric) continue
    const key = aggregatorKey(o.aggregator)
    const previous = seen.get(metric)
    if (previous !== undefined && previous !== key) conflicts.add(metric)
    seen.set(metric, key)
  }
  return [...conflicts]
}

/**
 * Whether min/max picks the worst or the best seed for this goal; null for the
 * other aggregators.
 */
export function extremeCase(kind: SeedAggregatorKind, goal: string): 'worst' | 'best' | null {
  if (kind !== 'min' && kind !== 'max') return null
  if (goal !== 'min' && goal !== 'max') return null
  return kind === goal ? 'best' : 'worst'
}

export function formatSeedAggregator(spec: SeedAggregatorSpec | undefined): string {
  if (spec === undefined) return 'mean (default)'
  const kind = aggregatorKind(spec)
  const option = SEED_AGGREGATORS.find((o) => o.kind === kind)
  if (!option) return typeof spec === 'string' ? spec : JSON.stringify(spec)
  const label = option.label.toLowerCase()
  if (kind === 'trimmed_mean') {
    const trim = typeof spec === 'object' ? (spec.trim ?? 0) : 0
    return `${label}, trim ${trim}`
  }
  return label
}
