import type { SeedAggregatorKind, SeedAggregatorSpec } from '../types/simlab'

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

export function isValidTrim(trim: number): boolean {
  return Number.isFinite(trim) && trim >= 0 && trim < 0.5
}

/** Values dropped from each end, ⌊trim·n⌋, as pylib/seed_aggregation.py computes it. */
export function trimmedPerSide(trim: number, seeds: number): number {
  return Math.floor(trim * seeds)
}

export function buildSeedAggregator(kind: SeedAggregatorKind, trim: number): SeedAggregatorSpec {
  return kind === 'trimmed_mean' ? { kind, trim } : kind
}

export function formatSeedAggregator(spec: SeedAggregatorSpec | undefined): string {
  if (spec === undefined) return 'mean (default)'
  const kind = typeof spec === 'string' ? spec : (spec.kind ?? 'mean')
  const option = SEED_AGGREGATORS.find((o) => o.kind === kind)
  if (!option) return typeof spec === 'string' ? spec : JSON.stringify(spec)
  const label = option.label.toLowerCase()
  if (kind === 'trimmed_mean') {
    const trim = typeof spec === 'object' ? (spec.trim ?? 0) : 0
    return `${label}, trim ${trim}`
  }
  return label
}
