import { describe, it, expect } from 'vitest'
import {
  aggregatorKind,
  aggregatorTrim,
  buildSeedAggregator,
  conflictingAggregatorMetrics,
  effectiveSeedAggregator,
  extremeCase,
  formatSeedAggregator,
  isValidSeedAggregator,
  isValidTrim,
  trimmedPerSide,
} from '../seedAggregator'

describe('buildSeedAggregator', () => {
  it('sends the bare kind when there is no parameter', () => {
    expect(buildSeedAggregator('median', 0.1)).toBe('median')
    expect(buildSeedAggregator('mean', 0.3)).toBe('mean')
  })

  it('sends the dict form for trimmed_mean', () => {
    expect(buildSeedAggregator('trimmed_mean', 0.2)).toEqual({ kind: 'trimmed_mean', trim: 0.2 })
  })
})

describe('aggregatorKind / aggregatorTrim', () => {
  it('reads both spec forms, defaulting to mean', () => {
    expect(aggregatorKind(undefined)).toBe('mean')
    expect(aggregatorKind('max')).toBe('max')
    expect(aggregatorKind({ kind: 'trimmed_mean', trim: 0.3 })).toBe('trimmed_mean')
  })

  it('keeps a stored trim and defaults otherwise', () => {
    expect(aggregatorTrim({ kind: 'trimmed_mean', trim: 0.3 })).toBe(0.3)
    expect(aggregatorTrim('median')).toBe(0.1)
  })
})

describe('trim validation', () => {
  it('accepts [0, 0.5)', () => {
    expect(isValidTrim(0)).toBe(true)
    expect(isValidTrim(0.49)).toBe(true)
  })

  it('rejects out of range and NaN', () => {
    expect(isValidTrim(0.5)).toBe(false)
    expect(isValidTrim(-0.1)).toBe(false)
    expect(isValidTrim(Number.NaN)).toBe(false)
  })

  it('only constrains trimmed_mean', () => {
    expect(isValidSeedAggregator('max')).toBe(true)
    expect(isValidSeedAggregator({ kind: 'trimmed_mean', trim: 0.6 })).toBe(false)
  })
})

describe('trimmedPerSide', () => {
  it('floors trim·n like the engine', () => {
    expect(trimmedPerSide(0.1, 8)).toBe(0)
    expect(trimmedPerSide(0.2, 5)).toBe(1)
    expect(trimmedPerSide(0.25, 8)).toBe(2)
  })
})

describe('effectiveSeedAggregator', () => {
  it('prefers the objective, then the experiment default', () => {
    const sim = { aggregator: 'median' as const }
    expect(effectiveSeedAggregator({ metric_name: 'l', goal: 'min', aggregator: 'max' }, sim)).toBe('max')
    expect(effectiveSeedAggregator({ metric_name: 'l', goal: 'min' }, sim)).toBe('median')
    expect(effectiveSeedAggregator({ metric_name: 'l', goal: 'min' }, {})).toBeUndefined()
  })
})

describe('extremeCase', () => {
  it('reads min/max against the goal', () => {
    expect(extremeCase('max', 'min')).toBe('worst')
    expect(extremeCase('min', 'max')).toBe('worst')
    expect(extremeCase('min', 'min')).toBe('best')
    expect(extremeCase('max', 'max')).toBe('best')
    expect(extremeCase('median', 'min')).toBeNull()
  })
})

describe('conflictingAggregatorMetrics', () => {
  it('flags a metric aggregated two ways', () => {
    expect(conflictingAggregatorMetrics([
      { metric_name: 'latency', goal: 'min', aggregator: 'max' },
      { metric_name: 'energy', goal: 'min', aggregator: 'max' },
      { metric_name: 'latency', goal: 'max', aggregator: 'min' },
    ])).toEqual(['latency'])
  })

  it('accepts a repeated metric with the same aggregator', () => {
    expect(conflictingAggregatorMetrics([
      { metric_name: 'latency', goal: 'min', aggregator: { kind: 'trimmed_mean', trim: 0.2 } },
      { metric_name: 'latency', goal: 'max', aggregator: { kind: 'trimmed_mean', trim: 0.2 } },
    ])).toEqual([])
  })
})

describe('formatSeedAggregator', () => {
  it('marks an absent aggregator as the default mean', () => {
    expect(formatSeedAggregator(undefined)).toBe('mean (default)')
  })

  it('formats both spec forms', () => {
    expect(formatSeedAggregator('max')).toBe('maximum')
    expect(formatSeedAggregator({ kind: 'median' })).toBe('median')
    expect(formatSeedAggregator({ kind: 'trimmed_mean', trim: 0.1 })).toBe('trimmed mean, trim 0.1')
  })

  it('shows an unknown kind as stored', () => {
    expect(formatSeedAggregator('geomean' as never)).toBe('geomean')
  })
})
