import { describe, it, expect } from 'vitest'
import {
  buildSeedAggregator,
  formatSeedAggregator,
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

describe('isValidTrim', () => {
  it('accepts [0, 0.5)', () => {
    expect(isValidTrim(0)).toBe(true)
    expect(isValidTrim(0.49)).toBe(true)
  })

  it('rejects out of range and NaN', () => {
    expect(isValidTrim(0.5)).toBe(false)
    expect(isValidTrim(-0.1)).toBe(false)
    expect(isValidTrim(Number.NaN)).toBe(false)
  })
})

describe('trimmedPerSide', () => {
  it('floors trim·n like the engine', () => {
    expect(trimmedPerSide(0.1, 8)).toBe(0)
    expect(trimmedPerSide(0.2, 5)).toBe(1)
    expect(trimmedPerSide(0.25, 8)).toBe(2)
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
