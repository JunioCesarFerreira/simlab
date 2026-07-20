import { describe, it, expect } from 'vitest'
import { referencePointCount, suggestedPopulationSize } from '../nsga3'

describe('referencePointCount', () => {
  // Canonical NSGA-III values (Deb & Jain 2014): M=3 p=12 → 91.
  it('matches known Das-Dennis lattice sizes', () => {
    expect(referencePointCount(3, 12)).toBe(91)
    expect(referencePointCount(3, 10)).toBe(66)
    expect(referencePointCount(2, 10)).toBe(11)
    expect(referencePointCount(5, 6)).toBe(210)
  })

  it('is 0 outside the domain', () => {
    expect(referencePointCount(1, 10)).toBe(0)
    expect(referencePointCount(3, 0)).toBe(0)
    expect(referencePointCount(3.5, 10)).toBe(0)
    expect(referencePointCount(NaN, 10)).toBe(0)
  })
})

describe('suggestedPopulationSize', () => {
  it('rounds H up to the next multiple of 4', () => {
    expect(suggestedPopulationSize(91)).toBe(92)
    expect(suggestedPopulationSize(92)).toBe(92)
    expect(suggestedPopulationSize(66)).toBe(68)
    expect(suggestedPopulationSize(0)).toBe(0)
  })
})
