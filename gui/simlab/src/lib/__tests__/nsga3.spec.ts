import { describe, it, expect } from 'vitest'
import {
  expectedMutatedVariables,
  referencePointCount,
  suggestedDivisions,
  suggestedPopulationSize,
} from '../nsga3'

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

describe('suggestedDivisions', () => {
  it('keeps roughly one reference direction per individual', () => {
    // M = 2: H = p + 1, so a population of 50 supports p = 49 — the value the
    // reference notebooks use, against the fixed p = 10 (11 directions) the
    // wizard used to send.
    expect(suggestedDivisions(2, 50)).toBe(49)
    expect(referencePointCount(2, suggestedDivisions(2, 50))).toBe(50)
  })

  it('never overshoots the population', () => {
    for (const M of [2, 3, 4, 6, 8]) {
      for (const pop of [12, 50, 100, 200]) {
        const p = suggestedDivisions(M, pop)
        expect(referencePointCount(M, p)).toBeLessThanOrEqual(pop)
        // ... and one division more would.
        expect(referencePointCount(M, p + 1)).toBeGreaterThan(pop)
      }
    }
  })

  it('shrinks as objectives grow, where a fixed p explodes', () => {
    expect(referencePointCount(6, 10)).toBe(3003)          // the old fixed default
    // The lattice is coarse in six objectives: p = 2 gives 21 directions and
    // the next step is already 56, past a population of 50. That granularity is
    // the construction's, not a choice — but 21 beats 3003.
    expect(suggestedDivisions(6, 50)).toBe(2)
    expect(referencePointCount(6, 2)).toBe(21)
    expect(referencePointCount(6, 3)).toBe(56)
  })

  it('rejects out-of-domain input', () => {
    expect(suggestedDivisions(1, 50)).toBe(0)
    expect(suggestedDivisions(3, 1)).toBe(0)
    expect(suggestedDivisions(2.5, 50)).toBe(0)
  })
})

describe('expectedMutatedVariables', () => {
  it('multiplies the two probabilities by the variable count', () => {
    // The wizard's old default over 10 variables: one variable every 20 children.
    expect(expectedMutatedVariables(0.1, 0.05, 10)).toBeCloseTo(0.05, 12)
    // The textbook convention: one variable per child.
    expect(expectedMutatedVariables(1.0, 0.1, 10)).toBeCloseTo(1.0, 12)
  })

  it('rejects out-of-domain input', () => {
    expect(expectedMutatedVariables(NaN, 0.1, 10)).toBe(0)
    expect(expectedMutatedVariables(0.1, -1, 10)).toBe(0)
  })
})
