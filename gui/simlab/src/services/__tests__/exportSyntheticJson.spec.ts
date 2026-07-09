import { describe, it, expect } from 'vitest'
import { exportSyntheticExperiment } from '../exportSyntheticJson'
import type { BenchmarkDraft } from '../../app/stores/syntheticStore'
import type { ObjectiveItem } from '../../types/simlab'

function makeDraft(overrides: Partial<BenchmarkDraft> = {}): BenchmarkDraft {
  return {
    benchmark: 'DTLZ2',
    M: 3,
    nVars: 10,
    noiseStd: 0,
    ...overrides,
  }
}

const objectives: ObjectiveItem[] = [
  { metric_name: 'f1', goal: 'min' },
  { metric_name: 'f2', goal: 'min' },
  { metric_name: 'f3', goal: 'min' },
]

function build(draft = makeDraft()) {
  return exportSyntheticExperiment({
    draft,
    name: '  dtlz2-run  ',
    strategy: 'nsga3',
    algorithm: { population_size: 50, number_of_generations: 20 },
    objectives,
    randomSeed: 42,
  })
}

describe('exportSyntheticExperiment', () => {
  it('flags the experiment as synthetic with the chosen benchmark', () => {
    const dto = build()
    const sim = dto.parameters.simulation as Record<string, any>
    expect(sim.synthetic).toEqual({ enabled: true, bench: 'DTLZ2', noise_std: 0 })
  })

  it('encodes the problem as P0 (canonical name) with n = nVars and no WSN fields', () => {
    const dto = build(makeDraft({ nVars: 7 }))
    const problem = dto.parameters.problem as Record<string, any>
    // MUST be the canonical key the mo-engine resolves the adapter by
    expect(problem.name).toBe('problem0')
    expect(problem.n).toBe(7)
    // Pure synthetic: no relays, sink, region or coverage in the problem doc
    expect(problem.number_of_relays).toBeUndefined()
    expect(problem.sink).toBeUndefined()
    expect(problem.region).toBeUndefined()
  })

  it('builds direct data-conversion metrics keyed by objective name', () => {
    const dto = build()
    expect(dto.data_conversion_config.metrics).toEqual([
      { name: 'f1', kind: 'direct', column: 'f1' },
      { name: 'f2', kind: 'direct', column: 'f2' },
      { name: 'f3', kind: 'direct', column: 'f3' },
    ])
  })

  it('trims the name, carries the seed, and leaves source repos empty', () => {
    const dto = build()
    expect(dto.name).toBe('dtlz2-run')
    const sim = dto.parameters.simulation as Record<string, any>
    expect(sim.random_seeds).toEqual([42])
    expect(dto.source_repository_options).toEqual({})
    expect(dto.parameters.strategy).toBe('nsga3')
  })

  it('passes the noise level through', () => {
    const dto = build(makeDraft({ noiseStd: 0.05 }))
    const sim = dto.parameters.simulation as Record<string, any>
    expect(sim.synthetic.noise_std).toBe(0.05)
  })
})
