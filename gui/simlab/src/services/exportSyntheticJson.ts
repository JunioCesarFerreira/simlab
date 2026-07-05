import type { BenchmarkDraft } from '../app/stores/syntheticStore'
import type { ExperimentCreateDto, ObjectiveItem } from '../types/simlab'

/**
 * Builds the ExperimentCreateDto for a synthetic-benchmark experiment.
 *
 * The "problem" is encoded as a P1-style configuration with no mobile nodes
 * and 100% trivially-covered trajectory — so no chromosome is penalised and
 * the mo-engine generates relay positions freely inside the region.
 *
 * n_relays = ceil(nVars / 2) — each relay contributes two (x,y) coordinates,
 * giving a genome of length 2·n_relays ≥ nVars decision variables.
 */
export function exportSyntheticExperiment(params: {
  draft: BenchmarkDraft
  name: string
  strategy: string
  algorithm: Record<string, unknown>
  objectives: ObjectiveItem[]
  randomSeed: number
}): ExperimentCreateDto {
  const { draft, name, strategy, algorithm, objectives, randomSeed } = params

  const nRelays = Math.ceil(draft.nVars / 2)
  const [xmin, ymin, xmax, ymax] = draft.region
  const regionCenter: [number, number] = [(xmin + xmax) / 2, (ymin + ymax) / 2]
  const radius = Math.max((xmax - xmin), (ymax - ymin))

  // Auto-generate data_conversion_config based on objective names.
  // The 'direct' kind is a sentinel: synthetic_data.py writes objectives
  // keyed by metric_name directly, bypassing CSV parsing entirely.
  const metrics = objectives.map(o => ({
    name: o.metric_name,
    kind: 'direct',
    column: o.metric_name,
  }))

  return {
    id: null,
    name: name.trim(),
    status: 'Waiting',
    system_message: null,
    created_time: null,
    start_time: null,
    end_time: null,
    generations: [],
    parameters: {
      strategy,
      algorithm: {
        ...algorithm,
        random_seed: randomSeed,
      },
      simulation: {
        duration: 1,
        random_seeds: [randomSeed],
        synthetic: {
          enabled: true,
          bench: draft.benchmark,
          noise_std: draft.noiseStd,
        },
      },
      problem: {
        name: 'benchmark',
        type: 'p1',
        radius_of_reach: radius,
        radius_of_inter: radius,
        region: draft.region,
        sink: regionCenter,
        mobile_nodes: [],
        min_coverage_percentage: 0.0,
        number_of_relays: nRelays,
      },
      objectives,
    },
    source_repository_options: {},
    data_conversion_config: {
      node_col: '_synthetic',
      time_col: '_synthetic',
      metrics,
    },
    pareto_front: null,
  }
}
