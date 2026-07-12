import type { BenchmarkDraft } from '../app/stores/syntheticStore'
import type { ExperimentCreateDto, ObjectiveItem } from '../types/simlab'

/**
 * Builds the ExperimentCreateDto for a synthetic-benchmark experiment.
 *
 * The "problem" is encoded as P0 — a pure analytical benchmark. The genome is a
 * flat real-valued decision vector x ∈ [0,1]ⁿ (genome length = n): no relays,
 * no sink, no region round-trip. The mo-engine drives it with textbook SBX +
 * polynomial mutation and evaluates the benchmark in-process (analytical
 * fast-path — no Simulation documents; only the batch strategy falls back to
 * the master-node's synthetic evaluator).
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
      // Encoded as P0 (pure synthetic benchmark). `name` MUST be the canonical
      // 'problem0' key — the mo-engine resolves the adapter and restores
      // chromosomes by this exact value. `n` is the decision-vector length.
      problem: {
        name: 'problem0',
        n: draft.nVars,
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
