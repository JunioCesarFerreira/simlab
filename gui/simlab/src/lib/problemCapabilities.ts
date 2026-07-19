/**
 * Per-problem GA capability map.
 *
 * Mirrors what each ProblemAdapter in mo-engine/lib/problem actually consumes
 * in set_ga_operator_configs (declared engine-side in CONSUMED_GA_KEYS). The
 * launch wizards use this to render only the knobs a problem reads and to
 * build a payload that matches what will really execute — keys not listed
 * here are ignored by the engine (with a warning in the mo-engine log).
 *
 * Keep in sync with:
 *   mo-engine/lib/problem/p0_synthetic.py … p4_mobile_sink_collection.py
 */

export interface ExtraParamSpec {
  /** Payload key, snake_case, exactly as the adapter reads it. */
  key: string
  label: string
  default: number
  min?: number
  max?: number
  step?: number
  hint?: string
}

export interface ProblemCapabilities {
  /** Selectable crossover methods. Empty → operator is fixed by design. */
  crossoverMethods: string[]
  defaultCrossoverMethod?: string
  /** Shown when the crossover operator is fixed (crossoverMethods empty). */
  fixedCrossoverLabel?: string
  /** Mutation is fixed by design for every problem; label shown as info. */
  fixedMutationLabel: string
  /** Whether the adapter reads apply_coverage_repair / repair_coverage_budget. */
  supportsCoverageRepair: boolean
  /** Problem-specific numeric hyperparameters the adapter consumes. */
  extraParams: ExtraParamSpec[]
}

const ETA_CX: ExtraParamSpec = {
  key: 'eta_cx',
  label: 'SBX distribution index (η_cx)',
  default: 20, min: 1, max: 200, step: 1,
  hint: 'Higher values keep offspring closer to the parents.',
}

const ETA_MT: ExtraParamSpec = {
  key: 'eta_mt',
  label: 'Polynomial mutation index (η_mt)',
  default: 20, min: 1, max: 200, step: 1,
  hint: 'Higher values produce smaller mutation steps.',
}

export const PROBLEM_CAPABILITIES: Record<string, ProblemCapabilities> = {
  problem0: {
    crossoverMethods: [],
    fixedCrossoverLabel: 'SBX (per-variable, bounded to [0,1])',
    fixedMutationLabel: 'polynomial (per-variable)',
    supportsCoverageRepair: false,
    extraParams: [ETA_CX, ETA_MT],
  },

  problem1: {
    crossoverMethods: ['sbx_with_radial_translate', 'rand_network'],
    defaultCrossoverMethod: 'sbx_with_radial_translate',
    fixedMutationLabel: 'polynomial on relay coordinates + bit-flip MAC, with connectivity repair',
    supportsCoverageRepair: true,
    extraParams: [
      { ...ETA_CX, default: 20 },
      { ...ETA_MT, default: 25 },
      {
        key: 'repair_coverage_budget',
        label: 'Coverage repair budget',
        default: 2, min: 0, max: 64, step: 1,
        hint: 'Max relay moves per coverage repair pass.',
      },
    ],
  },

  problem2: {
    crossoverMethods: [],
    fixedCrossoverLabel: 'uniform mask + connectivity repair (fixed by construction)',
    fixedMutationLabel: 'bit-flip mask + bit-flip MAC, with connectivity repair',
    supportsCoverageRepair: true,
    extraParams: [
      {
        key: 'repair_coverage_budget',
        label: 'Coverage repair budget',
        default: 8, min: 0, max: 64, step: 1,
        hint: 'Max candidate activations per coverage repair pass.',
      },
    ],
  },

  problem3: {
    crossoverMethods: [],
    fixedCrossoverLabel: 'uniform mask + connectivity repair (fixed by construction)',
    fixedMutationLabel: 'bit-flip mask + bit-flip MAC, with connectivity repair',
    supportsCoverageRepair: false,
    extraParams: [],
  },

  problem4: {
    crossoverMethods: [],
    fixedCrossoverLabel: 'route splice + repair, tau blend (fixed by construction)',
    fixedMutationLabel: 'route perturbation + Gaussian tau mutation',
    supportsCoverageRepair: false,
    extraParams: [
      {
        key: 'pm_tau',
        label: 'Tau mutation probability (pm_tau)',
        default: 0.5, min: 0, max: 1, step: 0.05,
        hint: 'Probability of mutating each dwell time (tau).',
      },
      {
        key: 'sigma_tau',
        label: 'Tau mutation std. dev. (sigma_tau)',
        default: 5, min: 0, max: 100, step: 0.5,
        hint: 'Std. deviation of the Gaussian perturbation on dwell times.',
      },
    ],
  },
}

const FALLBACK: ProblemCapabilities = {
  crossoverMethods: [],
  fixedCrossoverLabel: 'problem-defined',
  fixedMutationLabel: 'problem-defined',
  supportsCoverageRepair: false,
  extraParams: [],
}

export function capabilitiesFor(problemName: string | undefined | null): ProblemCapabilities {
  return PROBLEM_CAPABILITIES[problemName ?? ''] ?? FALLBACK
}

/** Initial extraParams record (key → default) for a problem's wizard form. */
export function defaultExtraParams(caps: ProblemCapabilities): Record<string, number> {
  return Object.fromEntries(caps.extraParams.map(p => [p.key, p.default]))
}
