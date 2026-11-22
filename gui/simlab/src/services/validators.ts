import type { ProblemDraft } from '../types/problem'
import { DEFAULT_P4, hasCoverageConstraint } from '../types/problem'

export type ValidationError = { field: string; message: string }

export function validateProblem(draft: ProblemDraft): ValidationError[] {
  const errors: ValidationError[] = []
  if (!draft.name.trim()) errors.push({ field: 'name', message: 'Name is required' })
  if (draft.radiusOfReach <= 0) errors.push({ field: 'radius_of_reach', message: 'radius_of_reach must be > 0' })
  if (draft.radiusOfInter <= 0) errors.push({ field: 'radius_of_inter', message: 'radius_of_inter must be > 0' })
  if (hasCoverageConstraint(draft.name)) {
    const alpha = draft.minCoveragePercentage
    if (!Number.isFinite(alpha) || alpha < 0 || alpha > 100)
      errors.push({ field: 'min_coverage_percentage', message: 'alpha (min_coverage_percentage) must be between 0 and 100' })
  }
  const [xmin, ymin, xmax, ymax] = draft.region
  if (xmin >= xmax) errors.push({ field: 'region', message: 'xmin must be < xmax' })
  if (ymin >= ymax) errors.push({ field: 'region', message: 'ymin must be < ymax' })
  if (!draft.sink) {
    errors.push({ field: 'sink', message: 'Sink is required' })
  } else {
    const { x, y } = draft.sink
    if (x < xmin || x > xmax || y < ymin || y > ymax)
      errors.push({ field: 'sink', message: 'Sink must be inside region' })
  }
  for (const c of draft.candidates) {
    if (c.x < xmin || c.x > xmax || c.y < ymin || c.y > ymax)
      errors.push({ field: 'candidates', message: `Candidate (${c.x},${c.y}) is outside region` })
  }
  if (draft.name === 'problem3') {
    const g = draft.gRequired ?? 1
    if (!Number.isInteger(g) || g < 1)
      errors.push({ field: 'g_required', message: 'g must be a whole number ≥ 1' })
  }
  if (draft.name === 'problem4') {
    const s = { ...DEFAULT_P4, ...draft.p4 }
    if (draft.targets.length === 0) errors.push({ field: 'targets', message: 'P4 needs at least one sensor mote' })
    if (draft.candidates.length === 0) errors.push({ field: 'candidates', message: 'P4 needs at least one stop besides the base' })
    for (const key of ['initialEnergy', 'bufferCapacity', 'speed', 'timeStep'] as const) {
      if (!(s[key] > 0)) errors.push({ field: 'p4', message: `${key} must be > 0` })
    }
    if (!(s.dataRate >= 0)) errors.push({ field: 'p4', message: 'dataRate must be ≥ 0' })
    if (!Number.isInteger(s.maxRouteLength) || s.maxRouteLength < 2)
      errors.push({ field: 'p4', message: 'maxRouteLength must be a whole number ≥ 2' })
    if (!(s.tauMin >= 0 && s.tauMin <= s.tauMax))
      errors.push({ field: 'p4', message: 'dwell bounds must satisfy 0 ≤ min ≤ max' })
  }
  for (const t of draft.targets) {
    if (t.x < xmin || t.x > xmax || t.y < ymin || t.y > ymax)
      errors.push({ field: 'targets', message: `Point (${t.x},${t.y}) is outside region` })
  }
  for (const m of draft.mobileNodes) {
    if (!m.name.trim()) errors.push({ field: 'mobile_nodes', message: 'Mobile node name required' })
    if (!m.sourceCode.trim()) errors.push({ field: 'mobile_nodes', message: `Node "${m.name}": source_code required` })
    if (m.speed <= 0) errors.push({ field: 'mobile_nodes', message: `Node "${m.name}": speed must be > 0` })
    if (m.timeStep <= 0) errors.push({ field: 'mobile_nodes', message: `Node "${m.name}": time_step must be > 0` })
    if (m.segments.length === 0) errors.push({ field: 'mobile_nodes', message: `Node "${m.name}": needs at least one segment` })
  }
  return errors
}

export function validateChromosome(draft: ProblemDraft): ValidationError[] {
  const errors: ValidationError[] = []
  const chrom = draft.chromosome
  if (!chrom) return errors

  if (chrom.kind !== draft.name) {
    errors.push({ field: 'chromosome', message: `Chromosome kind "${chrom.kind}" mismatches problem "${draft.name}"` })
    return errors
  }

  const [xmin, ymin, xmax, ymax] = draft.region

  if (chrom.kind === 'problem1') {
    if (chrom.relays.length > draft.numSensors) {
      errors.push({ field: 'chromosome.relays', message: `Too many relays: ${chrom.relays.length} > number_of_relays (${draft.numSensors})` })
    }
    if (chrom.relays.length < draft.numSensors) {
      errors.push({ field: 'chromosome.relays', message: `Missing relays: ${chrom.relays.length}/${draft.numSensors}` })
    }
    for (const r of chrom.relays) {
      if (r.x < xmin || r.x > xmax || r.y < ymin || r.y > ymax) {
        errors.push({ field: 'chromosome.relays', message: `Relay (${r.x},${r.y}) is outside region` })
      }
    }
  }

  if (chrom.kind === 'problem2' || chrom.kind === 'problem3') {
    if (chrom.mask.length !== draft.candidates.length) {
      errors.push({ field: 'chromosome.mask', message: `Mask length (${chrom.mask.length}) must equal candidates (${draft.candidates.length})` })
    }
    if (chrom.mask.some(b => b !== 0 && b !== 1)) {
      errors.push({ field: 'chromosome.mask', message: 'Mask entries must be 0 or 1' })
    }
  }

  if (chrom.kind === 'problem4') {
    const n = draft.candidates.length
    if (chrom.route.some(i => !Number.isInteger(i) || i < 0 || i >= n)) {
      errors.push({ field: 'chromosome.route', message: `Route indices must be integers in [0, ${n - 1}]` })
    }
    if (chrom.sojournTimes.length !== chrom.route.length) {
      errors.push({ field: 'chromosome.sojourn_times', message: `sojourn_times length (${chrom.sojournTimes.length}) must equal route length (${chrom.route.length})` })
    }
    if (chrom.sojournTimes.some(t => !isFinite(t) || t < 0)) {
      errors.push({ field: 'chromosome.sojourn_times', message: 'sojourn_times must be finite and ≥ 0' })
    }
  }

  return errors
}
