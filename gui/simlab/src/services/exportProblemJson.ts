import type { ProblemDraft, ExportedProblemFile, ExportedSojourn } from '../types/problem'
import { DEFAULT_P4, hasCandidates, hasTargets, hasCoverageConstraint } from '../types/problem'
import { buildSegmentExpressions } from './segmentExpressionBuilder'

/**
 * P4 in the shape of pylib's ProblemP4. The sink is the base and sojourn 0;
 * candidate j is sojourn j + 1; the targets are the sensor motes. The sink can
 * move between any two stops, and a stop sees the motes within reach.
 */
function exportP4(draft: ProblemDraft & { sink: { x: number; y: number } }): ExportedProblemFile {
  const s = { ...DEFAULT_P4, ...draft.p4 }
  const base: [number, number] = [draft.sink.x, draft.sink.y]
  const nodes = draft.targets.map(t => [t.x, t.y] as [number, number])
  const stops: Array<[number, number]> = [base, ...draft.candidates.map(c => [c.x, c.y] as [number, number])]
  const sojourns: ExportedSojourn[] = stops.map((position, id) => ({
    id,
    position,
    adjacency: stops.map((_, j) => j).filter(j => j !== id),
    visibleNodes: nodes
      .map((n, k) => [k, Math.hypot(n[0] - position[0], n[1] - position[1])] as const)
      .filter(([, d]) => d <= draft.radiusOfReach)
      .map(([k]) => k),
  }))
  return {
    problem: {
      name: draft.name,
      radius_of_reach: draft.radiusOfReach,
      radius_of_inter: draft.radiusOfInter,
      region: draft.region,
      sink: base,
      nodes,
      sink_base: base,
      sojourns,
      initial_energy: s.initialEnergy,
      buffer_capacity: s.bufferCapacity,
      data_rate: s.dataRate,
      speed: s.speed,
      time_step: s.timeStep,
      max_route_length: s.maxRouteLength,
      tau_bounds: [s.tauMin, s.tauMax],
      // The sink's path comes from the route; P4 has no other mobile mote.
      mobile_nodes: [],
    },
  }
}

export function exportProblem(draft: ProblemDraft): ExportedProblemFile {
  if (!draft.sink) throw new Error('Cannot export: sink is not defined')
  if (draft.name === 'problem4') return exportP4({ ...draft, sink: draft.sink })

  const candidatesOrSensors = hasCandidates(draft.name)
    ? { candidates: draft.candidates.map(c => [c.x, c.y] as [number, number]) }
    : { number_of_relays: draft.numSensors }

  const targetFields = hasTargets(draft.name)
    ? {
        radius_of_cover: draft.radiusOfCover,
        k_required: draft.kRequired,
        g_required: draft.gRequired ?? 1,
        targets: draft.targets.map(t => [t.x, t.y] as [number, number]),
      }
    : {}

  // alpha — only P1/P2 declare the trajectory coverage constraint; emitting it
  // elsewhere would ship a key those adapters silently ignore.
  const coverageFields = hasCoverageConstraint(draft.name)
    ? { min_coverage_percentage: draft.minCoveragePercentage }
    : {}

  return {
    problem: {
      name: draft.name,
      radius_of_reach: draft.radiusOfReach,
      radius_of_inter: draft.radiusOfInter,
      region: draft.region,
      ...targetFields,
      ...coverageFields,
      sink: [draft.sink.x, draft.sink.y],
      ...candidatesOrSensors,
      mobile_nodes: draft.mobileNodes.map(m => ({
        name: m.name,
        source_code: m.sourceCode,
        speed: m.speed,
        time_step: m.timeStep,
        is_closed: m.isClosed,
        is_round_trip: m.isRoundTrip,
        path_segments: m.segments.map(buildSegmentExpressions),
      })),
    },
  }
}
