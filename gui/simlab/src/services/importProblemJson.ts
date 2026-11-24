import { nanoid } from 'nanoid'
import type { ProblemDraft, ExportedProblemFile, ExportedProblem, DraftSegment } from '../types/problem'
import { DEFAULT_P4, hasCandidates, hasTargets } from '../types/problem'

export type ImportResult =
  | { ok: true; draft: ProblemDraft }
  | { ok: false; error: string }

function parseSegment(raw: [string, string]): DraftSegment {
  const [exprX, exprY] = raw
  return { type: 'custom', exprX, exprY }
}

export function importProblemJson(json: string): ImportResult {
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch (e) {
    return { ok: false, error: `Invalid JSON: ${e}` }
  }

  const file = parsed as ExportedProblemFile
  if (!file?.problem) return { ok: false, error: 'Field "problem" not found' }

  const p = file.problem

  if (typeof p.name !== 'string') return { ok: false, error: '"name" must be a string' }
  if (typeof p.radius_of_reach !== 'number') return { ok: false, error: '"radius_of_reach" must be a number' }
  if (typeof p.radius_of_inter !== 'number') return { ok: false, error: '"radius_of_inter" must be a number' }
  if (!Array.isArray(p.region) || p.region.length !== 4) return { ok: false, error: '"region" must be an array of 4 numbers' }
  if (p.name === 'problem4') return importP4(p)
  if (!Array.isArray(p.sink) || p.sink.length !== 2) return { ok: false, error: '"sink" must be [x, y]' }

  const draft: ProblemDraft = {
    name: p.name,
    radiusOfReach: p.radius_of_reach,
    radiusOfInter: p.radius_of_inter,
    radiusOfCover: p.radius_of_cover ?? 90,
    kRequired: p.k_required ?? 1,
    gRequired: p.g_required ?? 1,
    // alpha defaults to 100% (full trajectory coverage), matching
    // ProblemP1/ProblemP2.cast in pylib/config/problems.py.
    minCoveragePercentage: p.min_coverage_percentage ?? 100,
    region: p.region as [number, number, number, number],
    sink: { x: p.sink[0], y: p.sink[1] },
    candidates: hasCandidates(p.name)
      ? (p.candidates ?? []).map((c: [number, number]) => ({ id: nanoid(), x: c[0], y: c[1] }))
      : [],
    targets: hasTargets(p.name)
      ? (p.targets ?? []).map((t: [number, number]) => ({ id: nanoid(), x: t[0], y: t[1] }))
      : [],
    numSensors: !hasCandidates(p.name) ? (p.number_of_relays ?? 1) : 1,
    mobileNodes: (p.mobile_nodes ?? []).map((m: typeof p.mobile_nodes[0]) => ({
      id: nanoid(),
      name: m.name,
      sourceCode: m.source_code,
      speed: m.speed,
      timeStep: m.time_step,
      isClosed: m.is_closed,
      isRoundTrip: m.is_round_trip,
      segments: (m.path_segments ?? []).map(parseSegment),
    })),
    chromosome: null,
  }

  return { ok: true, draft }
}

/** The inverse of exportProblem for P4: sojourn 0 is the base, the others the candidates. */
function importP4(p: ExportedProblem): ImportResult {
  const base = p.sink_base ?? p.sink
  if (!Array.isArray(base) || base.length !== 2) return { ok: false, error: '"sink_base" must be [x, y]' }
  if (!Array.isArray(p.nodes)) return { ok: false, error: '"nodes" must be a list of [x, y]' }
  if (!Array.isArray(p.sojourns) || p.sojourns.length === 0) return { ok: false, error: '"sojourns" must be a non-empty list' }
  const tau = p.tau_bounds ?? [DEFAULT_P4.tauMin, DEFAULT_P4.tauMax]
  return {
    ok: true,
    draft: {
      name: 'problem4',
      radiusOfReach: p.radius_of_reach,
      radiusOfInter: p.radius_of_inter,
      radiusOfCover: 90,
      kRequired: 1,
      gRequired: 1,
      minCoveragePercentage: 100,
      region: p.region as [number, number, number, number],
      sink: { x: base[0], y: base[1] },
      candidates: p.sojourns.slice(1).map(s => ({ id: nanoid(), x: s.position[0], y: s.position[1] })),
      targets: p.nodes.map(n => ({ id: nanoid(), x: n[0], y: n[1] })),
      numSensors: 1,
      mobileNodes: [],
      chromosome: null,
      p4: {
        initialEnergy: p.initial_energy ?? DEFAULT_P4.initialEnergy,
        bufferCapacity: p.buffer_capacity ?? DEFAULT_P4.bufferCapacity,
        dataRate: p.data_rate ?? DEFAULT_P4.dataRate,
        speed: p.speed ?? DEFAULT_P4.speed,
        timeStep: p.time_step ?? DEFAULT_P4.timeStep,
        maxRouteLength: p.max_route_length ?? DEFAULT_P4.maxRouteLength,
        tauMin: tau[0],
        tauMax: tau[1],
      },
    },
  }
}
