import { nanoid } from 'nanoid'
import type { ProblemDraft, ExportedProblemFile, DraftSegment } from '../types/problem'
import { hasCandidates, hasTargets } from '../types/problem'

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
    return { ok: false, error: `JSON inválido: ${e}` }
  }

  const file = parsed as ExportedProblemFile
  if (!file?.problem) return { ok: false, error: 'Campo "problem" não encontrado' }

  const p = file.problem

  if (typeof p.name !== 'string') return { ok: false, error: '"name" deve ser string' }
  if (typeof p.radius_of_reach !== 'number') return { ok: false, error: '"radius_of_reach" deve ser número' }
  if (typeof p.radius_of_inter !== 'number') return { ok: false, error: '"radius_of_inter" deve ser número' }
  if (!Array.isArray(p.region) || p.region.length !== 4) return { ok: false, error: '"region" deve ser array de 4 números' }
  if (!Array.isArray(p.sink) || p.sink.length !== 2) return { ok: false, error: '"sink" deve ser [x, y]' }

  const draft: ProblemDraft = {
    name: p.name,
    radiusOfReach: p.radius_of_reach,
    radiusOfInter: p.radius_of_inter,
    radiusOfCover: p.radius_of_cover ?? 90,
    kRequired: p.k_required ?? 1,
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
