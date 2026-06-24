export type Region = [number, number, number, number] // [xmin, ymin, xmax, ymax]

export type DraftSegment =
  | { type: 'line'; start: [number, number]; end: [number, number] }
  | { type: 'ellipse'; center: [number, number]; radiusX: number; radiusY: number }
  | { type: 'custom'; exprX: string; exprY: string }

export type MobileRouteDraft = {
  id: string
  name: string
  sourceCode: string
  speed: number
  timeStep: number
  isClosed: boolean
  isRoundTrip: boolean
  segments: DraftSegment[]
}

export type CandidatePoint = {
  id: string
  x: number
  y: number
}

export type TargetPoint = {
  id: string
  x: number
  y: number
}

export type SinkPoint = {
  x: number
  y: number
}

export const PROBLEM_NAMES = ['problem1', 'problem2', 'problem3', 'problem4'] as const
export type ProblemName = typeof PROBLEM_NAMES[number]

export function hasCandidates(name: string): boolean {
  return name !== 'problem1'
}

export function hasTargets(name: string): boolean {
  return name === 'problem3'
}

export type ProblemDraft = {
  name: string
  radiusOfReach: number
  radiusOfInter: number
  radiusOfCover: number
  kRequired: number
  region: Region
  sink: SinkPoint | null
  candidates: CandidatePoint[]
  targets: TargetPoint[]
  numSensors: number
  mobileNodes: MobileRouteDraft[]
  chromosome: ChromosomeDraft | null
}

// ── Chromosome (solution) drafts ──────────────────────────────────────────────
export type MacProtocol = 'csma' | 'tsch'

export type RelayPoint = {
  id: string
  x: number
  y: number
}

export type ChromosomeP1Draft = {
  kind: 'problem1'
  macProtocol: MacProtocol
  relays: RelayPoint[]
}

export type ChromosomeP2Draft = {
  kind: 'problem2'
  macProtocol: MacProtocol
  /** Aligned to draft.candidates (same length, same order). */
  mask: number[]
}

export type ChromosomeP3Draft = {
  kind: 'problem3'
  macProtocol: MacProtocol
  /** Aligned to draft.candidates (same length, same order). */
  mask: number[]
}

export type ChromosomeP4Draft = {
  kind: 'problem4'
  macProtocol: MacProtocol
  /** Indices into draft.candidates. */
  route: number[]
  /** Same length as route. */
  sojournTimes: number[]
}

export type ChromosomeDraft =
  | ChromosomeP1Draft
  | ChromosomeP2Draft
  | ChromosomeP3Draft
  | ChromosomeP4Draft

// Exported (mo-engine) shape: camelCase → snake_case, MacProtocol → 0|1.
export type ExportedChromosomeP1 = {
  mac_protocol: 0 | 1
  relays: Array<{ x: number; y: number }>
}
export type ExportedChromosomeP2 = { mac_protocol: 0 | 1; mask: number[] }
export type ExportedChromosomeP3 = { mac_protocol: 0 | 1; mask: number[] }
export type ExportedChromosomeP4 = {
  mac_protocol: 0 | 1
  route: number[]
  sojourn_times: number[]
}

export type ExportedChromosomeFile = {
  problem_name: ProblemName
  chromosome:
    | ExportedChromosomeP1
    | ExportedChromosomeP2
    | ExportedChromosomeP3
    | ExportedChromosomeP4
}

export type ExportedMobileNode = {
  path_segments: Array<[string, string]>
  is_closed: boolean
  is_round_trip: boolean
  speed: number
  time_step: number
  name: string
  source_code: string
}

export type ExportedProblem = {
  name: string
  radius_of_reach: number
  radius_of_inter: number
  radius_of_cover?: number
  region: Region
  sink: [number, number]
  k_required?: number
  targets?: Array<[number, number]>
  candidates?: Array<[number, number]>
  number_of_relays?: number
  mobile_nodes: ExportedMobileNode[]
}

export type ExportedProblemFile = {
  problem: ExportedProblem
}
