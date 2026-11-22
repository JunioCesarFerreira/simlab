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

/** P3's sensing targets, or P4's sensor motes: points the user places. */
export function hasTargets(name: string): boolean {
  return name === 'problem3' || name === 'problem4'
}

/** What the placed points are called: P4's are the motes the sink collects from. */
export function targetsLabel(name: string): string {
  return name === 'problem4' ? 'Sensor motes' : 'Targets'
}

/**
 * P4's scalars. The sink is the base, the candidates are the stops and the
 * targets are the sensor motes; see exportProblem for how they map onto
 * pylib's ProblemP4.
 */
export type P4Settings = {
  initialEnergy: number
  bufferCapacity: number
  /** Data units per second, in the unit of bufferCapacity. */
  dataRate: number
  /** m/s */
  speed: number
  /** s */
  timeStep: number
  /** Stops in a route, the base at both ends included. */
  maxRouteLength: number
  /** Dwell time bounds, s. */
  tauMin: number
  tauMax: number
}

export const DEFAULT_P4: P4Settings = {
  initialEnergy: 1000,
  bufferCapacity: 4096,
  dataRate: 1.61,
  speed: 10,
  timeStep: 1,
  maxRouteLength: 20,
  tauMin: 0,
  tauMax: 200,
}

/**
 * Problems carrying the trajectory coverage constraint (alpha).
 *
 * P1 and P2 require the deployed motes to cover at least `alpha` percent of
 * the sampled mobile-node trajectory points; individuals below the level are
 * penalized by the adapter's `penalty_objectives`. P3 uses k-coverage over
 * explicit targets instead, and P4 has no coverage constraint at all.
 */
export function hasCoverageConstraint(name: string): boolean {
  return name === 'problem1' || name === 'problem2'
}

export type ProblemDraft = {
  name: string
  radiusOfReach: number
  radiusOfInter: number
  radiusOfCover: number
  kRequired: number
  /** P3: minimum neighbours of each deployed mote, the sink included. Absent → 1. */
  gRequired?: number
  /** P4 only. Absent → DEFAULT_P4. */
  p4?: P4Settings
  /** alpha: minimum trajectory coverage level, in percent [0, 100]. P1/P2 only. */
  minCoveragePercentage: number
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
  /** Indices into draft.candidates. The base is implicit at both ends: the
   *  exported route is [0, ...route + 1, 0], sojourn 0 being the base. */
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
  /** alpha: minimum trajectory coverage level, in percent [0, 100]. P1/P2 only. */
  min_coverage_percentage?: number
  targets?: Array<[number, number]>
  candidates?: Array<[number, number]>
  number_of_relays?: number
  mobile_nodes: ExportedMobileNode[]
  /** P3 */
  g_required?: number
  /** P4: see exportProblem */
  nodes?: Array<[number, number]>
  sink_base?: [number, number]
  sojourns?: ExportedSojourn[]
  initial_energy?: number
  buffer_capacity?: number
  data_rate?: number
  speed?: number
  time_step?: number
  max_route_length?: number
  tau_bounds?: [number, number]
}

export type ExportedSojourn = {
  id: number
  position: [number, number]
  adjacency: number[]
  visibleNodes: number[]
}

export type ExportedProblemFile = {
  problem: ExportedProblem
}
