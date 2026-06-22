import { defineStore } from 'pinia'
import { ref, watch } from 'vue'
import { nanoid } from 'nanoid'
import type { ProblemDraft, SinkPoint, MobileRouteDraft, DraftSegment, Region, TargetPoint, ChromosomeDraft, MacProtocol, RelayPoint } from '../../types/problem'
export type { ProblemDraft }
import { saveDraft, loadDraft as loadSavedDraft } from '../../services/persistence'

function createDefaultDraft(): ProblemDraft {
  return {
    name: 'problem2',
    radiusOfReach: 100,
    radiusOfInter: 200,
    radiusOfCover: 90,
    kRequired: 1,
    region: [-150, -150, 150, 150],
    sink: null,
    candidates: [],
    targets: [],
    numSensors: 1,
    mobileNodes: [],
    chromosome: null,
  }
}

function createChromosomeFor(name: string, draft: ProblemDraft): ChromosomeDraft | null {
  if (name === 'problem1') return { kind: 'problem1', macProtocol: 'csma', relays: [] }
  if (name === 'problem2') return { kind: 'problem2', macProtocol: 'csma', mask: draft.candidates.map(() => 0) }
  if (name === 'problem3') return { kind: 'problem3', macProtocol: 'csma', mask: draft.candidates.map(() => 0) }
  if (name === 'problem4') return { kind: 'problem4', macProtocol: 'csma', route: [], sojournTimes: [] }
  return null
}

export const useProblemStore = defineStore('problem', () => {
  const saved = loadSavedDraft()
  const draft = ref<ProblemDraft>(saved ? { ...createDefaultDraft(), ...saved } : createDefaultDraft())

  let saveTimer: ReturnType<typeof setTimeout> | null = null
  watch(draft, (val) => {
    if (saveTimer) clearTimeout(saveTimer)
    saveTimer = setTimeout(() => saveDraft(val), 500)
  }, { deep: true })

  function reset() {
    draft.value = createDefaultDraft()
  }

  function loadDraft(incoming: ProblemDraft) {
    draft.value = incoming
  }

  function updateMeta(fields: Partial<Pick<ProblemDraft, 'name' | 'radiusOfReach' | 'radiusOfInter' | 'radiusOfCover' | 'kRequired' | 'region' | 'numSensors'>>) {
    const nameChanging = fields.name && fields.name !== draft.value.name
    Object.assign(draft.value, fields)
    if (nameChanging) {
      draft.value.chromosome = null
    }
  }

  function setSink(sink: SinkPoint | null) {
    draft.value.sink = sink
  }

  function addCandidate(x: number, y: number) {
    draft.value.candidates.push({ id: nanoid(), x, y })
    const chrom = draft.value.chromosome
    if (chrom && (chrom.kind === 'problem2' || chrom.kind === 'problem3')) {
      chrom.mask.push(0)
    }
  }

  function removeCandidate(id: string) {
    const idx = draft.value.candidates.findIndex(c => c.id === id)
    if (idx < 0) return
    draft.value.candidates.splice(idx, 1)
    const chrom = draft.value.chromosome
    if (chrom && (chrom.kind === 'problem2' || chrom.kind === 'problem3')) {
      chrom.mask.splice(idx, 1)
    } else if (chrom && chrom.kind === 'problem4') {
      const keep: number[] = []
      const keepT: number[] = []
      chrom.route.forEach((r, i) => {
        if (r === idx) return
        keep.push(r > idx ? r - 1 : r)
        keepT.push(chrom.sojournTimes[i] ?? 0)
      })
      chrom.route = keep
      chrom.sojournTimes = keepT
    }
  }

  function moveCandidate(id: string, x: number, y: number) {
    const c = draft.value.candidates.find(c => c.id === id)
    if (c) { c.x = x; c.y = y }
  }

  function addTarget(x: number, y: number) {
    draft.value.targets.push({ id: nanoid(), x, y })
  }

  function removeTarget(id: string) {
    draft.value.targets = draft.value.targets.filter(t => t.id !== id)
  }

  function moveTarget(id: string, x: number, y: number) {
    const t = draft.value.targets.find(t => t.id === id)
    if (t) { t.x = x; t.y = y }
  }

  function addMobileNode(): string {
    const node: MobileRouteDraft = {
      id: nanoid(),
      name: `node${draft.value.mobileNodes.length + 1}`,
      sourceCode: 'node.c',
      speed: 5,
      timeStep: 1,
      isClosed: false,
      isRoundTrip: false,
      segments: [],
    }
    draft.value.mobileNodes.push(node)
    return node.id
  }

  function removeMobileNode(id: string) {
    draft.value.mobileNodes = draft.value.mobileNodes.filter(n => n.id !== id)
  }

  function updateMobileNode(id: string, fields: Partial<Omit<MobileRouteDraft, 'id' | 'segments'>>) {
    const node = draft.value.mobileNodes.find(n => n.id === id)
    if (node) Object.assign(node, fields)
  }

  function addSegmentToNode(nodeId: string, seg: DraftSegment) {
    const node = draft.value.mobileNodes.find(n => n.id === nodeId)
    if (node) node.segments.push(seg)
  }

  function removeSegmentFromNode(nodeId: string, index: number) {
    const node = draft.value.mobileNodes.find(n => n.id === nodeId)
    if (node) node.segments.splice(index, 1)
  }

  function removeLastSegmentFromNode(nodeId: string) {
    const node = draft.value.mobileNodes.find(n => n.id === nodeId)
    if (node && node.segments.length > 0) node.segments.pop()
  }

  function rescaleAll(factor: number) {
    if (!isFinite(factor) || factor <= 0) return
    const d = draft.value
    const f = factor
    draft.value = {
      ...d,
      region: [d.region[0]*f, d.region[1]*f, d.region[2]*f, d.region[3]*f] as Region,
      sink: d.sink ? { x: d.sink.x*f, y: d.sink.y*f } : null,
      candidates: d.candidates.map(c => ({ ...c, x: c.x*f, y: c.y*f })),
      targets: d.targets.map((t: TargetPoint) => ({ ...t, x: t.x*f, y: t.y*f })),
      mobileNodes: d.mobileNodes.map(node => ({
        ...node,
        segments: node.segments.map((seg): DraftSegment => {
          if (seg.type === 'line')
            return { type: 'line', start: [seg.start[0]*f, seg.start[1]*f], end: [seg.end[0]*f, seg.end[1]*f] }
          if (seg.type === 'ellipse')
            return { type: 'ellipse', center: [seg.center[0]*f, seg.center[1]*f], radiusX: seg.radiusX*f, radiusY: seg.radiusY*f }
          return seg
        }),
      })),
    }
  }

  function ensureChromosome() {
    if (!draft.value.chromosome) {
      draft.value.chromosome = createChromosomeFor(draft.value.name, draft.value)
    }
  }

  function clearChromosome() {
    draft.value.chromosome = null
  }

  function setMacProtocol(mac: MacProtocol) {
    ensureChromosome()
    if (draft.value.chromosome) draft.value.chromosome.macProtocol = mac
  }

  function setMaskAt(index: number, value: 0 | 1) {
    const chrom = draft.value.chromosome
    if (!chrom || (chrom.kind !== 'problem2' && chrom.kind !== 'problem3')) return
    if (index < 0 || index >= chrom.mask.length) return
    chrom.mask[index] = value
  }

  function toggleMaskAt(index: number) {
    ensureChromosome()
    const chrom = draft.value.chromosome
    if (!chrom || (chrom.kind !== 'problem2' && chrom.kind !== 'problem3')) return
    if (index < 0 || index >= chrom.mask.length) return
    chrom.mask[index] = chrom.mask[index] ? 0 : 1
  }

  function addRelay(x: number, y: number): string | null {
    ensureChromosome()
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem1') return null
    if (chrom.relays.length >= draft.value.numSensors) return null
    const relay: RelayPoint = { id: nanoid(), x, y }
    chrom.relays.push(relay)
    return relay.id
  }

  function moveRelay(id: string, x: number, y: number) {
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem1') return
    const r = chrom.relays.find(r => r.id === id)
    if (r) { r.x = x; r.y = y }
  }

  function removeRelay(id: string) {
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem1') return
    chrom.relays = chrom.relays.filter(r => r.id !== id)
  }

  function setRoute(route: number[]) {
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem4') return
    chrom.route = route
    if (chrom.sojournTimes.length > route.length) {
      chrom.sojournTimes.length = route.length
    } else {
      while (chrom.sojournTimes.length < route.length) chrom.sojournTimes.push(0)
    }
  }

  function setSojournTimes(times: number[]) {
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem4') return
    chrom.sojournTimes = times
  }

  function appendRouteStop(candidateIndex: number) {
    ensureChromosome()
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem4') return
    chrom.route.push(candidateIndex)
    chrom.sojournTimes.push(0)
  }

  function removeLastRouteStop(candidateIndex?: number) {
    const chrom = draft.value.chromosome
    if (!chrom || chrom.kind !== 'problem4') return
    const target = candidateIndex === undefined
      ? chrom.route.length - 1
      : chrom.route.lastIndexOf(candidateIndex)
    if (target < 0) return
    chrom.route.splice(target, 1)
    chrom.sojournTimes.splice(target, 1)
  }

  return {
    draft,
    reset,
    loadDraft,
    updateMeta,
    setSink,
    addCandidate,
    removeCandidate,
    moveCandidate,
    addTarget,
    removeTarget,
    moveTarget,
    addMobileNode,
    removeMobileNode,
    updateMobileNode,
    addSegmentToNode,
    removeSegmentFromNode,
    removeLastSegmentFromNode,
    rescaleAll,
    ensureChromosome,
    clearChromosome,
    setMacProtocol,
    setMaskAt,
    toggleMaskAt,
    addRelay,
    moveRelay,
    removeRelay,
    setRoute,
    setSojournTimes,
    appendRouteStop,
    removeLastRouteStop,
  }
})
