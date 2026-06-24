<template>
  <div class="canvas-wrapper" ref="wrapperRef" tabindex="0" @keydown.capture="handleKey">
    <canvas
      ref="canvasRef"
      :width="canvasW"
      :height="canvasH"
      :style="{ cursor: canvasCursor }"
      @mousedown="handleMouseDown"
      @mousemove="handleMouseMove"
      @mouseup="handleMouseUp"
      @dblclick="handleDblClick"
      @contextmenu.prevent="handleRightClick"
      @mouseleave="handleMouseLeave"
    />
    <div class="coords" v-if="hover">
      {{ fmtC(hover[0]) }}, {{ fmtC(hover[1]) }}
    </div>
    <div class="tool-hint" :class="{ warn: isToolWarn }">{{ toolHint }}</div>
    <div v-if="calibrationToast" class="calibration-toast">{{ calibrationToast }}</div>
  </div>

  <Teleport to="body">
    <div v-if="scaleCalibrateState?.locked" class="scale-input-overlay">
      <div class="scale-label">Comprimento real do segmento:</div>
      <div class="scale-dist-hint">Distância atual: {{ scaleCalibrateWorldDist.toFixed(2) }} u</div>
      <input
        ref="scaleInputRef"
        v-model="scaleRealLength"
        type="number"
        min="0.001"
        step="any"
        placeholder="ex: 50.0"
        @keydown.enter.stop="applyScaleCalibration"
        @keydown.escape.stop="cancelScaleCalibration"
      />
      <div class="scale-btns">
        <button class="apply" @mousedown.prevent="applyScaleCalibration">Aplicar</button>
        <button class="cancel" @mousedown.prevent="cancelScaleCalibration">Cancelar</button>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useEditorStore } from '../../../app/stores/editorStore'
import { sampleCustomSegment } from '../../../services/expressionEvaluator'
import { useTheme } from '../../../composables/useTheme'
import type { Region } from '../../../types/problem'

const problemStore = useProblemStore()
const editorStore = useEditorStore()
const { isDark } = useTheme()

/* Canvas colour palette — recomputed each draw() call so theme changes take effect immediately */
interface CanvasColors {
  canvasBg: string; grid: string; gridLabel: string; axis: string
  regionActive: string; handleBorder: string; surface: string
  nodeStroke: string; nodeStrokeEmph: string; connectivity: string
  measureBg: string; measureBorder: string; measureText: string
  scaleLabelText: string; dimText: string; ellipseAxis: string
}
let C: CanvasColors = buildColors(false)

function buildColors(dark: boolean): CanvasColors {
  return {
    canvasBg:       dark ? '#181825' : '#f0f4f9',
    grid:           dark ? '#313244' : '#c8d3e0',
    gridLabel:      dark ? '#6c7086' : '#64748b',
    axis:           dark ? '#45475a' : '#94a3b8',
    regionActive:   dark ? '#cdd6f4' : '#111827',
    handleBorder:   dark ? '#11111b' : '#f8fafc',
    surface:        dark ? '#1e1e2e' : '#ffffff',
    nodeStroke:     dark ? '#181825' : '#ffffff',
    nodeStrokeEmph: dark ? '#cdd6f4' : '#111827',
    connectivity:   dark ? 'rgba(137,180,250,0.30)' : '#bfdbfe',
    measureBg:      dark ? 'rgba(30,30,46,0.95)' : 'rgba(255,255,255,0.95)',
    measureBorder:  dark ? '#313244' : '#e5e7eb',
    measureText:    dark ? '#f9e2af' : '#92400e',
    scaleLabelText: dark ? '#89b4fa' : '#1d4ed8',
    dimText:        dark ? '#6c7086' : '#6b7280',
    ellipseAxis:    dark ? '#45475a' : '#d1d5db',
  }
}

const canvasRef = ref<HTMLCanvasElement | null>(null)
const wrapperRef = ref<HTMLDivElement | null>(null)
const canvasW = ref(800)
const canvasH = ref(600)
const hover = ref<[number, number] | null>(null)

const polylinePoints = ref<[number, number][]>([])
const ellipseDrag = ref<{ center: [number, number]; current: [number, number] } | null>(null)
const dragState = ref<{
  type: 'sink' | 'candidate' | 'target' | 'relay'
  id?: string
  downCanvas: [number, number]
  originalPos: [number, number]
  moved: boolean
} | null>(null)

let suppressNextClick = false

const scaleCalibrateState = ref<{
  anchor: [number, number]
  current: [number, number]
  locked: boolean
} | null>(null)

const scaleRealLength = ref<string>('')
const scaleInputRef = ref<HTMLInputElement | null>(null)

const scaleCalibrateWorldDist = computed(() => {
  const ms = scaleCalibrateState.value
  if (!ms) return 0
  return Math.hypot(ms.current[0] - ms.anchor[0], ms.current[1] - ms.anchor[1])
})

watch(() => scaleCalibrateState.value?.locked, (locked) => {
  if (locked) nextTick(() => scaleInputRef.value?.focus())
})

const calibrationToast = ref<string | null>(null)
let toastTimer: ReturnType<typeof setTimeout> | null = null

function applyScaleCalibration() {
  const ms = scaleCalibrateState.value
  if (!ms?.locked) return
  const raw = scaleInputRef.value?.value ?? scaleRealLength.value
  const realLen = parseFloat(raw)
  if (!isFinite(realLen) || realLen <= 0) return
  const worldDist = scaleCalibrateWorldDist.value
  if (worldDist < 0.001) return
  const factor = realLen / worldDist
  problemStore.rescaleAll(factor)
  const ib = editorStore.imageWorldBounds
  if (ib) editorStore.setImageWorldBounds([ib[0]*factor, ib[1]*factor, ib[2]*factor, ib[3]*factor])
  scaleCalibrateState.value = null
  scaleRealLength.value = ''
  if (toastTimer) clearTimeout(toastTimer)
  calibrationToast.value = `Escala aplicada — fator ${factor.toFixed(4)}×  |  nova região: [${problemStore.draft.region.map(v => v.toFixed(1)).join(', ')}]`
  toastTimer = setTimeout(() => { calibrationToast.value = null }, 4000)
}

function cancelScaleCalibration() {
  scaleCalibrateState.value = null
  scaleRealLength.value = ''
}

const measureState = ref<{
  anchor: [number, number]
  current: [number, number]
  locked: boolean
} | null>(null)

type RegionHandle = 'tl' | 't' | 'tr' | 'r' | 'br' | 'b' | 'bl' | 'l'

const HANDLE_CURSORS: Record<RegionHandle, string> = {
  tl: 'nwse-resize', tr: 'nesw-resize', bl: 'nesw-resize', br: 'nwse-resize',
  t: 'ns-resize',   b: 'ns-resize',
  l: 'ew-resize',   r: 'ew-resize',
}

const regionDrag = ref<{ handle: RegionHandle } | null>(null)
const hoveredHandle = ref<RegionHandle | null>(null)

function getRegionHandlePositions(): Record<RegionHandle, [number, number]> {
  const [xmin, ymin, xmax, ymax] = problemStore.draft.region
  const xmid = (xmin + xmax) / 2
  const ymid = (ymin + ymax) / 2
  return {
    tl: worldToCanvas(xmin, ymax), t: worldToCanvas(xmid, ymax), tr: worldToCanvas(xmax, ymax),
    l:  worldToCanvas(xmin, ymid),                                 r:  worldToCanvas(xmax, ymid),
    bl: worldToCanvas(xmin, ymin), b: worldToCanvas(xmid, ymin),  br: worldToCanvas(xmax, ymin),
  }
}

function hitTestRegionHandle(cx: number, cy: number): RegionHandle | null {
  const pos = getRegionHandlePositions()
  for (const [h, [px, py]] of Object.entries(pos) as [RegionHandle, [number, number]][]) {
    if (Math.hypot(cx - px, cy - py) <= 8) return h
  }
  return null
}

function applyRegionDrag(handle: RegionHandle, wx: number, wy: number) {
  const r = [...problemStore.draft.region] as [number, number, number, number]
  const MIN = 10
  if (handle === 'l' || handle === 'tl' || handle === 'bl') r[0] = Math.min(wx, r[2] - MIN)
  if (handle === 'r' || handle === 'tr' || handle === 'br') r[2] = Math.max(wx, r[0] + MIN)
  if (handle === 'b' || handle === 'bl' || handle === 'br') r[1] = Math.min(wy, r[3] - MIN)
  if (handle === 't' || handle === 'tl' || handle === 'tr') r[3] = Math.max(wy, r[1] + MIN)
  problemStore.updateMeta({ region: r })
}

let bgImage: HTMLImageElement | null = null
watch(() => editorStore.backgroundImage, (src) => {
  if (!src) { bgImage = null; draw(); return }
  const img = new Image()
  img.onload = () => { bgImage = img; draw() }
  img.src = src
}, { immediate: true })

let ro: ResizeObserver | null = null
onMounted(() => {
  ro = new ResizeObserver(entries => {
    for (const e of entries) {
      canvasW.value = Math.floor(e.contentRect.width)
      canvasH.value = Math.floor(e.contentRect.height)
    }
  })
  if (wrapperRef.value) ro.observe(wrapperRef.value)
})
onUnmounted(() => ro?.disconnect())

const region = computed<Region>(() => problemStore.draft.region)
const MARGIN = 48

const viewport = computed(() => {
  const [xmin, ymin, xmax, ymax] = region.value
  const worldW = xmax - xmin
  const worldH = ymax - ymin
  const availW = canvasW.value - MARGIN * 2
  const availH = canvasH.value - MARGIN * 2
  const scale = Math.min(availW / worldW, availH / worldH)
  const vw = worldW * scale
  const vh = worldH * scale
  const offX = MARGIN + (availW - vw) / 2
  const offY = MARGIN + (availH - vh) / 2
  return { xmin, ymin, xmax, ymax, scale, offX, offY, vw, vh }
})

function worldToCanvas(wx: number, wy: number): [number, number] {
  const { xmin, ymax, scale, offX, offY } = viewport.value
  return [
    offX + (wx - xmin) * scale,
    offY + (ymax - wy) * scale,
  ]
}

function canvasToWorld(cx: number, cy: number): [number, number] {
  const { xmin, ymax, scale, offX, offY } = viewport.value
  return [
    xmin + (cx - offX) / scale,
    ymax - (cy - offY) / scale,
  ]
}

function fmtC(v: number) { return v.toFixed(1) }
function snap(v: number) { return Math.round(v) }

const HIT_PX = 10

function hitSink(cx: number, cy: number): boolean {
  const sink = problemStore.draft.sink
  if (!sink) return false
  const [px, py] = worldToCanvas(sink.x, sink.y)
  return Math.hypot(cx - px, cy - py) <= HIT_PX
}

function hitCandidate(cx: number, cy: number): string | null {
  for (const c of problemStore.draft.candidates) {
    const [px, py] = worldToCanvas(c.x, c.y)
    if (Math.hypot(cx - px, cy - py) <= HIT_PX) return c.id
  }
  return null
}

function hitTarget(cx: number, cy: number): string | null {
  for (const t of problemStore.draft.targets) {
    const [px, py] = worldToCanvas(t.x, t.y)
    if (Math.hypot(cx - px, cy - py) <= HIT_PX) return t.id
  }
  return null
}

function hitRelay(cx: number, cy: number): string | null {
  const chrom = problemStore.draft.chromosome
  if (!chrom || chrom.kind !== 'problem1') return null
  for (const r of chrom.relays) {
    const [px, py] = worldToCanvas(r.x, r.y)
    if (Math.hypot(cx - px, cy - py) <= HIT_PX) return r.id
  }
  return null
}

const canvasCursor = computed(() => {
  if (regionDrag.value) return HANDLE_CURSORS[regionDrag.value.handle]
  if (hoveredHandle.value) return HANDLE_CURSORS[hoveredHandle.value]
  const t = editorStore.activeTool
  if (t === 'select') return dragState.value?.moved ? 'grabbing' : 'default'
  if (t === 'measure') return 'crosshair'
  if (t === 'chromosome-pick') return 'pointer'
  return 'crosshair'
})

const isToolWarn = computed(() => {
  const t = editorStore.activeTool
  if (['draw-line', 'draw-ellipse'].includes(t) && !editorStore.activeNodeId) return true
  if (t === 'place-relay') {
    const chrom = problemStore.draft.chromosome
    const n = (chrom && chrom.kind === 'problem1') ? chrom.relays.length : 0
    return n >= problemStore.draft.numSensors
  }
  return false
})

const toolHint = computed(() => {
  const t = editorStore.activeTool
  if (t === 'draw-line') {
    if (!editorStore.activeNodeId) return '⚠ Selecione um nó móvel na aba Nodes antes de desenhar'
    const n = polylinePoints.value.length
    return n === 0
      ? 'Clique para iniciar polilinha  ·  Esc cancela'
      : `${n} ponto(s)  ·  clique para continuar  ·  duplo-clique ou clique direito para finalizar`
  }
  if (t === 'draw-ellipse') {
    if (!editorStore.activeNodeId) return '⚠ Selecione um nó móvel na aba Nodes antes de desenhar'
    return ellipseDrag.value ? 'Solte para confirmar a elipse' : 'Arraste para definir centro e raios da elipse'
  }
  if (t === 'measure') {
    if (!measureState.value) return 'Arraste para medir distância  [M]  ·  Esc cancela'
    if (!measureState.value.locked) return 'Solte para fixar a medição  ·  Esc cancela'
    const d = measureDistance()
    return `Distância: ${d.toFixed(1)} u  ·  clique ou Esc para limpar`
  }
  if (t === 'scale-calibrate') {
    if (!scaleCalibrateState.value) return 'Arraste um segmento de comprimento conhecido  [R]  ·  Esc cancela'
    if (!scaleCalibrateState.value.locked) return 'Solte para fixar o segmento  ·  Esc cancela'
    return 'Digite o comprimento real e pressione Enter para recalibrar a escala'
  }
  if (t === 'place-relay') {
    const chrom = problemStore.draft.chromosome
    const n = (chrom && chrom.kind === 'problem1') ? chrom.relays.length : 0
    const lim = problemStore.draft.numSensors
    if (n >= lim) return `⚠ Limite atingido (${n}/${lim}) — ajuste "Number of Sensors"`
    return `Clique para adicionar relay  [N]  ·  ${n}/${lim}  ·  clique direito para remover`
  }
  if (t === 'chromosome-pick') {
    const name = problemStore.draft.name
    const chrom = problemStore.draft.chromosome
    if (name === 'problem2' || name === 'problem3') {
      const n = (chrom && (chrom.kind === 'problem2' || chrom.kind === 'problem3')) ? chrom.mask.filter(b => b === 1).length : 0
      return `Clique no candidato para ativar/desativar  [X]  ·  ${n} ativo(s)  ·  clique direito desativa`
    }
    if (name === 'problem4') {
      const n = (chrom && chrom.kind === 'problem4') ? chrom.route.length : 0
      return `Clique no candidato para adicionar à rota  [X]  ·  ${n} stop(s)  ·  clique direito remove último`
    }
    return 'Selecione um problema p2/p3/p4 para usar esta ferramenta'
  }
  const hints: Record<string, string> = {
    'select': 'Clique para selecionar · arraste para mover · Del para remover · clique direito para remover',
    'place-sink': 'Clique para posicionar o sink  [K]',
    'place-candidate': 'Clique para adicionar candidato  [C]  ·  clique direito para remover',
    'place-target': 'Clique para adicionar alvo  [T]  ·  clique direito para remover',
  }
  return hints[t] ?? ''
})

function measureDistance(): number {
  if (!measureState.value) return 0
  const [ax, ay] = measureState.value.anchor
  const [bx, by] = measureState.value.current
  return Math.hypot(bx - ax, by - ay)
}

function draw() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')!
  const { vw, vh, offX, offY } = viewport.value

  C = buildColors(isDark.value)
  ctx.fillStyle = C.canvasBg
  ctx.fillRect(0, 0, canvasW.value, canvasH.value)

  if (bgImage) {
    const calBounds = editorStore.imageWorldBounds
    let imgX: number, imgY: number, imgW: number, imgH: number

    if (calBounds) {
      const [ix1, iy1] = worldToCanvas(calBounds[0], calBounds[3])
      const [ix2, iy2] = worldToCanvas(calBounds[2], calBounds[1])
      imgX = ix1; imgY = iy1; imgW = ix2 - ix1; imgH = iy2 - iy1
    } else {
      const imgAspect = bgImage.naturalWidth / bgImage.naturalHeight
      const vpAspect = vw / vh
      if (imgAspect > vpAspect) {
        imgW = vw; imgH = vw / imgAspect; imgX = offX; imgY = offY + (vh - imgH) / 2
      } else {
        imgH = vh; imgW = vh * imgAspect; imgX = offX + (vw - imgW) / 2; imgY = offY
      }
    }

    ctx.save()
    ctx.globalAlpha = 0.55
    ctx.drawImage(bgImage, imgX, imgY, imgW, imgH)
    ctx.restore()
  }

  drawGrid(ctx)
  drawRegionBorder(ctx)
  drawConnectivityGraph(ctx)
  drawChromosomeConnectivityGraph(ctx)
  drawP4Route(ctx)
  drawMobileRoutes(ctx)
  drawTargets(ctx)
  drawCandidates(ctx)
  drawRelays(ctx)
  drawSink(ctx)
  drawPolylinePreview(ctx)
  drawEllipsePreview(ctx)
  drawMeasure(ctx)
  drawScaleCalibrate(ctx)
}

type NetNode = { x: number; y: number }

function drawReachGraph(ctx: CanvasRenderingContext2D, nodes: NetNode[], color: string, lineWidth: number) {
  if (nodes.length === 0) return
  const { radiusOfReach } = problemStore.draft
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const d = Math.hypot(nodes[j]!.x - nodes[i]!.x, nodes[j]!.y - nodes[i]!.y)
      if (d <= radiusOfReach) {
        const [ax, ay] = worldToCanvas(nodes[i]!.x, nodes[i]!.y)
        const [bx, by] = worldToCanvas(nodes[j]!.x, nodes[j]!.y)
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke()
      }
    }
  }
}

function drawConnectivityGraph(ctx: CanvasRenderingContext2D) {
  if (!editorStore.showConnectivity) return
  const nodes: NetNode[] = []
  if (problemStore.draft.sink) nodes.push(problemStore.draft.sink)
  for (const c of problemStore.draft.candidates) nodes.push({ x: c.x, y: c.y })
  drawReachGraph(ctx, nodes, C.connectivity, 1.5)
}

function drawChromosomeConnectivityGraph(ctx: CanvasRenderingContext2D) {
  if (!editorStore.showChromosomeConnectivity) return
  const chrom = problemStore.draft.chromosome
  if (!chrom) return
  const nodes: NetNode[] = []
  if (problemStore.draft.sink) nodes.push(problemStore.draft.sink)

  if (chrom.kind === 'problem1') {
    for (const r of chrom.relays) nodes.push({ x: r.x, y: r.y })
  } else if (chrom.kind === 'problem2' || chrom.kind === 'problem3') {
    const cands = problemStore.draft.candidates
    chrom.mask.forEach((bit, i) => {
      if (bit === 1 && i < cands.length) nodes.push({ x: cands[i]!.x, y: cands[i]!.y })
    })
  } else if (chrom.kind === 'problem4') {
    const cands = problemStore.draft.candidates
    const seen = new Set<number>()
    for (const idx of chrom.route) {
      if (idx < 0 || idx >= cands.length || seen.has(idx)) continue
      seen.add(idx)
      nodes.push({ x: cands[idx]!.x, y: cands[idx]!.y })
    }
  }
  drawReachGraph(ctx, nodes, '#8b5cf6', 2)
}

function drawGrid(ctx: CanvasRenderingContext2D) {
  const { xmin, ymin, xmax, ymax, scale } = viewport.value
  const span = xmax - xmin
  const rawStep = span / 8
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const step = [1, 2, 5, 10].map(f => f * mag).find(s => span / s <= 10) ?? mag * 10

  ctx.strokeStyle = C.grid
  ctx.lineWidth = 0.5
  ctx.fillStyle = C.gridLabel
  ctx.font = `${Math.max(9, Math.min(11, scale * 8))}px monospace`
  ctx.textAlign = 'center'

  for (let x = Math.ceil(xmin / step) * step; x <= xmax + 1e-9; x += step) {
    const [px] = worldToCanvas(x, 0)
    const [, py1] = worldToCanvas(0, ymin)
    const [, py2] = worldToCanvas(0, ymax)
    ctx.beginPath(); ctx.moveTo(px, py1); ctx.lineTo(px, py2); ctx.stroke()
    const [, labelY] = worldToCanvas(0, ymin)
    ctx.fillText(String(Math.round(x)), px, labelY + 13)
  }

  ctx.textAlign = 'right'
  for (let y = Math.ceil(ymin / step) * step; y <= ymax + 1e-9; y += step) {
    const [, py] = worldToCanvas(0, y)
    const [px1] = worldToCanvas(xmin, 0)
    const [px2] = worldToCanvas(xmax, 0)
    ctx.beginPath(); ctx.moveTo(px1, py); ctx.lineTo(px2, py); ctx.stroke()
    const [labelX] = worldToCanvas(xmin, 0)
    ctx.fillText(String(Math.round(y)), labelX - 4, py + 3)
  }

  ctx.strokeStyle = C.axis
  ctx.lineWidth = 1
  if (xmin <= 0 && 0 <= xmax) {
    const [px] = worldToCanvas(0, 0)
    const [, py1] = worldToCanvas(0, ymin)
    const [, py2] = worldToCanvas(0, ymax)
    ctx.beginPath(); ctx.moveTo(px, py1); ctx.lineTo(px, py2); ctx.stroke()
  }
  if (ymin <= 0 && 0 <= ymax) {
    const [, py] = worldToCanvas(0, 0)
    const [px1] = worldToCanvas(xmin, 0)
    const [px2] = worldToCanvas(xmax, 0)
    ctx.beginPath(); ctx.moveTo(px1, py); ctx.lineTo(px2, py); ctx.stroke()
  }
}

function drawRegionBorder(ctx: CanvasRenderingContext2D) {
  const { xmin, ymin, xmax, ymax } = viewport.value
  const [px1, py1] = worldToCanvas(xmin, ymax)
  const [px2, py2] = worldToCanvas(xmax, ymin)
  const active = !!regionDrag.value || !!hoveredHandle.value
  ctx.strokeStyle = active ? C.regionActive : '#3b82f6'
  ctx.lineWidth = active ? 2 : 1.5
  ctx.setLineDash([5, 4])
  ctx.strokeRect(px1, py1, px2 - px1, py2 - py1)
  ctx.setLineDash([])

  const pos = getRegionHandlePositions()
  for (const [h, [hx, hy]] of Object.entries(pos) as [RegionHandle, [number, number]][]) {
    const isActive = regionDrag.value?.handle === h || hoveredHandle.value === h
    ctx.beginPath()
    ctx.rect(hx - 4, hy - 4, 8, 8)
    ctx.fillStyle = isActive ? C.regionActive : '#3b82f6'
    ctx.fill()
    ctx.strokeStyle = C.handleBorder
    ctx.lineWidth = 1
    ctx.stroke()
  }
}

function drawTargets(ctx: CanvasRenderingContext2D) {
  const sel = editorStore.selected
  for (const t of problemStore.draft.targets) {
    const [px, py] = worldToCanvas(t.x, t.y)
    const selected = sel?.type === 'target' && sel.id === t.id
    const s = selected ? 7 : 5
    ctx.beginPath()
    ctx.moveTo(px, py - s); ctx.lineTo(px + s, py); ctx.lineTo(px, py + s); ctx.lineTo(px - s, py)
    ctx.closePath()
    ctx.fillStyle = '#f59e0b'
    ctx.fill()
    ctx.strokeStyle = selected ? '#8b5cf6' : C.surface
    ctx.lineWidth = selected ? 2.5 : 1
    ctx.stroke()
  }
}

function drawCandidates(ctx: CanvasRenderingContext2D) {
  const sel = editorStore.selected
  const chrom = problemStore.draft.chromosome
  const hasMask = chrom && (chrom.kind === 'problem2' || chrom.kind === 'problem3')
  const mask = hasMask ? chrom.mask : null
  for (let i = 0; i < problemStore.draft.candidates.length; i++) {
    const c = problemStore.draft.candidates[i]!
    const [px, py] = worldToCanvas(c.x, c.y)
    const selected = sel?.type === 'candidate' && sel.id === c.id
    const active = !!(mask && mask[i] === 1)
    ctx.beginPath()
    ctx.arc(px, py, selected ? 7 : active ? 6 : 5, 0, Math.PI * 2)
    ctx.fillStyle = active ? 'var(--color-primary)' : '#10b981'
    ctx.fill()
    ctx.strokeStyle = selected ? '#f59e0b' : active ? C.nodeStrokeEmph : C.nodeStroke
    ctx.lineWidth = selected ? 2.5 : active ? 2 : 1
    ctx.stroke()
    if (active) {
      ctx.beginPath()
      ctx.arc(px, py, 10, 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(59,130,246,0.5)'
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
  }
}

function drawRelays(ctx: CanvasRenderingContext2D) {
  const chrom = problemStore.draft.chromosome
  if (!chrom || chrom.kind !== 'problem1') return
  const sel = editorStore.selected
  for (const r of chrom.relays) {
    const [px, py] = worldToCanvas(r.x, r.y)
    const selected = sel?.type === 'relay' && sel.id === r.id
    const size = selected ? 7 : 5.5
    ctx.beginPath()
    ctx.moveTo(px, py - size); ctx.lineTo(px + size, py); ctx.lineTo(px, py + size); ctx.lineTo(px - size, py)
    ctx.closePath()
    ctx.fillStyle = '#3b82f6'
    ctx.fill()
    ctx.strokeStyle = selected ? '#f59e0b' : C.nodeStrokeEmph
    ctx.lineWidth = selected ? 2.5 : 1.5
    ctx.stroke()
    ctx.beginPath(); ctx.arc(px, py, 1.5, 0, Math.PI * 2)
    ctx.fillStyle = C.nodeStroke; ctx.fill()
  }
}

function drawP4Route(ctx: CanvasRenderingContext2D) {
  const chrom = problemStore.draft.chromosome
  if (!chrom || chrom.kind !== 'problem4') return
  const cands = problemStore.draft.candidates
  const pts: [number, number][] = []
  for (const idx of chrom.route) {
    if (idx < 0 || idx >= cands.length) continue
    pts.push(worldToCanvas(cands[idx]!.x, cands[idx]!.y))
  }
  if (pts.length < 1) return
  ctx.save()
  ctx.strokeStyle = '#8b5cf6'; ctx.lineWidth = 2; ctx.setLineDash([8, 4])
  ctx.beginPath(); ctx.moveTo(pts[0]![0], pts[0]![1])
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i]![0], pts[i]![1])
  ctx.stroke(); ctx.setLineDash([])
  drawArrowsAlongPath(ctx, pts, '#8b5cf6', 2)
  ctx.font = 'bold 10px monospace'; ctx.textAlign = 'center'
  pts.forEach(([x, y], i) => {
    ctx.fillStyle = '#8b5cf6'; ctx.beginPath(); ctx.arc(x + 10, y - 10, 8, 0, Math.PI * 2); ctx.fill()
    ctx.fillStyle = '#ffffff'; ctx.fillText(String(i + 1), x + 10, y - 7)
  })
  ctx.restore()
}

function drawSink(ctx: CanvasRenderingContext2D) {
  const sink = problemStore.draft.sink
  if (!sink) return
  const [px, py] = worldToCanvas(sink.x, sink.y)
  const selected = editorStore.selected?.type === 'sink'
  const r = selected ? 11 : 9
  ctx.strokeStyle = selected ? '#f59e0b' : '#ef4444'
  ctx.lineWidth = selected ? 3 : 2.5
  ctx.beginPath(); ctx.moveTo(px - r, py); ctx.lineTo(px + r, py); ctx.stroke()
  ctx.beginPath(); ctx.moveTo(px, py - r); ctx.lineTo(px, py + r); ctx.stroke()
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke()
}

const ROUTE_COLORS = ['#2563eb', '#0891b2', '#d97706', '#16a34a', '#9333ea', '#dc2626']
const ROUTE_DASHES = [[10, 5], [6, 4], [14, 4, 3, 4], [4, 4], [12, 3, 3, 3], [8, 6]]
const ARROW_SPACING = 60

function drawMobileRoutes(ctx: CanvasRenderingContext2D) {
  problemStore.draft.mobileNodes.forEach((node, ni) => {
    const color = ROUTE_COLORS[ni % ROUTE_COLORS.length]!
    const dash  = ROUTE_DASHES[ni % ROUTE_DASHES.length]!
    const isActive = editorStore.activeNodeId === node.id
    const lw = isActive ? 3 : 2

    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash)

    for (const seg of node.segments) {
      if (seg.type === 'line') {
        const [px1, py1] = worldToCanvas(seg.start[0], seg.start[1])
        const [px2, py2] = worldToCanvas(seg.end[0], seg.end[1])
        ctx.beginPath(); ctx.moveTo(px1, py1); ctx.lineTo(px2, py2); ctx.stroke()
        ctx.setLineDash([])
        drawArrowsAlongLine(ctx, px1, py1, px2, py2, color, lw)
        ctx.setLineDash(dash); ctx.strokeStyle = color
      } else if (seg.type === 'ellipse') {
        const [cpx, cpy] = worldToCanvas(seg.center[0], seg.center[1])
        const [ex] = worldToCanvas(seg.center[0] + seg.radiusX, seg.center[1])
        const [, ey] = worldToCanvas(seg.center[0], seg.center[1] + seg.radiusY)
        const rx = Math.abs(ex - cpx); const ry = Math.abs(ey - cpy)
        ctx.beginPath(); ctx.ellipse(cpx, cpy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke()
        ctx.setLineDash([])
        drawArrowsAlongEllipse(ctx, cpx, cpy, rx, ry, color, lw)
        ctx.setLineDash(dash); ctx.strokeStyle = color
        ctx.beginPath(); ctx.arc(cpx, cpy, 3, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill()
      } else if (seg.type === 'custom') {
        const pts = sampleCustomSegment(seg.exprX, seg.exprY, 80)
        if (pts && pts.length > 1) {
          const canvasPts = pts.map(([wx, wy]) => worldToCanvas(wx, wy))
          ctx.beginPath(); ctx.moveTo(canvasPts[0]![0], canvasPts[0]![1])
          for (let i = 1; i < canvasPts.length; i++) ctx.lineTo(canvasPts[i]![0], canvasPts[i]![1])
          ctx.stroke(); ctx.setLineDash([])
          drawArrowsAlongPath(ctx, canvasPts, color, lw)
          ctx.setLineDash(dash); ctx.strokeStyle = color
        } else {
          ctx.setLineDash([])
          ctx.fillStyle = '#ef4444'; ctx.font = '11px monospace'; ctx.textAlign = 'left'
          ctx.fillText('⚠ expr', viewport.value.offX + 4, viewport.value.offY + 14)
          ctx.setLineDash(dash)
        }
      }
    }

    ctx.setLineDash([])
    if (node.segments.length > 0) {
      const s = node.segments[0]!
      const lx = s.type === 'line' ? s.start[0] : s.type === 'ellipse' ? s.center[0] : 0
      const ly = s.type === 'line' ? s.start[1] : s.type === 'ellipse' ? s.center[1] : 0
      const [lpx, lpy] = worldToCanvas(lx, ly)
      ctx.fillStyle = color; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'left'
      ctx.fillText(node.name, lpx + 8, lpy - 8)
    }
  })
}

function drawArrowHead(ctx: CanvasRenderingContext2D, ax: number, ay: number, angle: number, color: string, size: number) {
  ctx.fillStyle = color
  ctx.beginPath()
  ctx.moveTo(ax + Math.cos(angle) * size,              ay + Math.sin(angle) * size)
  ctx.lineTo(ax + Math.cos(angle + 2.4) * size * 0.55, ay + Math.sin(angle + 2.4) * size * 0.55)
  ctx.lineTo(ax + Math.cos(angle - 2.4) * size * 0.55, ay + Math.sin(angle - 2.4) * size * 0.55)
  ctx.closePath(); ctx.fill()
}

function drawArrowsAlongLine(ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, color: string, lw: number) {
  const len = Math.hypot(x2 - x1, y2 - y1)
  if (len < ARROW_SPACING) return
  const angle = Math.atan2(y2 - y1, x2 - x1)
  const count = Math.floor(len / ARROW_SPACING)
  const step = len / (count + 1)
  for (let i = 1; i <= count; i++) {
    const t = (i * step) / len
    drawArrowHead(ctx, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, angle, color, 7 + lw)
  }
}

function drawArrowsAlongEllipse(ctx: CanvasRenderingContext2D, cx: number, cy: number, rx: number, ry: number, color: string, lw: number) {
  const a = Math.max(rx, ry), b = Math.min(rx, ry)
  const h = ((a - b) / (a + b)) ** 2
  const perim = Math.PI * (a + b) * (1 + 3 * h / (10 + Math.sqrt(4 - 3 * h)))
  if (perim < ARROW_SPACING) return
  const count = Math.max(1, Math.floor(perim / ARROW_SPACING))
  for (let i = 0; i < count; i++) {
    const t = (i + 0.5) / count
    const angle = 2 * Math.PI * t
    const px = cx + rx * Math.cos(angle)
    const py = cy + ry * Math.sin(angle)
    const tangent = Math.atan2(ry * Math.cos(angle), -rx * Math.sin(angle))
    drawArrowHead(ctx, px, py, tangent, color, 7 + lw)
  }
}

function drawArrowsAlongPath(ctx: CanvasRenderingContext2D, pts: [number, number][], color: string, lw: number) {
  const cumLen: number[] = [0]
  for (let i = 1; i < pts.length; i++) {
    cumLen.push(cumLen[i - 1]! + Math.hypot(pts[i]![0] - pts[i - 1]![0], pts[i]![1] - pts[i - 1]![1]))
  }
  const total = cumLen[cumLen.length - 1]!
  if (total < ARROW_SPACING) return
  const count = Math.floor(total / ARROW_SPACING)
  for (let k = 1; k <= count; k++) {
    const target = (k / (count + 1)) * total
    let i = 1
    while (i < cumLen.length - 1 && cumLen[i]! < target) i++
    const segLen = cumLen[i]! - cumLen[i - 1]!
    const t = segLen > 0 ? (target - cumLen[i - 1]!) / segLen : 0
    const px = pts[i - 1]![0] + (pts[i]![0] - pts[i - 1]![0]) * t
    const py = pts[i - 1]![1] + (pts[i]![1] - pts[i - 1]![1]) * t
    const angle = Math.atan2(pts[i]![1] - pts[i - 1]![1], pts[i]![0] - pts[i - 1]![0])
    drawArrowHead(ctx, px, py, angle, color, 7 + lw)
  }
}

function drawPolylinePreview(ctx: CanvasRenderingContext2D) {
  const pts = polylinePoints.value
  if (pts.length === 0) return
  ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5
  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = worldToCanvas(pts[i]![0], pts[i]![1])
    const [x2, y2] = worldToCanvas(pts[i + 1]![0], pts[i + 1]![1])
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke()
  }
  if (hover.value) {
    const [x1, y1] = worldToCanvas(pts[pts.length - 1]![0], pts[pts.length - 1]![1])
    const [x2, y2] = worldToCanvas(hover.value[0], hover.value[1])
    ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.setLineDash([])
  }
  for (const pt of pts) {
    const [px, py] = worldToCanvas(pt[0], pt[1])
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fillStyle = '#3b82f6'; ctx.fill()
  }
}

function drawEllipsePreview(ctx: CanvasRenderingContext2D) {
  const drag = ellipseDrag.value
  if (!drag) return
  const [cpx, cpy] = worldToCanvas(drag.center[0], drag.center[1])
  const [ex] = worldToCanvas(drag.current[0], drag.center[1])
  const [, ey] = worldToCanvas(drag.center[0], drag.current[1])
  const rx = Math.abs(ex - cpx); const ry = Math.abs(ey - cpy)
  ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5
  if (rx > 1 && ry > 1) {
    ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.ellipse(cpx, cpy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([])
    ctx.strokeStyle = C.ellipseAxis; ctx.lineWidth = 0.5
    ctx.beginPath(); ctx.moveTo(cpx, cpy); ctx.lineTo(ex, cpy); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(cpx, cpy); ctx.lineTo(cpx, ey); ctx.stroke()
    const wx = Math.abs(drag.current[0] - drag.center[0])
    const wy = Math.abs(drag.current[1] - drag.center[1])
    ctx.fillStyle = C.dimText; ctx.font = '10px monospace'; ctx.textAlign = 'center'
    ctx.fillText(`rx=${Math.round(wx)}`, (cpx + ex) / 2, cpy - 5)
    ctx.textAlign = 'left'; ctx.fillText(`ry=${Math.round(wy)}`, cpx + 4, (cpy + ey) / 2)
  }
  ctx.beginPath(); ctx.arc(cpx, cpy, 3, 0, Math.PI * 2); ctx.fillStyle = '#3b82f6'; ctx.fill()
}

function drawMeasure(ctx: CanvasRenderingContext2D) {
  const ms = measureState.value
  const current = ms?.locked ? ms.current : (hover.value ?? ms?.current ?? null)
  const liveAnchor = ms?.anchor ?? null

  if (!liveAnchor || !current) {
    if (editorStore.activeTool === 'measure' && hover.value) {
      const [hx, hy] = worldToCanvas(hover.value[0], hover.value[1])
      ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI * 2)
      ctx.strokeStyle = '#d97706'; ctx.lineWidth = 1.5; ctx.stroke()
    }
    return
  }

  const [ax, ay] = worldToCanvas(liveAnchor[0], liveAnchor[1])
  const [bx, by] = worldToCanvas(current[0], current[1])
  const dist = Math.hypot(current[0] - liveAnchor[0], current[1] - liveAnchor[1])
  const locked = ms?.locked ?? false

  ctx.strokeStyle = '#d97706'; ctx.lineWidth = locked ? 2 : 1.5
  ctx.setLineDash(locked ? [] : [6, 4])
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([])

  const len = Math.hypot(bx - ax, by - ay)
  if (len > 4) {
    const nx = -(by - ay) / len * 8; const ny = (bx - ax) / len * 8
    ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(ax + nx, ay + ny); ctx.lineTo(ax - nx, ay - ny); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(bx + nx, by + ny); ctx.lineTo(bx - nx, by - ny); ctx.stroke()
  }

  ctx.beginPath(); ctx.arc(ax, ay, 4, 0, Math.PI * 2); ctx.fillStyle = '#d97706'; ctx.fill()
  ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fillStyle = locked ? '#d97706' : C.nodeStroke; ctx.fill()

  const midX = (ax + bx) / 2; const midY = (ay + by) / 2
  const label = `${dist.toFixed(1)} u`
  ctx.font = 'bold 13px monospace'
  const tw = ctx.measureText(label).width
  ctx.fillStyle = C.measureBg; ctx.fillRect(midX - tw / 2 - 5, midY - 11, tw + 10, 17)
  ctx.strokeStyle = C.measureBorder; ctx.lineWidth = 1; ctx.strokeRect(midX - tw / 2 - 5, midY - 11, tw + 10, 17)
  ctx.fillStyle = C.measureText; ctx.textAlign = 'center'; ctx.fillText(label, midX, midY + 4)

  if (locked && len > 20) {
    const dx = Math.abs(current[0] - liveAnchor[0]); const dy = Math.abs(current[1] - liveAnchor[1])
    ctx.font = '10px monospace'; ctx.fillStyle = C.dimText; ctx.textAlign = 'left'
    ctx.fillText(`Δx=${dx.toFixed(1)}  Δy=${dy.toFixed(1)}`, midX + 8, midY + 20)
  }
}

function drawScaleCalibrate(ctx: CanvasRenderingContext2D) {
  const ms = scaleCalibrateState.value
  const liveAnchor = ms?.anchor ?? null
  const current = ms ? (ms.locked ? ms.current : (hover.value ?? ms.current)) : null

  if (!liveAnchor || !current) {
    if (editorStore.activeTool === 'scale-calibrate' && hover.value) {
      const [hx, hy] = worldToCanvas(hover.value[0], hover.value[1])
      ctx.beginPath(); ctx.arc(hx, hy, 4, 0, Math.PI * 2)
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5; ctx.stroke()
    }
    return
  }

  const [ax, ay] = worldToCanvas(liveAnchor[0], liveAnchor[1])
  const [bx, by] = worldToCanvas(current[0], current[1])
  const dist = Math.hypot(current[0] - liveAnchor[0], current[1] - liveAnchor[1])
  const locked = ms?.locked ?? false
  const len = Math.hypot(bx - ax, by - ay)

  ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = locked ? 2.5 : 1.5
  ctx.setLineDash(locked ? [] : [6, 4])
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([])

  if (len > 4) {
    const nx = -(by - ay) / len * 10; const ny = (bx - ax) / len * 10
    ctx.lineWidth = 2
    ctx.beginPath(); ctx.moveTo(ax + nx, ay + ny); ctx.lineTo(ax - nx, ay - ny); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(bx + nx, by + ny); ctx.lineTo(bx - nx, by - ny); ctx.stroke()
  }

  ctx.beginPath(); ctx.arc(ax, ay, 5, 0, Math.PI * 2); ctx.fillStyle = '#3b82f6'; ctx.fill()
  ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.fillStyle = locked ? '#3b82f6' : C.nodeStroke; ctx.fill()

  if (!locked) {
    const midX = (ax + bx) / 2; const midY = (ay + by) / 2
    const label = `${dist.toFixed(1)} u`
    ctx.font = 'bold 13px monospace'
    const tw = ctx.measureText(label).width
    ctx.fillStyle = C.measureBg; ctx.fillRect(midX - tw / 2 - 5, midY - 11, tw + 10, 17)
    ctx.strokeStyle = C.measureBorder; ctx.lineWidth = 1; ctx.strokeRect(midX - tw / 2 - 5, midY - 11, tw + 10, 17)
    ctx.fillStyle = C.scaleLabelText; ctx.textAlign = 'center'; ctx.fillText(label, midX, midY + 4)
  }

  if (locked && len > 4) {
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.globalAlpha = 0.4
    ctx.beginPath()
    ctx.ellipse((ax + bx) / 2, (ay + by) / 2, len / 2 + 12, 12, Math.atan2(by - ay, bx - ax), 0, Math.PI * 2)
    ctx.stroke(); ctx.globalAlpha = 1; ctx.setLineDash([])
  }
}

watch(
  [() => problemStore.draft, canvasW, canvasH, hover, polylinePoints, ellipseDrag,
    () => editorStore.selected, () => editorStore.imageWorldBounds,
    () => editorStore.showConnectivity, () => editorStore.showChromosomeConnectivity, hoveredHandle, regionDrag,
    measureState, scaleCalibrateState, isDark],
  () => draw(),
  { deep: true }
)

function getCanvasPos(e: MouseEvent): [number, number] {
  const rect = canvasRef.value!.getBoundingClientRect()
  return [
    (e.clientX - rect.left) * (canvasW.value / rect.width),
    (e.clientY - rect.top) * (canvasH.value / rect.height),
  ]
}

function handleMouseDown(e: MouseEvent) {
  if (e.button !== 0) return
  wrapperRef.value?.focus()
  const [cx, cy] = getCanvasPos(e)
  const [wx, wy] = canvasToWorld(cx, cy)
  const tool = editorStore.activeTool

  const rh = hitTestRegionHandle(cx, cy)
  if (rh) { regionDrag.value = { handle: rh }; return }

  if (tool === 'select') {
    if (hitSink(cx, cy)) {
      editorStore.setSelected({ type: 'sink' })
      dragState.value = { type: 'sink', downCanvas: [cx, cy], originalPos: [problemStore.draft.sink!.x, problemStore.draft.sink!.y], moved: false }
      return
    }
    const cid = hitCandidate(cx, cy)
    if (cid) {
      const c = problemStore.draft.candidates.find(c => c.id === cid)!
      editorStore.setSelected({ type: 'candidate', id: cid })
      dragState.value = { type: 'candidate', id: cid, downCanvas: [cx, cy], originalPos: [c.x, c.y], moved: false }
      return
    }
    const tid = hitTarget(cx, cy)
    if (tid) {
      const t = problemStore.draft.targets.find(t => t.id === tid)!
      editorStore.setSelected({ type: 'target', id: tid })
      dragState.value = { type: 'target', id: tid, downCanvas: [cx, cy], originalPos: [t.x, t.y], moved: false }
      return
    }
    const rid = hitRelay(cx, cy)
    if (rid) {
      const chrom = problemStore.draft.chromosome
      if (chrom && chrom.kind === 'problem1') {
        const r = chrom.relays.find(r => r.id === rid)!
        editorStore.setSelected({ type: 'relay', id: rid })
        dragState.value = { type: 'relay', id: rid, downCanvas: [cx, cy], originalPos: [r.x, r.y], moved: false }
        return
      }
    }
    editorStore.setSelected(null); dragState.value = null
  }

  if (tool === 'draw-ellipse' && editorStore.activeNodeId) {
    ellipseDrag.value = { center: [snap(wx), snap(wy)], current: [wx, wy] }
  }
  if (tool === 'measure') {
    measureState.value = { anchor: [wx, wy], current: [wx, wy], locked: false }
  }
  if (tool === 'scale-calibrate' && !scaleCalibrateState.value?.locked) {
    scaleCalibrateState.value = { anchor: [wx, wy], current: [wx, wy], locked: false }
  }
}

function handleMouseMove(e: MouseEvent) {
  const [cx, cy] = getCanvasPos(e)
  const [wx, wy] = canvasToWorld(cx, cy)
  hover.value = [wx, wy]

  if (regionDrag.value) { applyRegionDrag(regionDrag.value.handle, wx, wy); return }
  hoveredHandle.value = hitTestRegionHandle(cx, cy)

  if (dragState.value) {
    const ds = dragState.value
    const { scale } = viewport.value
    const dx = (cx - ds.downCanvas[0]) / scale
    const dy = -(cy - ds.downCanvas[1]) / scale
    if (!ds.moved && Math.hypot(dx, dy) > 2) ds.moved = true
    if (ds.moved) {
      const nx = snap(ds.originalPos[0] + dx); const ny = snap(ds.originalPos[1] + dy)
      if (ds.type === 'sink') problemStore.setSink({ x: nx, y: ny })
      else if (ds.type === 'candidate' && ds.id) problemStore.moveCandidate(ds.id, nx, ny)
      else if (ds.type === 'target' && ds.id) problemStore.moveTarget(ds.id, nx, ny)
      else if (ds.type === 'relay' && ds.id) problemStore.moveRelay(ds.id, nx, ny)
    }
  }

  if (ellipseDrag.value) ellipseDrag.value = { ...ellipseDrag.value, current: [wx, wy] }
  if (measureState.value && !measureState.value.locked) measureState.value = { ...measureState.value, current: [wx, wy] }
  if (scaleCalibrateState.value && !scaleCalibrateState.value.locked) scaleCalibrateState.value = { ...scaleCalibrateState.value, current: [wx, wy] }
}

function handleMouseUp(e: MouseEvent) {
  if (e.button !== 0) return
  if (regionDrag.value) { regionDrag.value = null; return }
  if (dragState.value) { dragState.value = null; return }

  if (ellipseDrag.value && editorStore.activeTool === 'draw-ellipse') {
    const { center, current } = ellipseDrag.value
    const rxW = Math.abs(snap(current[0]) - center[0]); const ryW = Math.abs(snap(current[1]) - center[1])
    if (rxW > 1 && ryW > 1 && editorStore.activeNodeId) {
      problemStore.addSegmentToNode(editorStore.activeNodeId, { type: 'ellipse', center, radiusX: rxW, radiusY: ryW })
    }
    ellipseDrag.value = null; return
  }

  if (measureState.value && !measureState.value.locked) {
    const [cx2, cy2] = getCanvasPos(e); const [wx2, wy2] = canvasToWorld(cx2, cy2)
    measureState.value = { ...measureState.value, current: [wx2, wy2], locked: true }; return
  }

  if (scaleCalibrateState.value && !scaleCalibrateState.value.locked) {
    const [cx2, cy2] = getCanvasPos(e); const [wx2, wy2] = canvasToWorld(cx2, cy2)
    scaleCalibrateState.value = { ...scaleCalibrateState.value, current: [wx2, wy2], locked: true }; return
  }

  if (suppressNextClick) { suppressNextClick = false; return }
  const [cx, cy] = getCanvasPos(e); const [wx, wy] = canvasToWorld(cx, cy)
  const tool = editorStore.activeTool

  if (tool === 'place-sink') {
    problemStore.setSink({ x: snap(wx), y: snap(wy) })
  } else if (tool === 'place-candidate') {
    problemStore.addCandidate(snap(wx), snap(wy))
  } else if (tool === 'place-target' && problemStore.draft.name === 'problem3') {
    problemStore.addTarget(snap(wx), snap(wy))
  } else if (tool === 'place-relay' && problemStore.draft.name === 'problem1') {
    problemStore.addRelay(snap(wx), snap(wy))
  } else if (tool === 'chromosome-pick') {
    const cid = hitCandidate(cx, cy)
    if (cid) {
      const idx = problemStore.draft.candidates.findIndex(c => c.id === cid)
      if (idx < 0) return
      const name = problemStore.draft.name
      if (name === 'problem2' || name === 'problem3') problemStore.toggleMaskAt(idx)
      else if (name === 'problem4') problemStore.appendRouteStop(idx)
    }
  } else if (tool === 'draw-line' && editorStore.activeNodeId) {
    const pt: [number, number] = [snap(wx), snap(wy)]
    const prev = polylinePoints.value
    if (prev.length > 0) {
      problemStore.addSegmentToNode(editorStore.activeNodeId, { type: 'line', start: prev[prev.length - 1]!, end: pt })
    }
    polylinePoints.value = [...prev, pt]
  }
}

function handleDblClick(_e: MouseEvent) {
  if (editorStore.activeTool === 'draw-line' && polylinePoints.value.length > 0) {
    if (editorStore.activeNodeId) problemStore.removeLastSegmentFromNode(editorStore.activeNodeId)
    polylinePoints.value = []
    suppressNextClick = true
  }
}

function handleRightClick(e: MouseEvent) {
  const [cx, cy] = getCanvasPos(e)
  const tool = editorStore.activeTool

  if (tool === 'draw-line' && polylinePoints.value.length > 0) { polylinePoints.value = []; return }

  if (tool === 'chromosome-pick') {
    const cid = hitCandidate(cx, cy)
    if (!cid) return
    const idx = problemStore.draft.candidates.findIndex(c => c.id === cid)
    if (idx < 0) return
    const name = problemStore.draft.name
    if (name === 'problem2' || name === 'problem3') problemStore.setMaskAt(idx, 0)
    else if (name === 'problem4') problemStore.removeLastRouteStop(idx)
    return
  }

  const cid = hitCandidate(cx, cy)
  if (cid) { problemStore.removeCandidate(cid); editorStore.setSelected(null); return }
  const tid = hitTarget(cx, cy)
  if (tid) { problemStore.removeTarget(tid); editorStore.setSelected(null); return }
  if (hitSink(cx, cy)) { problemStore.setSink(null); editorStore.setSelected(null); return }
  const rid = hitRelay(cx, cy)
  if (rid) { problemStore.removeRelay(rid); editorStore.setSelected(null) }
}

function handleMouseLeave() { hover.value = null }

function handleKey(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    polylinePoints.value = []; ellipseDrag.value = null; dragState.value = null
    regionDrag.value = null; measureState.value = null; cancelScaleCalibration()
    editorStore.setSelected(null); e.preventDefault()
  }

  if ((e.key === 'Delete' || e.key === 'Backspace') && !isInputTarget(e)) {
    const sel = editorStore.selected
    if (sel?.type === 'sink') { problemStore.setSink(null); editorStore.setSelected(null) }
    else if (sel?.type === 'candidate') { problemStore.removeCandidate(sel.id); editorStore.setSelected(null) }
    else if (sel?.type === 'target') { problemStore.removeTarget(sel.id); editorStore.setSelected(null) }
    else if (sel?.type === 'relay') { problemStore.removeRelay(sel.id); editorStore.setSelected(null) }
    e.preventDefault()
  }

  if (!e.ctrlKey && !e.metaKey && !isInputTarget(e)) {
    const map: Record<string, typeof editorStore.activeTool> = {
      's': 'select', 'k': 'place-sink', 'c': 'place-candidate', 't': 'place-target',
      'n': 'place-relay', 'x': 'chromosome-pick', 'l': 'draw-line', 'e': 'draw-ellipse',
      'm': 'measure', 'r': 'scale-calibrate',
    }
    const t = map[e.key.toLowerCase()]
    if (t) {
      const name = problemStore.draft.name
      if (t === 'place-candidate' && name === 'problem1') { /* disabled */ }
      else if (t === 'place-target' && name !== 'problem3') { /* only p3 */ }
      else if (t === 'place-relay' && name !== 'problem1') { /* only p1 */ }
      else if (t === 'chromosome-pick' && name === 'problem1') { /* only p2/p3/p4 */ }
      else {
        editorStore.setTool(t)
        polylinePoints.value = []; ellipseDrag.value = null
        measureState.value = null; cancelScaleCalibration()
      }
    }
    if (e.key.toLowerCase() === 'g') editorStore.toggleConnectivity()
    if (e.key.toLowerCase() === 'h') editorStore.toggleChromosomeConnectivity()
  }
}

function isInputTarget(e: KeyboardEvent) {
  const tag = (e.target as HTMLElement)?.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
}
</script>

<style scoped>
.canvas-wrapper {
  position: relative; flex: 1; background: #f0f4f9; overflow: hidden; outline: none;
}
:global(html.dark) .canvas-wrapper {
  background: #181825;
}
canvas { display: block; width: 100%; height: 100%; }
.coords {
  position: absolute; bottom: 8px; right: 12px;
  font-size: 11px; color: #4b5875; background: rgba(240,244,249,0.85);
  padding: 2px 8px; border-radius: 4px; pointer-events: none; font-family: monospace;
  border: 1px solid #b8c5d4;
}
:global(html.dark) .coords {
  color: #6c7086; background: rgba(24,24,37,0.88); border-color: #313244;
}
.tool-hint {
  position: absolute; bottom: 8px; left: 12px;
  font-size: 11px; color: #4b5875; background: rgba(240,244,249,0.85);
  padding: 2px 8px; border-radius: 4px; pointer-events: none; max-width: calc(100% - 160px);
  border: 1px solid #b8c5d4;
}
:global(html.dark) .tool-hint {
  color: #6c7086; background: rgba(24,24,37,0.88); border-color: #313244;
}
.tool-hint.warn { color: #f97316; }
.calibration-toast {
  position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
  background: #3b82f6; color: #ffffff;
  padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 700;
  pointer-events: none; white-space: nowrap; z-index: 20;
}
.scale-input-overlay {
  position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
  background: var(--color-surface); border: 1.5px solid #3b82f6; border-radius: 8px;
  padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; min-width: 240px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.7), 0 0 0 9999px rgba(0,0,0,0.35); z-index: 9999;
}
.scale-label { font-size: 11px; color: var(--color-text-muted); }
.scale-dist-hint { font-size: 10px; color: var(--color-text-muted); font-family: monospace; }
.scale-input-overlay input {
  padding: 5px 7px; border: 1px solid #3b82f6; background: var(--color-surface); color: var(--color-text);
  border-radius: 4px; font-size: 13px; width: 100%; box-sizing: border-box;
}
.scale-btns { display: flex; gap: 6px; }
.scale-btns button { flex: 1; padding: 5px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; }
.scale-btns .apply { background: #3b82f6; color: #ffffff; }
.scale-btns .cancel { background: var(--color-border); color: var(--color-text); border: 1px solid var(--color-border); }
</style>
