<template>
  <div class="toolbar">
    <div class="tool-group">
      <button
        v-for="tool in tools"
        :key="tool.id"
        :class="{ active: activeTool === tool.id }"
        :disabled="(tool.id === 'place-candidate' && !candidatesEnabled) || (tool.id === 'place-relay' && !relaysEnabled)"
        :title="tool.label"
        @click="editorStore.setTool(tool.id)"
      >{{ tool.icon }}</button>
    </div>
    <div class="tool-group">
      <label class="icon-btn" title="Load background image">
        🖼
        <input type="file" accept="image/*" @change="loadImage" style="display:none" />
      </label>
    </div>
    <div class="spacer" />
    <button title="Import JSON" @click="$emit('import')">⬆</button>
    <button :class="{ active: showConnectivity }" title="Connectivity Graph  [G] — sink + all candidates within reach" @click="editorStore.toggleConnectivity()">⬡</button>
    <button :class="{ active: showChromosomeConnectivity }" class="chrom" title="Chromosome Connectivity  [H] — sink + nodes active in current chromosome" @click="editorStore.toggleChromosomeConnectivity()">⬢</button>
    <button :class="{ active: showJson }" title="JSON Preview  [{ }]" @click="editorStore.toggleJsonPreview()">{ }</button>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useEditorStore, type EditorTool } from '../../../app/stores/editorStore'
import { useProblemStore } from '../../../app/stores/problemStore'
import { hasCandidates, hasTargets } from '../../../types/problem'

defineEmits<{ import: [] }>()

const editorStore = useEditorStore()
const problemStore = useProblemStore()
const activeTool = computed(() => editorStore.activeTool)
const showJson = computed(() => editorStore.showJsonPreview)
const showConnectivity = computed(() => editorStore.showConnectivity)
const showChromosomeConnectivity = computed(() => editorStore.showChromosomeConnectivity)
const candidatesEnabled = computed(() => hasCandidates(problemStore.draft.name))
const targetsEnabled = computed(() => hasTargets(problemStore.draft.name))
const relaysEnabled = computed(() => problemStore.draft.name === 'problem1')
const chromosomePickEnabled = computed(() => ['problem2', 'problem3', 'problem4'].includes(problemStore.draft.name))

const allTools: { id: EditorTool; icon: string; label: string; visibleWhen?: () => boolean }[] = [
  { id: 'select',           icon: '↖',  label: 'Select / Move  [S]' },
  { id: 'place-sink',       icon: '⊕',  label: 'Place Sink  [K]' },
  { id: 'place-candidate',  icon: '●',  label: 'Place Candidate  [C]' },
  { id: 'place-target',     icon: '◇',  label: 'Place Target  [T]', visibleWhen: () => targetsEnabled.value },
  { id: 'place-relay',      icon: '⧫',  label: 'Place Relay  [N] — problem1 chromosome', visibleWhen: () => relaysEnabled.value },
  { id: 'chromosome-pick',  icon: '▣',  label: 'Chromosome Pick  [X] — click candidates to toggle mask (p2/p3) or append route stop (p4)', visibleWhen: () => chromosomePickEnabled.value },
  { id: 'draw-line',        icon: '╱',  label: 'Draw Polyline  [L] — right-click/Esc to finish' },
  { id: 'draw-ellipse',     icon: '○',  label: 'Draw Ellipse  [E] — drag to define radii' },
  { id: 'measure',          icon: '📏', label: 'Tape Measure  [M] — click-drag to measure distance' },
  { id: 'scale-calibrate',  icon: '⚖',  label: 'Scale Calibration  [R] — drag a known segment then enter its real length' },
]
const tools = computed(() => allTools.filter(t => !t.visibleWhen || t.visibleWhen()))

function loadImage(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = e => editorStore.setBackground(e.target?.result as string)
  reader.readAsDataURL(file)
}
</script>

<style scoped>
.toolbar { display: flex; align-items: center; gap: 4px; padding: 4px 10px; background: var(--color-surface); border-bottom: 1px solid var(--color-border); height: 44px; }
.tool-group { display: flex; gap: 2px; }
button, .icon-btn { width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; background: var(--color-border); color: #1e293b; border: 1px solid #94a3b8; border-radius: 6px; cursor: pointer; font-size: 15px; }
button.active { background: var(--color-primary); color: var(--color-surface); border-color: var(--color-primary); }
button.chrom.active { background: #8b5cf6; border-color: #8b5cf6; }
button:disabled { opacity: 0.35; cursor: not-allowed; }
.icon-btn { cursor: pointer; }
.spacer { flex: 1; }
</style>
