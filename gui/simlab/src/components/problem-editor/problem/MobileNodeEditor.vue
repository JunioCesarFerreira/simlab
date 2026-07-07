<template>
  <div v-if="node" class="mobile-node-editor">
    <h3>{{ node.name }}</h3>
    <label>Name<input v-model="node.name" /></label>
    <label>Source Code<input v-model="node.sourceCode" /></label>
    <div class="coord-row">
      <span class="axis-label">Spd</span>
      <input v-model.number="node.speed" type="number" min="0.1" step="0.1" placeholder="speed" />
      <input v-model.number="node.timeStep" type="number" min="0.01" step="0.01" placeholder="step" />
    </div>
    <div class="checkboxes">
      <label class="inline"><input v-model="node.isClosed" type="checkbox" /> Closed</label>
      <label class="inline"><input v-model="node.isRoundTrip" type="checkbox" /> Round Trip</label>
    </div>

    <div class="segments-section">
      <div class="seg-header">Segments ({{ node.segments.length }})</div>
      <div v-for="(seg, i) in node.segments" :key="i" class="segment-item">
        <span class="seg-type">{{ seg.type }}</span>
        <span class="seg-desc">{{ describeSegment(seg) }}</span>
        <button @click="problemStore.removeSegmentFromNode(node.id, i)">✕</button>
      </div>
      <div v-if="node.segments.length === 0" class="empty">No segments yet</div>
    </div>

    <SegmentEditor :node-id="node.id" />
  </div>
  <div v-else class="no-selection">Select a mobile node to edit</div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useEditorStore } from '../../../app/stores/editorStore'
import SegmentEditor from './SegmentEditor.vue'
import type { DraftSegment } from '../../../types/problem'

const problemStore = useProblemStore()
const editorStore = useEditorStore()

const node = computed(() => {
  if (!editorStore.activeNodeId) return null
  return problemStore.draft.mobileNodes.find(n => n.id === editorStore.activeNodeId) ?? null
})

function describeSegment(seg: DraftSegment): string {
  if (seg.type === 'line') return `(${seg.start[0]},${seg.start[1]})→(${seg.end[0]},${seg.end[1]})`
  if (seg.type === 'ellipse') return `c=(${seg.center[0]},${seg.center[1]}) rx=${seg.radiusX} ry=${seg.radiusY}`
  return `x=${seg.exprX}`
}
</script>

<style scoped>
.mobile-node-editor { display: flex; flex-direction: column; gap: 6px; }
label { display: flex; flex-direction: column; font-size: 12px; gap: 2px; }
label.inline { flex-direction: row; align-items: center; gap: 4px; }
input:not([type=checkbox]) { padding: 3px 4px; border: 1px solid var(--color-border); background: var(--color-surface); color: var(--color-text); border-radius: 4px; font-size: 11px; width: 100%; min-width: 0; }
.coord-row { display: grid; grid-template-columns: 28px 60px 60px; gap: 4px; align-items: center; }
.axis-label { font-size: 10px; color: #9ca3af; text-align: right; }
.checkboxes { display: flex; gap: 12px; font-size: 12px; }
.segments-section { margin-top: 4px; }
.seg-header { font-size: 11px; color: var(--color-text-muted); margin-bottom: 4px; }
.segment-item { display: flex; align-items: center; gap: 6px; padding: 3px 6px; background: var(--color-border); border-radius: 4px; font-size: 11px; margin-bottom: 3px; }
.seg-type { color: var(--color-primary); min-width: 44px; }
.seg-desc { flex: 1; color: var(--color-text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.segment-item button { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 12px; padding: 0; }
.empty { font-size: 11px; color: #9ca3af; text-align: center; padding: 6px; }
.no-selection { font-size: 12px; color: #9ca3af; text-align: center; padding: 16px; }
</style>
