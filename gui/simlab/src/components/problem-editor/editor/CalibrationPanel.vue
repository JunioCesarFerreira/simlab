<template>
  <div v-if="hasImage" class="calibration-panel">
    <div class="section-title" @click="open = !open">
      <span>Image calibration</span>
      <span class="chevron">{{ open ? '▴' : '▾' }}</span>
    </div>

    <div v-if="open" class="body">
      <p class="hint">
        Define which world coordinates the image covers.
        Leave blank for automatic adjustment without distortion.
      </p>

      <label class="row-label">
        <span>xmin img</span>
        <input v-model.number="bounds[0]" type="number" placeholder="auto" @input="apply" />
      </label>
      <label class="row-label">
        <span>ymin img</span>
        <input v-model.number="bounds[1]" type="number" placeholder="auto" @input="apply" />
      </label>
      <label class="row-label">
        <span>xmax img</span>
        <input v-model.number="bounds[2]" type="number" placeholder="auto" @input="apply" />
      </label>
      <label class="row-label">
        <span>ymax img</span>
        <input v-model.number="bounds[3]" type="number" placeholder="auto" @input="apply" />
      </label>

      <div class="actions">
        <button @click="alignToRegion">Align to region</button>
        <button class="danger" @click="clear">Clear</button>
      </div>

      <div v-if="editorStore.imageWorldBounds" class="status ok">
        ✓ Calibrated: [{{ editorStore.imageWorldBounds.map(v => Math.round(v)).join(', ') }}]
      </div>
      <div v-else class="status">
        Auto (no distortion, no alignment)
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useEditorStore } from '../../../app/stores/editorStore'
import { useProblemStore } from '../../../app/stores/problemStore'

const editorStore = useEditorStore()
const problemStore = useProblemStore()

const open = ref(false)
const hasImage = computed(() => !!editorStore.backgroundImage)

const bounds = ref<(number | '')[]>(['', '', '', ''])

watch(() => editorStore.imageWorldBounds, (b) => {
  if (b) bounds.value = [...b]
  else bounds.value = ['', '', '', '']
}, { immediate: true })

function apply() {
  const nums = bounds.value.map(Number) as [number, number, number, number]
  const [a, b, c, d] = nums
  if (isFinite(a) && isFinite(b) && isFinite(c) && isFinite(d) && a < c && b < d) {
    editorStore.setImageWorldBounds([a, b, c, d])
  }
}

function alignToRegion() {
  const r = problemStore.draft.region
  bounds.value = [...r]
  editorStore.setImageWorldBounds([...r])
}

function clear() {
  bounds.value = ['', '', '', '']
  editorStore.setImageWorldBounds(null)
}
</script>

<style scoped>
.calibration-panel { border-top: 1px solid var(--color-border); margin-top: 8px; padding-top: 6px; }
.section-title {
  display: flex; justify-content: space-between; align-items: center;
  cursor: pointer; font-size: 12px; color: var(--color-primary); padding: 2px 0;
  user-select: none;
}
.chevron { font-size: 10px; }
.body { display: flex; flex-direction: column; gap: 5px; margin-top: 6px; }
.hint { font-size: 11px; color: #9ca3af; line-height: 1.4; }
.row-label {
  display: grid; grid-template-columns: 60px 1fr; align-items: center; gap: 6px;
  font-size: 11px; color: var(--color-text-muted);
}
input {
  padding: 3px 6px; border: 1px solid var(--color-border);
  background: var(--color-surface); color: var(--color-text);
  border-radius: 4px; font-size: 11px; width: 100%;
}
.actions { display: flex; gap: 6px; margin-top: 2px; }
.actions button {
  flex: 1; font-size: 11px; padding: 4px;
  background: var(--color-border); color: var(--color-text);
  border: 1px solid #d1d5db; border-radius: 4px; cursor: pointer;
}
.actions button.danger { color: #ef4444; border-color: #ef4444; }
.status { font-size: 10px; color: #9ca3af; margin-top: 2px; }
.status.ok { color: #10b981; }
</style>
