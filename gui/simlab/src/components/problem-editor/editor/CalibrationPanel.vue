<template>
  <div class="calibration-panel" v-if="hasImage">
    <div class="section-title" @click="open = !open">
      <span>Calibração da imagem</span>
      <span class="chevron">{{ open ? '▴' : '▾' }}</span>
    </div>

    <div v-if="open" class="body">
      <p class="hint">
        Defina quais coordenadas mundo a imagem cobre.
        Deixe em branco para ajuste automático sem deformação.
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
        <button @click="alignToRegion">Alinhar à região</button>
        <button class="danger" @click="clear">Limpar</button>
      </div>

      <div v-if="editorStore.imageWorldBounds" class="status ok">
        ✓ Calibrado: [{{ editorStore.imageWorldBounds.map(v => Math.round(v)).join(', ') }}]
      </div>
      <div v-else class="status">
        Auto (sem deformação, sem alinhamento)
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
  const [a, b, c, d] = bounds.value.map(Number)
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
.calibration-panel { border-top: 1px solid #313244; margin-top: 8px; padding-top: 6px; }
.section-title {
  display: flex; justify-content: space-between; align-items: center;
  cursor: pointer; font-size: 12px; color: #89b4fa; padding: 2px 0;
  user-select: none;
}
.chevron { font-size: 10px; }
.body { display: flex; flex-direction: column; gap: 5px; margin-top: 6px; }
.hint { font-size: 11px; color: #6c7086; line-height: 1.4; }
.row-label {
  display: grid; grid-template-columns: 60px 1fr; align-items: center; gap: 6px;
  font-size: 11px; color: #a6adc8;
}
input {
  padding: 3px 6px; border: 1px solid #444;
  background: #1e1e2e; color: #cdd6f4;
  border-radius: 4px; font-size: 11px; width: 100%;
}
.actions { display: flex; gap: 6px; margin-top: 2px; }
.actions button {
  flex: 1; font-size: 11px; padding: 4px;
  background: #313244; color: #cdd6f4;
  border: 1px solid #45475a; border-radius: 4px; cursor: pointer;
}
.actions button.danger { color: #f38ba8; border-color: #f38ba8; }
.status { font-size: 10px; color: #6c7086; margin-top: 2px; }
.status.ok { color: #a6e3a1; }
</style>
