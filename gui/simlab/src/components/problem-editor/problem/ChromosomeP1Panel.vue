<template>
  <div class="chrom-panel">
    <h3>Chromosome · problem1</h3>
    <MacProtocolField />

    <div class="list-header">
      <span>Relays ({{ relays.length }}/{{ limit }})</span>
      <button class="ghost" :disabled="atLimit" @click="addEmpty">+ relay</button>
    </div>

    <div v-if="relays.length === 0" class="empty">
      Nenhum relay — use a ferramenta ⊗ no canvas ou o botão acima
    </div>

    <div v-for="(r, i) in relays" :key="r.id" class="row">
      <span class="idx">#{{ i + 1 }}</span>
      <input
        type="number"
        :value="r.x"
        title="x"
        @change="update(r.id, 'x', +($event.target as HTMLInputElement).value)"
      />
      <input
        type="number"
        :value="r.y"
        title="y"
        @change="update(r.id, 'y', +($event.target as HTMLInputElement).value)"
      />
      <button class="remove" title="Remover" @click="problemStore.removeRelay(r.id)">✕</button>
    </div>

    <div v-if="hasError('chromosome.relays')" class="err-msg">
      ⚠ {{ errorFor('chromosome.relays') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useValidation } from '../../../composables/useValidation'
import MacProtocolField from './MacProtocolField.vue'
import type { ChromosomeP1Draft } from '../../../types/problem'

const problemStore = useProblemStore()
const { hasError, errorFor } = useValidation()

const chrom = computed(() => problemStore.draft.chromosome as ChromosomeP1Draft | null)
const relays = computed(() => chrom.value?.relays ?? [])
const limit = computed(() => problemStore.draft.numSensors)
const atLimit = computed(() => relays.value.length >= limit.value)

function update(id: string, axis: 'x' | 'y', value: number) {
  const r = relays.value.find(r => r.id === id)
  if (!r || !isFinite(value)) return
  if (axis === 'x') problemStore.moveRelay(id, value, r.y)
  else problemStore.moveRelay(id, r.x, value)
}

function addEmpty() {
  const [xmin, ymin, xmax, ymax] = problemStore.draft.region
  problemStore.addRelay(Math.round((xmin + xmax) / 2), Math.round((ymin + ymax) / 2))
}
</script>

<style scoped>
.chrom-panel { display: flex; flex-direction: column; gap: 6px; }
.list-header { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--color-primary); }
.list-header button.ghost { background: none; border: 1px dashed #d1d5db; color: var(--color-text-muted); padding: 2px 6px; border-radius: 4px; cursor: pointer; font-size: 11px; }
.list-header button.ghost:disabled { opacity: 0.4; cursor: not-allowed; }
.empty { font-size: 11px; color: #9ca3af; text-align: center; padding: 6px 0; }
.row { display: grid; grid-template-columns: 20px 1fr 1fr 18px; gap: 3px; align-items: center; }
.idx { font-size: 10px; color: #9ca3af; text-align: right; }
input { padding: 2px 4px; border: 1px solid var(--color-border); background: var(--color-surface); color: var(--color-text); border-radius: 4px; font-size: 11px; width: 100%; min-width: 0; }
.remove { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 11px; padding: 0; line-height: 1; }
.err-msg { font-size: 10px; color: #ef4444; }
</style>
