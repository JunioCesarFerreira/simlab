<template>
  <div class="chrom-panel">
    <h3>Chromosome · problem4</h3>
    <MacProtocolField />

    <div class="list-header">
      <span>Route ({{ route.length }} stops)</span>
      <button class="ghost" :disabled="candidates.length === 0" @click="appendStop">+ stop</button>
    </div>

    <div v-if="candidates.length === 0" class="empty">
      Add candidates on the canvas first
    </div>

    <div v-for="(idx, i) in route" :key="`stop-${i}`" class="row">
      <span class="idx">#{{ i + 1 }}</span>
      <select :value="idx" @change="setIndex(i, +($event.target as HTMLSelectElement).value)">
        <option v-for="(_, j) in candidates" :key="j" :value="j">
          cand #{{ j + 1 }}
        </option>
      </select>
      <input
        type="number"
        min="0"
        step="any"
        :value="sojournTimes[i] ?? 0"
        title="sojourn time"
        @change="setTime(i, +($event.target as HTMLInputElement).value)"
      />
      <button class="remove" title="Remove" @click="removeStop(i)">✕</button>
    </div>

    <div v-if="hasError('chromosome.route')" class="err-msg">
      ⚠ {{ errorFor('chromosome.route') }}
    </div>
    <div v-if="hasError('chromosome.sojourn_times')" class="err-msg">
      ⚠ {{ errorFor('chromosome.sojourn_times') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useValidation } from '../../../composables/useValidation'
import MacProtocolField from './MacProtocolField.vue'
import type { ChromosomeP4Draft } from '../../../types/problem'

const problemStore = useProblemStore()
const { hasError, errorFor } = useValidation()

const candidates = computed(() => problemStore.draft.candidates)
const chrom = computed(() => problemStore.draft.chromosome as ChromosomeP4Draft | null)
const route = computed(() => chrom.value?.route ?? [])
const sojournTimes = computed(() => chrom.value?.sojournTimes ?? [])

function appendStop() {
  problemStore.setRoute([...route.value, 0])
}

function removeStop(i: number) {
  const r = [...route.value]
  const t = [...sojournTimes.value]
  r.splice(i, 1); t.splice(i, 1)
  problemStore.setRoute(r)
  problemStore.setSojournTimes(t)
}

function setIndex(i: number, value: number) {
  const r = [...route.value]
  r[i] = value
  problemStore.setRoute(r)
}

function setTime(i: number, value: number) {
  if (!isFinite(value)) return
  const t = [...sojournTimes.value]
  t[i] = value
  problemStore.setSojournTimes(t)
}
</script>

<style scoped>
.chrom-panel { display: flex; flex-direction: column; gap: 6px; }
.list-header { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--color-primary); }
.list-header button.ghost { background: none; border: 1px dashed #d1d5db; color: var(--color-text-muted); padding: 2px 6px; border-radius: 4px; cursor: pointer; font-size: 11px; }
.list-header button.ghost:disabled { opacity: 0.4; cursor: not-allowed; }
.empty { font-size: 11px; color: #9ca3af; text-align: center; padding: 6px 0; }
.row { display: grid; grid-template-columns: 22px 1fr 1fr 18px; gap: 3px; align-items: center; }
.idx { font-size: 10px; color: #9ca3af; text-align: right; }
input, select { padding: 2px 4px; border: 1px solid var(--color-border); background: var(--color-surface); color: var(--color-text); border-radius: 4px; font-size: 11px; width: 100%; min-width: 0; }
.remove { background: none; border: none; color: #ef4444; cursor: pointer; font-size: 11px; padding: 0; line-height: 1; }
.err-msg { font-size: 10px; color: #ef4444; }
</style>
