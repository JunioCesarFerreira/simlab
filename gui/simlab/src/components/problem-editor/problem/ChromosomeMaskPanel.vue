<template>
  <div class="chrom-panel">
    <h3>Chromosome · {{ problemName }}</h3>
    <MacProtocolField />

    <div class="list-header">
      <span>Mask ({{ activeCount }}/{{ candidates.length }} ativos)</span>
      <div class="bulk">
        <button class="ghost" :disabled="candidates.length === 0" @click="setAll(1)">all</button>
        <button class="ghost" :disabled="candidates.length === 0" @click="setAll(0)">none</button>
      </div>
    </div>

    <div v-if="candidates.length === 0" class="empty">
      Adicione candidates no canvas primeiro
    </div>

    <div v-for="(c, i) in candidates" :key="c.id" class="row">
      <input
        type="checkbox"
        :checked="mask[i] === 1"
        @change="problemStore.toggleMaskAt(i)"
      />
      <span class="idx">#{{ i + 1 }}</span>
      <span class="coord">({{ c.x }}, {{ c.y }})</span>
    </div>

    <div v-if="hasError('chromosome.mask')" class="err-msg">
      ⚠ {{ errorFor('chromosome.mask') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { useValidation } from '../../../composables/useValidation'
import MacProtocolField from './MacProtocolField.vue'
import type { ChromosomeP2Draft, ChromosomeP3Draft } from '../../../types/problem'

const problemStore = useProblemStore()
const { hasError, errorFor } = useValidation()

const problemName = computed(() => problemStore.draft.name)
const candidates = computed(() => problemStore.draft.candidates)
const chrom = computed(() => problemStore.draft.chromosome as ChromosomeP2Draft | ChromosomeP3Draft | null)
const mask = computed(() => chrom.value?.mask ?? [])
const activeCount = computed(() => mask.value.reduce((a, b) => a + (b === 1 ? 1 : 0), 0))

function setAll(value: 0 | 1) {
  mask.value.forEach((_, i) => problemStore.setMaskAt(i, value))
}
</script>

<style scoped>
.chrom-panel { display: flex; flex-direction: column; gap: 6px; }
.list-header { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--color-primary); }
.bulk { display: flex; gap: 4px; }
.bulk button.ghost { background: none; border: 1px dashed #d1d5db; color: var(--color-text-muted); padding: 1px 6px; border-radius: 4px; cursor: pointer; font-size: 10px; }
.bulk button.ghost:disabled { opacity: 0.4; cursor: not-allowed; }
.empty { font-size: 11px; color: #9ca3af; text-align: center; padding: 6px 0; }
.row { display: grid; grid-template-columns: 16px 24px 1fr; gap: 5px; align-items: center; font-size: 11px; }
.idx { color: #9ca3af; }
.coord { color: var(--color-text-muted); font-family: monospace; }
input[type="checkbox"] { accent-color: var(--color-primary); }
.err-msg { font-size: 10px; color: #ef4444; }
</style>
