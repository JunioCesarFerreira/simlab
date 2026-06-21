<template>
  <div class="dispatcher">
    <div v-if="!chromosome" class="empty-state">
      <p>Nenhum cromossomo definido para <b>{{ problemName }}</b>.</p>
      <button class="primary" @click="problemStore.ensureChromosome()">Criar cromossomo</button>
    </div>
    <template v-else>
      <ChromosomeP1Panel v-if="chromosome.kind === 'problem1'" />
      <ChromosomeMaskPanel v-else-if="chromosome.kind === 'problem2' || chromosome.kind === 'problem3'" />
      <ChromosomeP4Panel v-else-if="chromosome.kind === 'problem4'" />
      <div class="footer">
        <button @click="copyJson">{{ copied ? 'Copied!' : 'Copy JSON' }}</button>
        <button @click="downloadJson">Download</button>
        <button class="danger" @click="problemStore.clearChromosome()">Clear</button>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'
import { exportChromosome } from '../../../services/exportChromosomeJson'
import ChromosomeP1Panel from './ChromosomeP1Panel.vue'
import ChromosomeMaskPanel from './ChromosomeMaskPanel.vue'
import ChromosomeP4Panel from './ChromosomeP4Panel.vue'

const problemStore = useProblemStore()
const copied = ref(false)

const problemName = computed(() => problemStore.draft.name)
const chromosome = computed(() => problemStore.draft.chromosome)

const jsonText = computed(() => {
  if (!chromosome.value) return ''
  return JSON.stringify(exportChromosome(chromosome.value), null, 2)
})

function copyJson() {
  if (!jsonText.value) return
  navigator.clipboard.writeText(jsonText.value)
  copied.value = true
  setTimeout(() => (copied.value = false), 2000)
}

function downloadJson() {
  if (!jsonText.value) return
  const blob = new Blob([jsonText.value], { type: 'application/json' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `${problemStore.draft.name}-chromosome.json`
  a.click()
  URL.revokeObjectURL(a.href)
}
</script>

<style scoped>
.dispatcher { display: flex; flex-direction: column; gap: 8px; }
.empty-state { text-align: center; padding: 20px 4px; color: #6c7086; font-size: 12px; display: flex; flex-direction: column; gap: 10px; }
.empty-state p { line-height: 1.4; }
button.primary { align-self: center; background: #89b4fa; color: #1e1e2e; border: none; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; }
.footer { display: flex; gap: 4px; margin-top: 6px; border-top: 1px solid #313244; padding-top: 8px; }
.footer button { flex: 1; padding: 4px 6px; font-size: 11px; background: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; cursor: pointer; }
.footer button.danger { background: none; color: #f38ba8; border-color: #45475a; flex: 0 0 auto; padding: 4px 8px; }
</style>
