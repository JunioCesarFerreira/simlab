<template>
  <div class="step">
    <p class="hint">
      Defina os seeds de aleatoriedade usados nas simulações. Cada seed gera uma simulação independente por geração,
      aumentando a robustez estatística dos resultados.
    </p>

    <div class="field-group">
      <label class="field-label">Seeds de aleatoriedade <span class="required">*</span></label>
      <div class="seeds-row">
        <div class="chips">
          <span v-for="(seed, i) in modelValue.randomSeeds" :key="i" class="chip">
            {{ seed }}
            <button class="chip-remove" @click="removeSeed(i)" title="Remover seed">×</button>
          </span>
          <span v-if="modelValue.randomSeeds.length === 0" class="empty-chips">Nenhum seed adicionado.</span>
        </div>
        <div class="add-row">
          <input
            v-model="newSeed"
            type="number"
            step="1"
            placeholder="ex: 42"
            @keydown.enter.prevent="addSeed"
          />
          <button class="add-btn" @click="addSeed" :disabled="!canAdd">+ Adicionar</button>
        </div>
      </div>
      <span v-if="showError" class="err">Adicione ao menos um seed.</span>
    </div>

    <div class="presets">
      <span class="presets-label">Sugestões rápidas:</span>
      <button v-for="preset in PRESETS" :key="preset.label" class="preset-btn" @click="applyPreset(preset.seeds)">
        {{ preset.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

export interface Step3Value {
  randomSeeds: number[]
}

const props = defineProps<{ modelValue: Step3Value; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: Step3Value] }>()

const newSeed = ref<string>('')
const canAdd = computed(() => {
  const v = parseInt(newSeed.value, 10)
  return Number.isInteger(v) && !props.modelValue.randomSeeds.includes(v)
})
const showError = computed(() => props.showValidation && props.modelValue.randomSeeds.length === 0)

const PRESETS = [
  { label: '1 seed (42)', seeds: [42] },
  { label: '3 seeds', seeds: [1, 2, 3] },
  { label: '5 seeds', seeds: [1, 2, 3, 4, 5] },
  { label: '10 seeds', seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] },
]

function addSeed() {
  const v = parseInt(newSeed.value, 10)
  if (!Number.isInteger(v) || props.modelValue.randomSeeds.includes(v)) return
  emit('update:modelValue', { randomSeeds: [...props.modelValue.randomSeeds, v].sort((a, b) => a - b) })
  newSeed.value = ''
}

function removeSeed(i: number) {
  const seeds = [...props.modelValue.randomSeeds]
  seeds.splice(i, 1)
  emit('update:modelValue', { randomSeeds: seeds })
}

function applyPreset(seeds: number[]) {
  emit('update:modelValue', { randomSeeds: seeds })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; }
.field-group { display: flex; flex-direction: column; gap: 6px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
.seeds-row { display: flex; flex-direction: column; gap: 8px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; min-height: 32px; align-items: flex-start; }
.chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 10px; background: var(--color-primary-light);
  border: 1px solid #bfdbfe; border-radius: 99px;
  font-size: 12px; font-weight: 600; color: var(--color-primary); font-family: monospace;
}
.chip-remove {
  background: none; border: none; color: var(--color-primary); cursor: pointer;
  font-size: 14px; line-height: 1; padding: 0; opacity: 0.6;
}
.chip-remove:hover { opacity: 1; }
.empty-chips { font-size: 12px; color: var(--color-text-muted); align-self: center; }
.add-row { display: flex; gap: 8px; }
.add-row input {
  flex: 1; padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none;
}
.add-row input:focus { border-color: var(--color-primary); }
.add-btn {
  padding: 7px 14px; background: var(--color-primary); color: #fff;
  border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; border: none; cursor: pointer;
}
.add-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.presets { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.presets-label { font-size: 11px; color: var(--color-text-muted); }
.preset-btn {
  padding: 4px 10px; font-size: 11px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  color: var(--color-text); cursor: pointer;
}
.preset-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }
.err { font-size: 11px; color: #ef4444; }
</style>
