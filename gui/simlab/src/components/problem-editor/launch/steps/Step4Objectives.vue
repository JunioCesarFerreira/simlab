<template>
  <div class="step">
    <p class="hint">
      Defina as métricas que o algoritmo deve otimizar. Cada objetivo mapeia para uma métrica calculada
      a partir dos logs de simulação (configurada no passo seguinte).
    </p>

    <div class="objectives-list">
      <div v-for="(obj, i) in modelValue" :key="i" class="obj-row">
        <div class="obj-index">{{ i + 1 }}</div>
        <input
          :value="obj.metric_name"
          type="text"
          placeholder="ex: energy, coverage, latency"
          list="metric-suggestions"
          class="obj-name"
          :class="{ invalid: showValidation && !obj.metric_name.trim() }"
          @input="updateObj(i, 'metric_name', ($event.target as HTMLInputElement).value)"
        />
        <select
          :value="obj.goal"
          class="obj-goal"
          @change="updateObj(i, 'goal', ($event.target as HTMLSelectElement).value)"
        >
          <option value="min">Minimizar</option>
          <option value="max">Maximizar</option>
        </select>
        <button class="remove-btn" :disabled="modelValue.length <= 1" @click="remove(i)" title="Remover">×</button>
      </div>
    </div>

    <datalist id="metric-suggestions">
      <option value="energy" />
      <option value="coverage" />
      <option value="latency" />
      <option value="packet_loss" />
      <option value="throughput" />
      <option value="lifetime" />
    </datalist>

    <button class="add-btn" @click="add">+ Adicionar objetivo</button>

    <span v-if="showValidation && hasEmpty" class="err">
      Todos os objetivos precisam ter um nome de métrica.
    </span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ObjectiveItem } from '../../../../types/simlab'

const props = defineProps<{ modelValue: ObjectiveItem[]; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: ObjectiveItem[]] }>()

const hasEmpty = computed(() => props.modelValue.some(o => !o.metric_name.trim()))

function updateObj(i: number, field: keyof ObjectiveItem, value: string) {
  const next = props.modelValue.map((o, idx) => idx === i ? { ...o, [field]: value } : o)
  emit('update:modelValue', next)
}

function remove(i: number) {
  if (props.modelValue.length <= 1) return
  emit('update:modelValue', props.modelValue.filter((_, idx) => idx !== i))
}

function add() {
  emit('update:modelValue', [...props.modelValue, { metric_name: '', goal: 'min' }])
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; }
.objectives-list { display: flex; flex-direction: column; gap: 8px; }
.obj-row { display: grid; grid-template-columns: 24px 1fr 140px 32px; gap: 8px; align-items: center; }
.obj-index {
  width: 24px; height: 24px; border-radius: 50%; background: var(--color-primary-light);
  color: var(--color-primary); font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
input, select {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface);
  outline: none; width: 100%;
}
input:focus, select:focus { border-color: var(--color-primary); }
input.invalid { border-color: #ef4444; }
.remove-btn {
  width: 32px; height: 32px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: none; color: #ef4444; font-size: 16px; cursor: pointer; display: flex; align-items: center; justify-content: center;
}
.remove-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.add-btn {
  align-self: flex-start; padding: 7px 14px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); cursor: pointer;
}
.add-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }
.err { font-size: 11px; color: #ef4444; }
</style>
