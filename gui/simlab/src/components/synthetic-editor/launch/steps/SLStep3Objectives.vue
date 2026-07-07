<template>
  <div class="step">
    <p class="hint">
      Objectives are pre-populated based on the benchmark (f1…f{{ objectiveCount }}, all minimised).
      You can rename each metric — names will appear in charts and analysis.
    </p>

    <div class="objectives-list">
      <div v-for="(obj, i) in modelValue" :key="i" class="obj-row">
        <div class="obj-index">{{ i + 1 }}</div>
        <input
          :value="obj.metric_name"
          type="text"
          placeholder="e.g. f1, energy, coverage"
          class="obj-name"
          :class="{ invalid: showValidation && !obj.metric_name.trim() }"
          @input="updateName(i, ($event.target as HTMLInputElement).value)"
        />
        <span class="obj-goal">minimize</span>
      </div>
    </div>

    <span v-if="showValidation && hasEmpty" class="err">
      All objectives must have a name.
    </span>

    <div class="note">
      Goals are fixed to <strong>minimize</strong> for all standard benchmarks
      (DTLZ2, ZDT1, SCH1 are defined as minimisation problems).
      If you need maximisation, create the experiment via the Problems editor instead.
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ObjectiveItem } from '../../../../types/simlab'

const props = defineProps<{
  modelValue: ObjectiveItem[]
  showValidation: boolean
  objectiveCount: number
}>()
const emit = defineEmits<{ 'update:modelValue': [v: ObjectiveItem[]] }>()

const hasEmpty = computed(() => props.modelValue.some(o => !o.metric_name.trim()))

function updateName(i: number, value: string) {
  const next = props.modelValue.map((o, idx) =>
    idx === i ? { ...o, metric_name: value } : o
  )
  emit('update:modelValue', next)
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; }
.objectives-list { display: flex; flex-direction: column; gap: 8px; }
.obj-row {
  display: grid;
  grid-template-columns: 28px 1fr 72px;
  gap: 8px;
  align-items: center;
}
.obj-index {
  width: 28px; height: 28px;
  border-radius: 50%;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 700; color: var(--color-text-muted);
  flex-shrink: 0;
}
.obj-name {
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  color: var(--color-text);
  background: var(--color-surface);
  outline: none;
  transition: border-color 0.12s;
}
.obj-name:focus { border-color: var(--color-primary); }
.obj-name.invalid { border-color: #ef4444; }
.obj-goal {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-muted);
  text-align: center;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 6px 8px;
}
.err { font-size: 11px; color: #ef4444; }
.note {
  font-size: 11px;
  color: var(--color-text-muted);
  line-height: 1.5;
  padding: 10px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
}
</style>
