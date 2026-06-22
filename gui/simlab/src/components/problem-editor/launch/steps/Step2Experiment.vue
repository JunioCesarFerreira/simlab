<template>
  <div class="step">
    <div class="field-group">
      <label class="field-label" for="exp-name">Nome do experimento <span class="required">*</span></label>
      <input
        id="exp-name"
        :value="modelValue.name"
        type="text"
        placeholder="ex: nsga2-p2-run-01"
        :class="{ invalid: touched.name && !modelValue.name.trim() }"
        @input="update('name', ($event.target as HTMLInputElement).value)"
        @blur="touch('name')"
      />
      <span v-if="touched.name && !modelValue.name.trim()" class="err">Nome obrigatório.</span>
    </div>

    <div class="field-group">
      <label class="field-label" for="exp-strategy">Estratégia de otimização</label>
      <select
        id="exp-strategy"
        :value="modelValue.strategy"
        @change="update('strategy', ($event.target as HTMLSelectElement).value)"
      >
        <option value="nsga2">NSGA-II</option>
      </select>
    </div>

    <div class="section-divider">Hiperparâmetros do algoritmo</div>

    <div class="fields-row">
      <div class="field-group">
        <label class="field-label" for="pop-size">Tamanho da população <span class="required">*</span></label>
        <input
          id="pop-size"
          :value="modelValue.populationSize"
          type="number" min="2" step="1"
          :class="{ invalid: touched.populationSize && modelValue.populationSize < 2 }"
          @input="updateInt('populationSize', ($event.target as HTMLInputElement).value)"
          @blur="touch('populationSize')"
        />
        <span v-if="touched.populationSize && modelValue.populationSize < 2" class="err">Mínimo 2.</span>
      </div>
      <div class="field-group">
        <label class="field-label" for="num-gen">Número de gerações <span class="required">*</span></label>
        <input
          id="num-gen"
          :value="modelValue.numberOfGenerations"
          type="number" min="1" step="1"
          :class="{ invalid: touched.numberOfGenerations && modelValue.numberOfGenerations < 1 }"
          @input="updateInt('numberOfGenerations', ($event.target as HTMLInputElement).value)"
          @blur="touch('numberOfGenerations')"
        />
        <span v-if="touched.numberOfGenerations && modelValue.numberOfGenerations < 1" class="err">Mínimo 1.</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue'

export interface Step2Value {
  name: string
  strategy: string
  populationSize: number
  numberOfGenerations: number
}

const props = defineProps<{ modelValue: Step2Value }>()
const emit = defineEmits<{ 'update:modelValue': [v: Step2Value] }>()

const touched = reactive({ name: false, populationSize: false, numberOfGenerations: false })

function touch(field: keyof typeof touched) { touched[field] = true }

function update(field: keyof Step2Value, value: string) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}

function updateInt(field: keyof Step2Value, raw: string) {
  emit('update:modelValue', { ...props.modelValue, [field]: parseInt(raw, 10) || 0 })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--color-text-muted); border-bottom: 1px solid var(--color-border); padding-bottom: 4px; margin-top: 4px;
}
.fields-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.field-group { display: flex; flex-direction: column; gap: 4px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
input, select {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface);
  outline: none; transition: border-color 0.12s;
}
input:focus, select:focus { border-color: var(--color-primary); }
input.invalid { border-color: #ef4444; }
.err { font-size: 11px; color: #ef4444; }
</style>
