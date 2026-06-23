<template>
  <div class="step">
    <div class="field-group">
      <label class="field-label" for="exp-name">Experiment name <span class="required">*</span></label>
      <input
        id="exp-name"
        :value="modelValue.name"
        type="text"
        placeholder="e.g. nsga3-p2-run-01"
        :class="{ invalid: touched.name && !modelValue.name.trim() }"
        @input="update('name', ($event.target as HTMLInputElement).value)"
        @blur="touch('name')"
      />
      <span v-if="touched.name && !modelValue.name.trim()" class="err">Name is required.</span>
    </div>

    <div class="field-group">
      <label class="field-label" for="exp-strategy">Optimization strategy</label>
      <select
        id="exp-strategy"
        :value="modelValue.strategy"
        @change="update('strategy', ($event.target as HTMLSelectElement).value)"
      >
        <option value="nsga2">NSGA-II</option>
        <option value="nsga3">NSGA-III</option>
      </select>
    </div>

    <div class="section-divider">Algorithm hyperparameters</div>

    <div class="fields-row">
      <div class="field-group">
        <label class="field-label" for="pop-size">Population size <span class="required">*</span></label>
        <input
          id="pop-size"
          :value="modelValue.populationSize"
          type="number" min="2" step="1"
          :class="{ invalid: touched.populationSize && modelValue.populationSize < 2 }"
          @input="updateNum('populationSize', ($event.target as HTMLInputElement).value, 'int')"
          @blur="touch('populationSize')"
        />
        <span v-if="touched.populationSize && modelValue.populationSize < 2" class="err">Minimum 2.</span>
      </div>
      <div class="field-group">
        <label class="field-label" for="num-gen">Number of generations <span class="required">*</span></label>
        <input
          id="num-gen"
          :value="modelValue.numberOfGenerations"
          type="number" min="1" step="1"
          :class="{ invalid: touched.numberOfGenerations && modelValue.numberOfGenerations < 1 }"
          @input="updateNum('numberOfGenerations', ($event.target as HTMLInputElement).value, 'int')"
          @blur="touch('numberOfGenerations')"
        />
        <span v-if="touched.numberOfGenerations && modelValue.numberOfGenerations < 1" class="err">Minimum 1.</span>
      </div>
    </div>

    <div class="fields-row">
      <div class="field-group">
        <label class="field-label" for="prob-cx">Crossover probability</label>
        <input
          id="prob-cx"
          :value="modelValue.probCx"
          type="number" min="0" max="1" step="0.01"
          @input="updateNum('probCx', ($event.target as HTMLInputElement).value, 'float')"
        />
      </div>
      <div class="field-group">
        <label class="field-label" for="prob-mt">Mutation probability</label>
        <input
          id="prob-mt"
          :value="modelValue.probMt"
          type="number" min="0" max="1" step="0.01"
          @input="updateNum('probMt', ($event.target as HTMLInputElement).value, 'float')"
        />
      </div>
    </div>

    <div class="fields-row">
      <div class="field-group">
        <label class="field-label" for="per-gene-prob">Per-gene mutation prob.</label>
        <input
          id="per-gene-prob"
          :value="modelValue.perGeneProb"
          type="number" min="0" max="1" step="0.01"
          @input="updateNum('perGeneProb', ($event.target as HTMLInputElement).value, 'float')"
        />
      </div>
      <div class="field-group">
        <label class="field-label" for="random-seed">Algorithm random seed</label>
        <input
          id="random-seed"
          :value="modelValue.randomSeed"
          type="number" step="1"
          @input="updateNum('randomSeed', ($event.target as HTMLInputElement).value, 'int')"
        />
      </div>
    </div>

    <div class="fields-row">
      <div class="field-group">
        <label class="field-label" for="selection-method">Selection method</label>
        <select
          id="selection-method"
          :value="modelValue.selectionMethod"
          @change="update('selectionMethod', ($event.target as HTMLSelectElement).value)"
        >
          <option value="tournament">tournament</option>
          <option value="roulette">roulette</option>
        </select>
      </div>
      <div class="field-group">
        <label class="field-label" for="crossover-method">Crossover method</label>
        <select
          id="crossover-method"
          :value="modelValue.crossoverMethod"
          @change="update('crossoverMethod', ($event.target as HTMLSelectElement).value)"
        >
          <option value="uniform_mask">uniform_mask</option>
          <option value="sbx_with_radial_translate">sbx_with_radial_translate</option>
          <option value="one_point">one_point</option>
          <option value="two_point">two_point</option>
        </select>
      </div>
    </div>

    <div class="fields-row half">
      <div class="field-group">
        <label class="field-label" for="mutation-method">Mutation method</label>
        <select
          id="mutation-method"
          :value="modelValue.mutationMethod"
          @change="update('mutationMethod', ($event.target as HTMLSelectElement).value)"
        >
          <option value="bitflip">bitflip</option>
          <option value="polynomial">polynomial</option>
          <option value="gaussian">gaussian</option>
        </select>
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
  randomSeed: number
  probCx: number
  probMt: number
  perGeneProb: number
  selectionMethod: string
  crossoverMethod: string
  mutationMethod: string
}

const props = defineProps<{ modelValue: Step2Value }>()
const emit = defineEmits<{ 'update:modelValue': [v: Step2Value] }>()

const touched = reactive({ name: false, populationSize: false, numberOfGenerations: false })

function touch(field: keyof typeof touched) { touched[field] = true }

function update(field: keyof Step2Value, value: string) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}

function updateNum(field: keyof Step2Value, raw: string, kind: 'int' | 'float') {
  const v = kind === 'int' ? (parseInt(raw, 10) || 0) : (parseFloat(raw) || 0)
  emit('update:modelValue', { ...props.modelValue, [field]: v })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--color-text-muted); border-bottom: 1px solid var(--color-border); padding-bottom: 4px; margin-top: 4px;
}
.fields-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.fields-row.half { display: grid; grid-template-columns: 1fr; gap: 12px; }
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
