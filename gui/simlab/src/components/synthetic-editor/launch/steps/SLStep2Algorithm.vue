<template>
  <div class="step">
    <!-- Name -->
    <div class="field-group">
      <label class="field-label" for="exp-name">Experiment name <span class="required">*</span></label>
      <input
        id="exp-name"
        :value="modelValue.name"
        type="text"
        placeholder="e.g. dtlz2-nsga3-m3-n10-run01"
        :class="{ invalid: touched.name && !modelValue.name.trim() }"
        @input="update('name', ($event.target as HTMLInputElement).value)"
        @blur="touch('name')"
      />
      <span v-if="touched.name && !modelValue.name.trim()" class="err">Name is required.</span>
    </div>

    <!-- Strategy -->
    <div class="field-group">
      <label class="field-label" for="exp-strategy">Optimization strategy</label>
      <select
        id="exp-strategy"
        :value="modelValue.strategy"
        @change="update('strategy', ($event.target as HTMLSelectElement).value)"
      >
        <optgroup label="NSGA-II">
          <option value="nsga2">NSGA-II (native)</option>
          <option value="nsga2_deap">NSGA-II + DEAP</option>
          <option value="nsga2_pymoo">NSGA-II + pymoo</option>
        </optgroup>
        <optgroup label="NSGA-III">
          <option value="nsga3">NSGA-III (native)</option>
          <option value="nsga3_deap">NSGA-III + DEAP</option>
          <option value="nsga3_pymoo">NSGA-III + pymoo</option>
        </optgroup>
        <optgroup label="Baseline">
          <option value="random_search">Random Search</option>
        </optgroup>
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
        <label class="field-label" for="num-gen">Generations <span class="required">*</span></label>
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

    <div v-if="isNsga3" class="fields-row half">
      <div class="field-group">
        <label class="field-label" for="divisions">Reference point divisions</label>
        <input
          id="divisions"
          :value="modelValue.divisions"
          type="number" min="1" step="1"
          @input="updateNum('divisions', ($event.target as HTMLInputElement).value, 'int')"
        />
        <span class="hint-small">Das-Dennis reference points for NSGA-III niching.</span>
      </div>
    </div>

    <template v-if="isEvolutionary">
      <div class="fields-row">
        <div class="field-group">
          <label class="field-label" for="prob-cx">Crossover prob.</label>
          <input
id="prob-cx" :value="modelValue.probCx" type="number" min="0" max="1" step="0.01"
            @input="updateNum('probCx', ($event.target as HTMLInputElement).value, 'float')" />
        </div>
        <div class="field-group">
          <label class="field-label" for="prob-mt">Mutation prob.</label>
          <input
id="prob-mt" :value="modelValue.probMt" type="number" min="0" max="1" step="0.01"
            @input="updateNum('probMt', ($event.target as HTMLInputElement).value, 'float')" />
        </div>
      </div>
      <div class="fields-row">
        <div class="field-group">
          <label class="field-label" for="per-gene-prob">Per-gene mutation prob.</label>
          <input
id="per-gene-prob" :value="modelValue.perGeneProb" type="number" min="0" max="1" step="0.01"
            @input="updateNum('perGeneProb', ($event.target as HTMLInputElement).value, 'float')" />
        </div>
        <div class="field-group">
          <label class="field-label" for="random-seed">Random seed</label>
          <input
id="random-seed" :value="modelValue.randomSeed" type="number" step="1"
            @input="updateNum('randomSeed', ($event.target as HTMLInputElement).value, 'int')" />
        </div>
      </div>
      <div class="fields-row">
        <div class="field-group">
          <label class="field-label" for="sel-method">Selection method</label>
          <select
id="sel-method" :value="modelValue.selectionMethod"
            @change="update('selectionMethod', ($event.target as HTMLSelectElement).value)">
            <option value="tournament">tournament</option>
            <option value="roulette">roulette</option>
          </select>
        </div>
        <div class="field-group">
          <label class="field-label" for="cx-method">Crossover method</label>
          <select
id="cx-method" :value="modelValue.crossoverMethod"
            @change="update('crossoverMethod', ($event.target as HTMLSelectElement).value)">
            <option value="uniform_mask">uniform_mask</option>
            <option value="sbx_with_radial_translate">sbx_with_radial_translate</option>
            <option value="one_point">one_point</option>
            <option value="two_point">two_point</option>
          </select>
        </div>
      </div>
      <div class="fields-row half">
        <div class="field-group">
          <label class="field-label" for="mt-method">Mutation method</label>
          <select
id="mt-method" :value="modelValue.mutationMethod"
            @change="update('mutationMethod', ($event.target as HTMLSelectElement).value)">
            <option value="bitflip">bitflip</option>
            <option value="polynomial">polynomial</option>
            <option value="gaussian">gaussian</option>
          </select>
        </div>
      </div>
    </template>

    <template v-else>
      <div class="fields-row half">
        <div class="field-group">
          <label class="field-label" for="random-seed">Random seed</label>
          <input
id="random-seed" :value="modelValue.randomSeed" type="number" step="1"
            @input="updateNum('randomSeed', ($event.target as HTMLInputElement).value, 'int')" />
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { reactive, computed } from 'vue'

export interface SLStep2Value {
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
  divisions: number
}

const NSGA3_STRATEGIES = ['nsga3', 'nsga3_deap', 'nsga3_pymoo']
const EVOLUTIONARY_STRATEGIES = ['nsga2', 'nsga2_deap', 'nsga2_pymoo', 'nsga3', 'nsga3_deap', 'nsga3_pymoo']

const props = defineProps<{ modelValue: SLStep2Value; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: SLStep2Value] }>()

const isNsga3 = computed(() => NSGA3_STRATEGIES.includes(props.modelValue.strategy))
const isEvolutionary = computed(() => EVOLUTIONARY_STRATEGIES.includes(props.modelValue.strategy))

const touched = reactive({ name: false, populationSize: false, numberOfGenerations: false })
function touch(field: keyof typeof touched) { touched[field] = true }

function update(field: keyof SLStep2Value, value: string) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}
function updateNum(field: keyof SLStep2Value, raw: string, kind: 'int' | 'float') {
  const v = kind === 'int' ? (parseInt(raw, 10) || 0) : (parseFloat(raw) || 0)
  emit('update:modelValue', { ...props.modelValue, [field]: v })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--color-text-muted);
  border-bottom: 1px solid var(--color-border); padding-bottom: 4px; margin-top: 4px;
}
.fields-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.fields-row.half { grid-template-columns: 1fr; }
.field-group { display: flex; flex-direction: column; gap: 4px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
.hint-small { font-size: 11px; color: var(--color-text-muted); }
input, select {
  padding: 7px 10px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); font-size: 13px;
  color: var(--color-text); background: var(--color-surface);
  outline: none; transition: border-color 0.12s; width: 100%;
}
input:focus, select:focus { border-color: var(--color-primary); }
input.invalid, select.invalid { border-color: #ef4444; }
.err { font-size: 11px; color: #ef4444; }
</style>
