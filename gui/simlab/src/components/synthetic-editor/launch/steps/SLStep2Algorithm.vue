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
        <div class="h-calc" :class="{ warn: refPointsExceedPop }">
          <span>
            H = C(M+p−1, p) = C({{ objectivesCount + modelValue.divisions - 1 }}, {{ modelValue.divisions }}) =
            <strong>{{ refPoints }}</strong> reference points
            <span class="h-calc-detail">(M = {{ objectivesCount }} objectives, p = {{ modelValue.divisions }})</span>
          </span>
          <button
            v-if="refPointsExceedPop"
            class="h-calc-apply"
            title="NSGA-III niching needs population_size ≥ H"
            @click="updateNum('populationSize', String(suggestedPop), 'int')"
          >
            set population to {{ suggestedPop }}
          </button>
        </div>
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
          <label class="field-label" for="eta-cx">SBX distribution index (η_cx)</label>
          <input
id="eta-cx" :value="modelValue.etaCx" type="number" min="1" max="200" step="1"
            @input="updateNum('etaCx', ($event.target as HTMLInputElement).value, 'float')" />
          <span class="hint-small">Higher values keep offspring closer to the parents.</span>
        </div>
        <div class="field-group">
          <label class="field-label" for="eta-mt">Polynomial mutation index (η_mt)</label>
          <input
id="eta-mt" :value="modelValue.etaMt" type="number" min="1" max="200" step="1"
            @input="updateNum('etaMt', ($event.target as HTMLInputElement).value, 'float')" />
          <span class="hint-small">Higher values produce smaller mutation steps.</span>
        </div>
      </div>
      <div class="fields-row half">
        <div class="field-group">
          <label class="field-label">Genetic operators</label>
          <div class="fixed-op">
            tournament selection · SBX crossover · polynomial mutation
            <span class="fixed-tag">fixed</span>
          </div>
          <span class="hint-small">P0 uses the textbook real-coded operator pair; only the indices above are tunable.</span>
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
import { referencePointCount, suggestedPopulationSize } from '../../../../lib/nsga3'

export interface SLStep2Value {
  name: string
  strategy: string
  populationSize: number
  numberOfGenerations: number
  randomSeed: number
  probCx: number
  probMt: number
  perGeneProb: number
  etaCx: number
  etaMt: number
  divisions: number
}

const NSGA3_STRATEGIES = ['nsga3', 'nsga3_deap', 'nsga3_pymoo']
const EVOLUTIONARY_STRATEGIES = ['nsga2', 'nsga2_deap', 'nsga2_pymoo', 'nsga3', 'nsga3_deap', 'nsga3_pymoo']

const props = defineProps<{ modelValue: SLStep2Value; objectivesCount: number; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: SLStep2Value] }>()

const isNsga3 = computed(() => NSGA3_STRATEGIES.includes(props.modelValue.strategy))
const isEvolutionary = computed(() => EVOLUTIONARY_STRATEGIES.includes(props.modelValue.strategy))

const refPoints = computed(() => referencePointCount(props.objectivesCount, props.modelValue.divisions))
const suggestedPop = computed(() => suggestedPopulationSize(refPoints.value))
const refPointsExceedPop = computed(() => refPoints.value > 0 && props.modelValue.populationSize < refPoints.value)

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
.fixed-op {
  padding: 7px 10px; border: 1px dashed var(--color-border); border-radius: var(--radius-sm);
  font-size: 12px; color: var(--color-text-muted); background: var(--color-bg);
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.fixed-tag {
  font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--color-text-muted); border: 1px solid var(--color-border);
  border-radius: 999px; padding: 1px 7px; flex-shrink: 0;
}
.h-calc {
  display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap;
  margin-top: 2px; padding: 6px 10px; border: 1px solid var(--color-border);
  border-radius: var(--radius-sm); background: var(--color-bg);
  font-size: 11px; color: var(--color-text-muted);
}
.h-calc.warn { border-color: #fbbf24; }
.h-calc strong { color: var(--color-text); }
.h-calc-detail { color: var(--color-text-muted); }
.h-calc-apply {
  padding: 3px 8px; border: 1px solid #fbbf24; border-radius: var(--radius-sm);
  background: #fef3c7; color: #92400e; font-size: 11px; font-weight: 600;
  cursor: pointer; flex-shrink: 0;
}
.h-calc-apply:hover { background: #fde68a; }
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
