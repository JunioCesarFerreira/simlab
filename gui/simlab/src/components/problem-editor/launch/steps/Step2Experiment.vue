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
        <optgroup label="NSGA">
          <option value="nsga2">NSGA-II</option>
          <option value="nsga3">NSGA-III (native)</option>
          <option value="nsga3_deap">NSGA-III + DEAP</option>
          <option value="nsga3_pymoo">NSGA-III + pymoo</option>
        </optgroup>
        <optgroup label="Baseline">
          <option value="random_search">Random Search</option>
        </optgroup>
      </select>
      <span class="strategy-hint">{{ strategyHint }}</span>
    </div>

    <!-- Source Repository Options -->
    <div class="section-divider">Source repository options <span class="required">*</span></div>

    <p class="hint">
      Map each MAC protocol to a source repository (firmware). The engine uses this to build and
      run simulations. If only one mapping is provided, the algorithm will use it regardless of the
      chromosome's MAC protocol.
    </p>

    <div class="repo-list">
      <div v-for="(opt, i) in modelValue.sourceOptions" :key="i" class="repo-row">
        <select
          :value="opt.protocol"
          class="proto-select"
          @change="updateSourceOpt(i, 'protocol', ($event.target as HTMLSelectElement).value)"
        >
          <option value="csma">csma</option>
          <option value="tsch">tsch</option>
        </select>

        <span class="arrow">→</span>

        <div class="repo-select-wrap">
          <div v-if="repositoriesLoading" class="repo-loading">Loading…</div>
          <select
            v-else
            :value="opt.repoId"
            :class="{ invalid: showValidation && !opt.repoId }"
            @change="updateSourceOpt(i, 'repoId', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">Select repository…</option>
            <option v-for="r in repositories" :key="r.id" :value="r.id">
              {{ r.name }}
              <template v-if="r.description"> — {{ r.description }}</template>
            </option>
          </select>
          <span v-if="showValidation && !opt.repoId" class="err">Required.</span>
        </div>

        <button
          class="remove-repo-btn"
          :disabled="modelValue.sourceOptions.length <= 1"
          @click="removeSourceOpt(i)"
          title="Remove mapping"
        >×</button>
      </div>
    </div>

    <button class="add-repo-btn" @click="addSourceOpt">+ Add protocol mapping</button>

    <!-- Algorithm Hyperparameters -->
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

    <div v-if="isNsga3" class="fields-row half">
      <div class="field-group">
        <label class="field-label" for="divisions">Reference point divisions</label>
        <input
          id="divisions"
          :value="modelValue.divisions"
          type="number" min="1" step="1"
          @input="updateNum('divisions', ($event.target as HTMLInputElement).value, 'int')"
        />
        <span class="hint-small">Partitions per objective axis for NSGA-III niching (das-Dennis).</span>
      </div>
    </div>

    <template v-if="isEvolutionary">
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
    </template>

    <template v-else>
      <div class="fields-row half">
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
    </template>
  </div>
</template>

<script setup lang="ts">
import { reactive, computed } from 'vue'
import type { SourceRepositoryDto } from '../../../../types/simlab'

export interface SourceOption {
  protocol: string
  repoId: string
}

export interface Step2Value {
  name: string
  strategy: string
  sourceOptions: SourceOption[]
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
const EVOLUTIONARY_STRATEGIES = ['nsga2', 'nsga3', 'nsga3_deap', 'nsga3_pymoo']

const STRATEGY_HINTS: Record<string, string> = {
  nsga2:        'NSGA-II — crowding-distance selection, suitable for 2-3 objectives.',
  nsga3:        'NSGA-III — reference-point niching, better for 3+ objectives (native implementation).',
  nsga3_deap:   'NSGA-III — environmental selection via DEAP\'s selNSGA3. Requires deap ≥ 1.3.',
  nsga3_pymoo:  'NSGA-III — environmental selection via pymoo\'s ReferenceDirectionSurvival. Requires pymoo ≥ 0.6.',
  random_search:'Random Search — samples the decision space uniformly; no selection or variation. Useful as a baseline.',
}

const props = defineProps<{
  modelValue: Step2Value
  repositories: SourceRepositoryDto[]
  repositoriesLoading: boolean
  showValidation: boolean
}>()
const emit = defineEmits<{ 'update:modelValue': [v: Step2Value] }>()

const isNsga3 = computed(() => NSGA3_STRATEGIES.includes(props.modelValue.strategy))
const isEvolutionary = computed(() => EVOLUTIONARY_STRATEGIES.includes(props.modelValue.strategy))
const strategyHint = computed(() => STRATEGY_HINTS[props.modelValue.strategy] ?? '')

const touched = reactive({ name: false, populationSize: false, numberOfGenerations: false })

function touch(field: keyof typeof touched) { touched[field] = true }

function update(field: keyof Step2Value, value: string) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}

function updateNum(field: keyof Step2Value, raw: string, kind: 'int' | 'float') {
  const v = kind === 'int' ? (parseInt(raw, 10) || 0) : (parseFloat(raw) || 0)
  emit('update:modelValue', { ...props.modelValue, [field]: v })
}

function updateSourceOpt(i: number, field: keyof SourceOption, value: string) {
  const sourceOptions = props.modelValue.sourceOptions.map((o, idx) =>
    idx === i ? { ...o, [field]: value } : o
  )
  emit('update:modelValue', { ...props.modelValue, sourceOptions })
}

function removeSourceOpt(i: number) {
  if (props.modelValue.sourceOptions.length <= 1) return
  const sourceOptions = props.modelValue.sourceOptions.filter((_, idx) => idx !== i)
  emit('update:modelValue', { ...props.modelValue, sourceOptions })
}

function addSourceOpt() {
  const usedProtos = new Set(props.modelValue.sourceOptions.map(o => o.protocol))
  const next = usedProtos.has('csma') ? 'tsch' : 'csma'
  const sourceOptions = [...props.modelValue.sourceOptions, { protocol: next, repoId: '' }]
  emit('update:modelValue', { ...props.modelValue, sourceOptions })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; margin-top: -6px; }
.strategy-hint { font-size: 11px; color: var(--color-text-muted); line-height: 1.4; }
.hint-small { font-size: 11px; color: var(--color-text-muted); }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--color-text-muted); border-bottom: 1px solid var(--color-border); padding-bottom: 4px; margin-top: 4px;
}
.fields-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.fields-row.half { display: grid; grid-template-columns: 1fr; }
.field-group { display: flex; flex-direction: column; gap: 4px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }

/* Source repo rows */
.repo-list { display: flex; flex-direction: column; gap: 8px; }
.repo-row { display: grid; grid-template-columns: 90px 20px 1fr 32px; gap: 8px; align-items: start; }
.arrow { font-size: 14px; color: var(--color-text-muted); align-self: center; text-align: center; }
.repo-select-wrap { display: flex; flex-direction: column; gap: 3px; }
.repo-loading { font-size: 12px; color: var(--color-text-muted); padding: 7px 0; }
.proto-select { width: 100%; }
.remove-repo-btn {
  width: 32px; height: 32px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  background: none; color: #ef4444; font-size: 16px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
}
.remove-repo-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.add-repo-btn {
  align-self: flex-start; padding: 6px 12px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 12px; color: var(--color-text); cursor: pointer;
}
.add-repo-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }

input, select {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface);
  outline: none; transition: border-color 0.12s; width: 100%;
}
input:focus, select:focus { border-color: var(--color-primary); }
input.invalid, select.invalid { border-color: #ef4444; }
.err { font-size: 11px; color: #ef4444; }
</style>
