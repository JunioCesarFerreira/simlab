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
          title="Remove mapping"
          @click="removeSourceOpt(i)"
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

      <div class="section-divider">Genetic operators — {{ problemName }}</div>

      <div class="fields-row">
        <div class="field-group">
          <label class="field-label">Selection method</label>
          <div class="fixed-op">tournament <span class="fixed-tag">fixed</span></div>
          <span class="hint-small">All strategies currently use tournament selection.</span>
        </div>
        <div class="field-group">
          <label class="field-label" for="crossover-method">Crossover method</label>
          <select
            v-if="caps.crossoverMethods.length"
            id="crossover-method"
            :value="modelValue.crossoverMethod"
            @change="update('crossoverMethod', ($event.target as HTMLSelectElement).value)"
          >
            <option v-for="m in caps.crossoverMethods" :key="m" :value="m">{{ m }}</option>
          </select>
          <div v-else class="fixed-op">{{ caps.fixedCrossoverLabel }} <span class="fixed-tag">fixed</span></div>
        </div>
      </div>

      <div class="fields-row half">
        <div class="field-group">
          <label class="field-label">Mutation method</label>
          <div class="fixed-op">{{ caps.fixedMutationLabel }} <span class="fixed-tag">fixed</span></div>
        </div>
      </div>

      <div v-if="caps.supportsCoverageRepair" class="fields-row half">
        <div class="field-group">
          <label class="checkbox-label" for="apply-coverage-repair">
            <input
              id="apply-coverage-repair"
              type="checkbox"
              :checked="modelValue.applyCoverageRepair"
              @change="updateBool('applyCoverageRepair', ($event.target as HTMLInputElement).checked)"
            />
            Apply coverage repair
          </label>
          <span class="hint-small">
            Repair individuals whose trajectory coverage falls below the problem threshold.
            When disabled, infeasible individuals are only penalized.
          </span>
        </div>
      </div>

      <template v-if="caps.extraParams.length">
        <div class="section-divider">Problem-specific parameters</div>
        <div class="fields-row extras">
          <div v-for="p in caps.extraParams" :key="p.key" class="field-group">
            <label class="field-label" :for="`extra-${p.key}`">{{ p.label }}</label>
            <input
              :id="`extra-${p.key}`"
              type="number"
              :min="p.min" :max="p.max" :step="p.step"
              :value="modelValue.extraParams[p.key]"
              @input="updateExtra(p.key, ($event.target as HTMLInputElement).value)"
            />
            <span v-if="p.hint" class="hint-small">{{ p.hint }}</span>
          </div>
        </div>
      </template>
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
import { capabilitiesFor } from '../../../../lib/problemCapabilities'
import { referencePointCount, suggestedPopulationSize } from '../../../../lib/nsga3'

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
  crossoverMethod: string
  divisions: number
  applyCoverageRepair: boolean
  /** Problem-specific numeric hyperparameters (see problemCapabilities.ts). */
  extraParams: Record<string, number>
}

const NSGA3_STRATEGIES = ['nsga3', 'nsga3_deap', 'nsga3_pymoo']
const EVOLUTIONARY_STRATEGIES = ['nsga2', 'nsga2_deap', 'nsga2_pymoo', 'nsga3', 'nsga3_deap', 'nsga3_pymoo']

const STRATEGY_HINTS: Record<string, string> = {
  nsga2:        'NSGA-II — crowding-distance selection, suitable for 2–3 objectives (native implementation).',
  nsga2_deap:   'NSGA-II — environmental selection via DEAP\'s selNSGA2. Requires deap ≥ 1.3.',
  nsga2_pymoo:  'NSGA-II — environmental selection via pymoo\'s RankAndCrowding. Requires pymoo ≥ 0.6.',
  nsga3:        'NSGA-III — reference-point niching, better for 3+ objectives (native implementation).',
  nsga3_deap:   'NSGA-III — environmental selection via DEAP\'s selNSGA3. Requires deap ≥ 1.3.',
  nsga3_pymoo:  'NSGA-III — environmental selection via pymoo\'s ReferenceDirectionSurvival. Requires pymoo ≥ 0.6.',
  random_search:'Random Search — samples the decision space uniformly; no selection or variation. Useful as a baseline.',
}

const props = defineProps<{
  modelValue: Step2Value
  problemName: string
  objectivesCount: number
  repositories: SourceRepositoryDto[]
  repositoriesLoading: boolean
  showValidation: boolean
}>()
const emit = defineEmits<{ 'update:modelValue': [v: Step2Value] }>()

const refPoints = computed(() => referencePointCount(props.objectivesCount, props.modelValue.divisions))
const suggestedPop = computed(() => suggestedPopulationSize(refPoints.value))
const refPointsExceedPop = computed(() => refPoints.value > 0 && props.modelValue.populationSize < refPoints.value)

const caps = computed(() => capabilitiesFor(props.problemName))
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

function updateBool(field: keyof Step2Value, value: boolean) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}

function updateExtra(key: string, raw: string) {
  const extraParams = { ...props.modelValue.extraParams, [key]: parseFloat(raw) || 0 }
  emit('update:modelValue', { ...props.modelValue, extraParams })
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
.checkbox-label {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px; font-weight: 500; color: var(--color-text); cursor: pointer;
}
.checkbox-label input[type="checkbox"] { width: auto; margin: 0; cursor: pointer; }
.required { color: var(--color-primary); }
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
.fields-row.extras { grid-template-columns: 1fr 1fr; }
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
