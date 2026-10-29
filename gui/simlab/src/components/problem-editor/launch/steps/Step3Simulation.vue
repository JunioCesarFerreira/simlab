<template>
  <div class="step">
    <div class="field-group">
      <label class="field-label" for="sim-duration">Simulation duration (seconds) <span class="required">*</span></label>
      <input
        id="sim-duration"
        :value="modelValue.duration"
        type="number" min="1" step="1"
        @input="updateDuration(($event.target as HTMLInputElement).value)"
      />
    </div>

    <div class="section-divider">Random seeds</div>

    <p class="hint">
      Each seed runs an independent simulation per generation, improving the statistical robustness of results.
    </p>

    <div class="field-group">
      <label class="field-label">Seeds <span class="seed-count">({{ modelValue.randomSeeds.length }})</span> <span class="required">*</span></label>
      <div class="seeds-row">
        <div class="chips">
          <span v-for="(seed, i) in modelValue.randomSeeds" :key="i" class="chip">
            {{ seed }}
            <button class="chip-remove" title="Remove seed" @click="removeSeed(i)">×</button>
          </span>
          <span v-if="modelValue.randomSeeds.length === 0" class="empty-chips">No seeds added.</span>
        </div>
        <div class="add-row">
          <input
            v-model="newSeed"
            type="number"
            step="1"
            placeholder="e.g. 42"
            @keydown.enter.prevent="addSeed"
          />
          <button class="add-btn" :disabled="!canAdd" @click="addSeed">+ Add</button>
          <button class="rand-btn" title="Insert a random seed" @click="addRandomSeed">⚄ Random</button>
        </div>
      </div>
      <span v-if="showError" class="err">Add at least one seed.</span>
    </div>

    <div class="presets">
      <span class="presets-label">Quick presets:</span>
      <button v-for="preset in PRESETS" :key="preset.label" class="preset-btn" @click="applyPreset(preset.seeds)">
        {{ preset.label }}
      </button>
    </div>

    <!-- Import / Export -->
    <div class="io-row">
      <span class="io-label">Seeds file:</span>
      <button
        class="io-btn"
        :disabled="modelValue.randomSeeds.length === 0"
        title="Export current seeds to a JSON file"
        @click="exportSeeds"
      >
        ↓ Export
      </button>
      <button
        class="io-btn"
        title="Import seeds from a JSON file (replaces current list)"
        @click="triggerImport"
      >
        ↑ Import
      </button>
      <span v-if="importFeedback" :class="['io-feedback', importOk ? 'io-feedback--ok' : 'io-feedback--err']">
        {{ importFeedback }}
      </span>
      <input
        ref="fileInputEl"
        type="file"
        accept=".json"
        class="hidden-input"
        @change="handleImport"
      />
    </div>

    <div class="section-divider">Seed aggregation</div>

    <p class="hint">
      How the seeds of one individual are combined into a single value per metric — the objective
      value the optimizer compares.
    </p>

    <div class="field-group">
      <label class="field-label" for="sim-aggregator">Aggregation function</label>
      <select
        id="sim-aggregator"
        :value="modelValue.aggregator"
        @change="updateAggregator(($event.target as HTMLSelectElement).value as SeedAggregatorKind)"
      >
        <option v-for="opt in SEED_AGGREGATORS" :key="opt.kind" :value="opt.kind">{{ opt.label }}</option>
      </select>
      <span class="agg-hint">{{ aggregatorHint }}</span>
    </div>

    <div v-if="modelValue.aggregator === 'trimmed_mean'" class="field-group">
      <label class="field-label" for="sim-trim">Trim per end (fraction) <span class="required">*</span></label>
      <input
        id="sim-trim"
        :value="modelValue.trim"
        type="number" min="0" max="0.49" step="0.05"
        @input="updateTrim(($event.target as HTMLInputElement).value)"
      />
      <span v-if="trimError" class="err">Trim must be at least 0 and below 0.5.</span>
      <span v-else class="agg-hint">{{ trimSummary }}</span>
    </div>

    <p v-if="extremeAggregator" class="agg-note warn-banner">
      Applied to every metric whatever its objective's goal: the {{ extremeAggregator.name }} is the
      worst case for a metric you {{ extremeAggregator.worstFor }} and the best case for one you
      {{ extremeAggregator.bestFor }}.
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onBeforeUnmount } from 'vue'
import type { SeedAggregatorKind } from '../../../../types/simlab'
import { SEED_AGGREGATORS, isValidTrim, trimmedPerSide } from '../../../../lib/seedAggregator'

export interface Step3Value {
  duration: number
  randomSeeds: number[]
  aggregator: SeedAggregatorKind
  /** Only used when aggregator is trimmed_mean. */
  trim: number
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
  { label: 'p2 example (8 seeds)', seeds: [336157, 667370, 35239, 873465, 987654, 123456, 493499, 5343] },
]

function updateDuration(raw: string) {
  emit('update:modelValue', { ...props.modelValue, duration: parseInt(raw, 10) || 1 })
}

// ── Seed aggregation ─────────────────────────────────────────────────────────

const aggregatorHint = computed(
  () => SEED_AGGREGATORS.find((o) => o.kind === props.modelValue.aggregator)?.hint ?? '',
)
const trimError = computed(() => !isValidTrim(props.modelValue.trim))

const trimSummary = computed(() => {
  const n = props.modelValue.randomSeeds.length
  if (n === 0) return 'Fraction of the seeds dropped from each end.'
  const k = trimmedPerSide(props.modelValue.trim, n)
  if (k === 0) return `With ${n} seed${n !== 1 ? 's' : ''} this drops nothing, so it equals the mean.`
  return `With ${n} seeds, drops ${k} from each end and averages the remaining ${n - 2 * k}.`
})

const extremeAggregator = computed(() => {
  if (props.modelValue.aggregator === 'max') return { name: 'maximum', worstFor: 'minimize', bestFor: 'maximize' }
  if (props.modelValue.aggregator === 'min') return { name: 'minimum', worstFor: 'maximize', bestFor: 'minimize' }
  return null
})

function updateAggregator(kind: SeedAggregatorKind) {
  emit('update:modelValue', { ...props.modelValue, aggregator: kind })
}

function updateTrim(raw: string) {
  emit('update:modelValue', { ...props.modelValue, trim: parseFloat(raw) })
}

function addSeed() {
  const v = parseInt(newSeed.value, 10)
  if (!Number.isInteger(v) || props.modelValue.randomSeeds.includes(v)) return
  emit('update:modelValue', { ...props.modelValue, randomSeeds: [...props.modelValue.randomSeeds, v] })
  newSeed.value = ''
}

function removeSeed(i: number) {
  const seeds = [...props.modelValue.randomSeeds]
  seeds.splice(i, 1)
  emit('update:modelValue', { ...props.modelValue, randomSeeds: seeds })
}

function applyPreset(seeds: number[]) {
  emit('update:modelValue', { ...props.modelValue, randomSeeds: seeds })
}

function addRandomSeed() {
  for (let attempts = 0; attempts < 50; attempts++) {
    const v = Math.floor(Math.random() * 999983) + 1
    if (!props.modelValue.randomSeeds.includes(v)) {
      emit('update:modelValue', { ...props.modelValue, randomSeeds: [...props.modelValue.randomSeeds, v] })
      return
    }
  }
}

// ── Import / Export ──────────────────────────────────────────────────────────

const fileInputEl = ref<HTMLInputElement | null>(null)
const importFeedback = ref('')
const importOk = ref(true)
let feedbackTimer: ReturnType<typeof setTimeout> | null = null

onBeforeUnmount(() => { if (feedbackTimer) clearTimeout(feedbackTimer) })

function flash(msg: string, ok: boolean) {
  importFeedback.value = msg
  importOk.value = ok
  if (feedbackTimer) clearTimeout(feedbackTimer)
  feedbackTimer = setTimeout(() => { importFeedback.value = '' }, 3000)
}

function exportSeeds() {
  const content = JSON.stringify({ seeds: props.modelValue.randomSeeds }, null, 2)
  const blob = new Blob([content], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'seeds.json'
  a.click()
  URL.revokeObjectURL(url)
}

function triggerImport() {
  if (!fileInputEl.value) return
  fileInputEl.value.value = ''   // allow re-importing the same file
  fileInputEl.value.click()
}

function handleImport(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return

  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const raw = JSON.parse(e.target?.result as string)
      // Accept either { seeds: [...] } or a bare array
      const arr: unknown = Array.isArray(raw) ? raw : raw?.seeds
      if (!Array.isArray(arr) || arr.length === 0) {
        flash('Invalid file — expected { "seeds": [...] }', false)
        return
      }
      const seeds = arr.map((v) => parseInt(String(v), 10))
      if (seeds.some((v) => !Number.isInteger(v) || isNaN(v))) {
        flash('Invalid file — seeds must be integers', false)
        return
      }
      const unique = [...new Set(seeds)]
      emit('update:modelValue', { ...props.modelValue, randomSeeds: unique })
      flash(`✓ Imported ${unique.length} seed${unique.length !== 1 ? 's' : ''}`, true)
    } catch {
      flash('Invalid JSON file', false)
    }
  }
  reader.readAsText(file)
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--color-text-muted); border-bottom: 1px solid var(--color-border); padding-bottom: 4px; margin-top: 4px;
}
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
input {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none; width: 100%;
}
input:focus { border-color: var(--color-primary); }
select {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface); outline: none; width: 100%;
}
select:focus { border-color: var(--color-primary); }
.agg-hint { font-size: 11px; color: var(--color-text-muted); line-height: 1.4; }
.agg-note {
  font-size: 12px; line-height: 1.5; color: #b45309;
  background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.35);
  border-radius: var(--radius-sm); padding: 8px 10px; margin: 0;
}
.add-btn {
  padding: 7px 14px; background: var(--color-primary); color: #fff;
  border-radius: var(--radius-sm); font-size: 13px; font-weight: 600; border: none; cursor: pointer;
  white-space: nowrap;
}
.add-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.rand-btn {
  padding: 7px 13px; background: none; color: #7c3aed;
  border: 1px solid #7c3aed; border-radius: var(--radius-sm);
  font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap;
  transition: background 0.12s, color 0.12s;
}
.rand-btn:hover { background: rgba(124,58,237,0.1); }
.seed-count { color: var(--color-text-muted); font-weight: 400; }
.presets { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.presets-label { font-size: 11px; color: var(--color-text-muted); }
.preset-btn {
  padding: 4px 10px; font-size: 11px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  color: var(--color-text); cursor: pointer;
}
.preset-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }
.err { font-size: 11px; color: #ef4444; }

/* ── Import / Export ──────────────────────────────────────────────────────── */

.io-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.io-label {
  font-size: 11px;
  color: var(--color-text-muted);
}

.io-btn {
  padding: 4px 10px;
  font-size: 11px;
  font-weight: 600;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text);
  cursor: pointer;
  transition: border-color 0.12s, color 0.12s;
}

.io-btn:hover:not(:disabled) {
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.io-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.io-feedback {
  font-size: 11px;
  font-weight: 500;
}

.io-feedback--ok  { color: #16a34a; }
.io-feedback--err { color: #ef4444; }

.hidden-input { display: none; }
</style>
