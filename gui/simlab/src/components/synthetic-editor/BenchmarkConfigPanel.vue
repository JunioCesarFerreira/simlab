<template>
  <div class="config-panel">
    <div class="panel-header">
      <span class="panel-title">Benchmark configuration</span>
    </div>

    <!-- Benchmark selector -->
    <div class="field-group">
      <label class="field-label">Benchmark function</label>
      <div class="bench-cards">
        <button
          v-for="b in benchmarks"
          :key="b.id"
          class="bench-card"
          :class="{ active: draft.benchmark === b.id }"
          @click="set({ benchmark: b.id as BenchmarkId })"
        >
          <span class="bench-name">{{ b.label }}</span>
          <span class="bench-tag">M{{ b.max_objectives ? '=2' : `≥2` }}</span>
        </button>
      </div>
      <p v-if="activeBench" class="bench-desc">{{ activeBench.description }}</p>
    </div>

    <div class="divider">Decision space</div>

    <!-- M — Objectives -->
    <div class="field-row">
      <div class="field-group">
        <label class="field-label">
          Objectives (M)
          <span v-if="mLocked" class="locked-tag">locked</span>
        </label>
        <input
          type="number" :value="draft.M" min="2" max="10" step="1"
          :disabled="mLocked"
          @input="set({ M: clampInt($event, 2, 10) })"
        />
        <span class="hint">{{ activeBench?.n_min_formula }}</span>
      </div>
      <div class="field-group">
        <label class="field-label">
          Variables (n)
          <span v-if="nLocked" class="locked-tag">locked</span>
        </label>
        <input
          type="number" :value="draft.nVars" :min="nMin" max="100" step="1"
          :disabled="nLocked"
          @input="set({ nVars: clampInt($event, nMin, 100) })"
        />
        <span class="hint">{{ nLocked ? 'SCH1 uses only x₀' : `n_relays = ${nRelays}` }}</span>
      </div>
    </div>

    <!-- Noise -->
    <div class="field-group">
      <label class="field-label">Gaussian noise σ</label>
      <input
        type="number" :value="draft.noiseStd" min="0" step="0.01"
        @input="set({ noiseStd: clampFloat($event, 0) })"
      />
      <span class="hint">Added to each objective after evaluation. 0 = deterministic.</span>
    </div>

    <div class="divider">Search region Ω ⊂ ℝ²</div>

    <div class="field-row">
      <div class="field-group">
        <label class="field-label">x min</label>
        <input
type="number" :value="draft.region[0]" step="1"
          @input="updateRegion(0, $event)" />
      </div>
      <div class="field-group">
        <label class="field-label">y min</label>
        <input
type="number" :value="draft.region[1]" step="1"
          @input="updateRegion(1, $event)" />
      </div>
    </div>
    <div class="field-row">
      <div class="field-group">
        <label class="field-label">x max</label>
        <input
type="number" :value="draft.region[2]" step="1"
          @input="updateRegion(2, $event)" />
      </div>
      <div class="field-group">
        <label class="field-label">y max</label>
        <input
type="number" :value="draft.region[3]" step="1"
          @input="updateRegion(3, $event)" />
      </div>
    </div>
    <p v-if="regionErr" class="err">{{ regionErr }}</p>

    <div class="summary-box">
      <div class="summary-row">
        <span class="summary-key">Genome length</span>
        <span class="summary-val">{{ 2 * nRelays }} floats ({{ nRelays }} relay pairs)</span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Benchmark</span>
        <span class="summary-val">{{ draft.benchmark }} · M={{ draft.M }} · n={{ draft.nVars }}</span>
      </div>
      <div class="summary-row">
        <span class="summary-key">Noise σ</span>
        <span class="summary-val">{{ draft.noiseStd === 0 ? 'none (deterministic)' : draft.noiseStd }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useSyntheticStore, type BenchmarkId } from '../../app/stores/syntheticStore'
import { getBenchmarks, type BenchmarkInfo } from '../../api/synthetic'

const store = useSyntheticStore()
const draft = computed(() => store.draft)
const nRelays = computed(() => store.nRelays())

const benchmarks = ref<BenchmarkInfo[]>([
  { id: 'DTLZ2', label: 'DTLZ2', min_objectives: 2, max_objectives: null,
    description: 'Scalable benchmark — Pareto front is a unit hypersphere segment in Q₁. Supports any M ≥ 2. Reference: Deb et al. (2005).', n_min_formula: 'n ≥ M-1' },
  { id: 'ZDT1',  label: 'ZDT1',  min_objectives: 2, max_objectives: 2,
    description: 'Two-objective benchmark with a convex Pareto front. f1 = x1, f2 = g·(1 − √(f1/g)). Reference: Zitzler, Deb & Thiele (2000).', n_min_formula: 'n ≥ 2' },
  { id: 'SCH1',  label: 'SCH1',  min_objectives: 2, max_objectives: 2,
    description: 'Schaffer bilinear — two objectives, one effective variable. f1 = x², f2 = (x−2)². Reference: Schaffer (1985).', n_min_formula: 'n ≥ 1' },
])

onMounted(async () => {
  try {
    benchmarks.value = await getBenchmarks()
  } catch { /* use static fallback */ }
})

const activeBench = computed(() =>
  benchmarks.value.find(b => b.id === draft.value.benchmark) ?? null
)
const mLocked = computed(() =>
  draft.value.benchmark === 'ZDT1' || draft.value.benchmark === 'SCH1'
)
const nLocked = computed(() => draft.value.benchmark === 'SCH1')
const nMin = computed(() => {
  if (draft.value.benchmark === 'SCH1') return 1
  if (draft.value.benchmark === 'ZDT1') return 2
  return Math.max(1, draft.value.M - 1)
})
const regionErr = computed(() => {
  const [x1, y1, x2, y2] = draft.value.region
  if (x2 <= x1) return 'x max must be greater than x min.'
  if (y2 <= y1) return 'y max must be greater than y min.'
  return ''
})

function set(patch: Parameters<typeof store.setDraft>[0]) {
  store.setDraft(patch)
}

function clampInt(ev: Event, min: number, max: number): number {
  const v = parseInt((ev.target as HTMLInputElement).value, 10)
  return isNaN(v) ? min : Math.min(max, Math.max(min, v))
}

function clampFloat(ev: Event, min: number): number {
  const v = parseFloat((ev.target as HTMLInputElement).value)
  return isNaN(v) ? min : Math.max(min, v)
}

function updateRegion(idx: number, ev: Event) {
  const v = parseFloat((ev.target as HTMLInputElement).value)
  if (isNaN(v)) return
  const r: [number, number, number, number] = [...draft.value.region]
  r[idx] = v
  store.setDraft({ region: r })
}
</script>

<style scoped>
.config-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
  overflow-y: auto;
  height: 100%;
}
.panel-header {
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 10px;
}
.panel-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-text-muted);
}
.field-group { display: flex; flex-direction: column; gap: 4px; }
.field-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--color-text);
  display: flex;
  align-items: center;
  gap: 6px;
}
.locked-tag {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-muted);
  background: var(--color-surface-hover);
  border: 1px solid var(--color-border);
  border-radius: 3px;
  padding: 1px 5px;
}
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.hint { font-size: 11px; color: var(--color-text-muted); }
.err { font-size: 11px; color: #ef4444; }
.divider {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-text-muted);
  border-bottom: 1px solid var(--color-border);
  padding-bottom: 4px;
  margin-top: 4px;
}

/* Benchmark cards */
.bench-cards { display: flex; gap: 6px; flex-wrap: wrap; }
.bench-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 8px 14px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-bg);
  cursor: pointer;
  transition: border-color 0.12s, background 0.12s;
  min-width: 70px;
}
.bench-card:hover { border-color: #f59e0b; }
.bench-card.active {
  border-color: #f59e0b;
  background: rgba(245, 158, 11, 0.08);
}
.bench-name { font-size: 13px; font-weight: 700; color: var(--color-text); }
.bench-tag { font-size: 10px; color: var(--color-text-muted); }
.bench-desc {
  font-size: 11px;
  color: var(--color-text-muted);
  line-height: 1.5;
  margin-top: 2px;
}

/* Summary */
.summary-box {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.summary-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}
.summary-key { color: var(--color-text-muted); }
.summary-val { font-weight: 600; color: var(--color-text); font-family: monospace; }

input {
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  font-size: 13px;
  color: var(--color-text);
  background: var(--color-surface);
  outline: none;
  width: 100%;
  transition: border-color 0.12s;
}
input:focus { border-color: #f59e0b; }
input:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
