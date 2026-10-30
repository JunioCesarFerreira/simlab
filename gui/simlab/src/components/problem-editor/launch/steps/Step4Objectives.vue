<template>
  <div class="step">
    <p class="hint">
      Define the metrics the algorithm must optimize. Each objective maps to a metric calculated
      from the simulation logs (configured in the next step); its value for an individual is that
      metric combined across the seeds with the function chosen under <em>Across seeds</em>.
    </p>

    <div class="objectives-list">
      <div class="obj-row obj-head" aria-hidden="true">
        <span />
        <span>Metric</span>
        <span>Goal</span>
        <span>Across seeds</span>
        <span />
      </div>

      <div v-for="(obj, i) in modelValue" :key="i" class="obj-item">
        <div class="obj-row">
          <div class="obj-index">{{ i + 1 }}</div>
          <input
            :value="obj.metric_name"
            type="text"
            placeholder="e.g. energy, coverage, latency"
            list="metric-suggestions"
            class="obj-name"
            :aria-label="`Objective ${i + 1} metric`"
            :class="{ invalid: showValidation && !obj.metric_name.trim() }"
            @input="updateObj(i, 'metric_name', ($event.target as HTMLInputElement).value)"
          />
          <select
            :value="obj.goal"
            class="obj-goal"
            :aria-label="`Objective ${i + 1} goal`"
            @change="updateObj(i, 'goal', ($event.target as HTMLSelectElement).value)"
          >
            <option value="min">Minimize</option>
            <option value="max">Maximize</option>
          </select>
          <select
            :value="aggregatorKind(obj.aggregator)"
            class="obj-agg"
            :aria-label="`Objective ${i + 1} aggregation across seeds`"
            :title="hintOf(aggregatorKind(obj.aggregator))"
            @change="updateAggregator(i, ($event.target as HTMLSelectElement).value as SeedAggregatorKind)"
          >
            <option v-for="opt in SEED_AGGREGATORS" :key="opt.kind" :value="opt.kind">{{ opt.label }}</option>
          </select>
          <button class="remove-btn" :disabled="modelValue.length <= 1" title="Remove" @click="remove(i)">×</button>
        </div>

        <div v-if="aggregatorKind(obj.aggregator) === 'trimmed_mean'" class="obj-sub">
          <label class="sub-label" :for="`obj-trim-${i}`">Trim per end</label>
          <input
            :id="`obj-trim-${i}`"
            :value="aggregatorTrim(obj.aggregator)"
            type="number" min="0" max="0.49" step="0.05"
            class="trim-input"
            :class="{ invalid: !isValidSeedAggregator(obj.aggregator) }"
            @input="updateTrim(i, ($event.target as HTMLInputElement).value)"
          />
          <span v-if="!isValidSeedAggregator(obj.aggregator)" class="err">
            Trim must be at least 0 and below 0.5.
          </span>
          <span v-else class="sub-hint">{{ trimSummary(aggregatorTrim(obj.aggregator)) }}</span>
        </div>
        <div v-else-if="extremeOf(obj)" class="obj-sub">
          <span class="sub-hint" :class="`case--${extremeOf(obj)}`">
            {{ extremeOf(obj) === 'worst'
              ? 'Pessimistic: the individual is judged by its worst seed.'
              : 'Optimistic: the individual is judged by its best seed.' }}
          </span>
        </div>
      </div>
    </div>

    <datalist id="metric-suggestions">
      <option value="energy" />
      <option value="latency" />
      <option value="throughput" />
      <option value="coverage" />
      <option value="packet_loss" />
      <option value="lifetime" />
    </datalist>

    <button class="add-btn" @click="add">+ Add objective</button>

    <span v-if="showValidation && hasEmpty" class="err">
      All objectives must have a metric name.
    </span>
    <span v-if="conflicts.length" class="err">
      {{ conflicts.join(', ') }}: one metric cannot be combined across seeds in two ways.
    </span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ObjectiveItem, SeedAggregatorKind } from '../../../../types/simlab'
import {
  DEFAULT_SEED_AGGREGATOR,
  SEED_AGGREGATORS,
  aggregatorKind,
  aggregatorTrim,
  buildSeedAggregator,
  conflictingAggregatorMetrics,
  extremeCase,
  isValidSeedAggregator,
  trimmedPerSide,
} from '../../../../lib/seedAggregator'

const props = defineProps<{ modelValue: ObjectiveItem[]; seedsCount: number; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: ObjectiveItem[]] }>()

const hasEmpty = computed(() => props.modelValue.some(o => !o.metric_name.trim()))
const conflicts = computed(() => conflictingAggregatorMetrics(props.modelValue))

function hintOf(kind: SeedAggregatorKind): string {
  return SEED_AGGREGATORS.find(o => o.kind === kind)?.hint ?? ''
}

function extremeOf(obj: ObjectiveItem) {
  return extremeCase(aggregatorKind(obj.aggregator), obj.goal)
}

function trimSummary(trim: number): string {
  const n = props.seedsCount
  const k = trimmedPerSide(trim, n)
  if (k === 0) return `With ${n} seed${n !== 1 ? 's' : ''} this drops nothing, so it equals the mean.`
  return `With ${n} seeds, drops ${k} from each end and averages the remaining ${n - 2 * k}.`
}

function replace(i: number, patch: Partial<ObjectiveItem>) {
  emit('update:modelValue', props.modelValue.map((o, idx) => idx === i ? { ...o, ...patch } : o))
}

function updateObj(i: number, field: 'metric_name' | 'goal', value: string) {
  replace(i, { [field]: value })
}

function updateAggregator(i: number, kind: SeedAggregatorKind) {
  replace(i, { aggregator: buildSeedAggregator(kind, aggregatorTrim(props.modelValue[i]!.aggregator)) })
}

function updateTrim(i: number, raw: string) {
  replace(i, { aggregator: buildSeedAggregator('trimmed_mean', parseFloat(raw)) })
}

function remove(i: number) {
  if (props.modelValue.length <= 1) return
  emit('update:modelValue', props.modelValue.filter((_, idx) => idx !== i))
}

function add() {
  emit('update:modelValue', [
    ...props.modelValue,
    { metric_name: '', goal: 'min', aggregator: DEFAULT_SEED_AGGREGATOR },
  ])
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; }
.objectives-list { display: flex; flex-direction: column; gap: 8px; }
.obj-item { display: flex; flex-direction: column; gap: 6px; }
.obj-row { display: grid; grid-template-columns: 24px 1fr 112px 132px 32px; gap: 8px; align-items: center; }
.obj-head {
  font-size: 11px; font-weight: 600; color: var(--color-text-muted);
  text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: -4px;
}
.obj-sub { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding-left: 32px; }
.sub-label { font-size: 12px; color: var(--color-text); }
.trim-input { width: 84px; padding: 4px 8px; font-size: 12px; }
.sub-hint { font-size: 11px; color: var(--color-text-muted); line-height: 1.4; }
.case--worst { color: #b45309; }
.case--best { color: #7c3aed; }
html.dark .case--worst { color: #f9e2af; }
html.dark .case--best { color: #cba6f7; }
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
