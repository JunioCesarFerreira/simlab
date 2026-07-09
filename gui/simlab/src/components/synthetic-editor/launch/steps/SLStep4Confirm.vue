<template>
  <div class="step">
    <p class="hint">Review all settings before launching.</p>

    <div class="section-divider">Benchmark</div>
    <div class="kv-grid">
      <span class="kv-key">Function</span>   <span class="kv-val mono">{{ draft.benchmark }}</span>
      <span class="kv-key">Objectives M</span> <span class="kv-val mono">{{ draft.M }}</span>
      <span class="kv-key">Variables n</span>  <span class="kv-val mono">{{ draft.nVars }} (x ∈ [0,1]ⁿ)</span>
      <span class="kv-key">Noise σ</span>     <span class="kv-val mono">{{ draft.noiseStd === 0 ? 'none' : draft.noiseStd }}</span>
    </div>

    <div class="section-divider">Experiment</div>
    <div class="kv-grid">
      <span class="kv-key">Name</span>       <span class="kv-val">{{ alg.name }}</span>
      <span class="kv-key">Strategy</span>   <span class="kv-val mono">{{ alg.strategy }}</span>
      <span class="kv-key">Population</span> <span class="kv-val mono">{{ alg.populationSize }}</span>
      <span class="kv-key">Generations</span> <span class="kv-val mono">{{ alg.numberOfGenerations }}</span>
      <span class="kv-key">Seed</span>       <span class="kv-val mono">{{ alg.randomSeed }}</span>
    </div>

    <div class="section-divider">Objectives</div>
    <div class="obj-tags">
      <span v-for="o in objectives" :key="o.metric_name" class="obj-tag">
        {{ o.metric_name }} <span class="obj-tag-goal">min</span>
      </span>
    </div>

    <div class="info-box">
      <span class="info-icon">ℹ</span>
      <span>
        This experiment will not consume any Cooja simulation slots.
        The master-node evaluates the {{ draft.benchmark }} function directly.
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSyntheticStore } from '../../../../app/stores/syntheticStore'
import type { ObjectiveItem } from '../../../../types/simlab'
import type { SLStep2Value } from './SLStep2Algorithm.vue'

defineProps<{
  alg: SLStep2Value
  objectives: ObjectiveItem[]
}>()

const store = useSyntheticStore()
const draft = computed(() => store.draft)
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--color-text-muted);
  border-bottom: 1px solid var(--color-border); padding-bottom: 4px;
}
.kv-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 16px;
}
.kv-key { font-size: 12px; color: var(--color-text-muted); }
.kv-val { font-size: 12px; font-weight: 600; color: var(--color-text); }
.mono { font-family: monospace; }

.obj-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.obj-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: 99px;
  font-size: 12px;
  font-weight: 600;
  color: #d97706;
  font-family: monospace;
}
.obj-tag-goal {
  font-size: 10px;
  color: var(--color-text-muted);
  font-weight: 400;
  font-family: sans-serif;
}

.info-box {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  background: var(--color-primary-light);
  border: 1px solid rgba(59, 130, 246, 0.2);
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--color-text);
  line-height: 1.5;
}
.info-icon { color: var(--color-primary); flex-shrink: 0; font-size: 14px; }
</style>
