<template>
  <div class="step">
    <div class="banner">
      <span class="banner-icon">⬡</span>
      <div>
        <p class="banner-title">Synthetic benchmark experiment</p>
        <p class="banner-sub">No Cooja containers required — objectives are evaluated analytically.</p>
      </div>
    </div>

    <div class="section-divider">Benchmark</div>

    <div class="kv-grid">
      <span class="kv-key">Function</span>
      <span class="kv-val mono">{{ draft.benchmark }}</span>
      <span class="kv-key">Objectives M</span>
      <span class="kv-val mono">{{ draft.M }}</span>
      <span class="kv-key">Variables n</span>
      <span class="kv-val mono">{{ draft.nVars }}</span>
      <span class="kv-key">Genome</span>
      <span class="kv-val mono">x ∈ [0,1]ⁿ ({{ draft.nVars }} floats)</span>
      <span class="kv-key">Noise σ</span>
      <span class="kv-val mono">{{ draft.noiseStd === 0 ? 'none' : draft.noiseStd }}</span>
    </div>

    <p class="note">
      The benchmark is evaluated directly on the decision vector x ∈ [0,1]ⁿ
      (no region round-trip). Proceed to configure the optimization algorithm parameters.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSyntheticStore } from '../../../../app/stores/syntheticStore'

const store = useSyntheticStore()
const draft = computed(() => store.draft)
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }

.banner {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 14px;
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: var(--radius-sm);
}
.banner-icon { font-size: 22px; color: #f59e0b; flex-shrink: 0; margin-top: 1px; }
.banner-title { font-size: 13px; font-weight: 700; color: var(--color-text); }
.banner-sub { font-size: 11px; color: var(--color-text-muted); margin-top: 2px; }

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
.kv-key { font-size: 12px; color: var(--color-text-muted); align-self: center; }
.kv-val { font-size: 12px; font-weight: 600; color: var(--color-text); }
.mono { font-family: monospace; }
.note {
  font-size: 11px;
  color: var(--color-text-muted);
  line-height: 1.5;
  padding: 10px;
  background: var(--color-bg);
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
}
</style>
