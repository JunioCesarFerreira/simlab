<template>
  <div class="step-problem">
    <div v-if="!hasSink" class="warn-banner">
      ⚠ The problem has no sink placed. Position the sink in the editor before continuing.
    </div>

    <div class="summary-grid">
      <div class="summary-item">
        <span class="label">Type</span>
        <span class="value badge">{{ draft.name }}</span>
      </div>
      <div class="summary-item">
        <span class="label">Region</span>
        <span class="value mono">
          [{{ draft.region[0] }}, {{ draft.region[1] }}] →
          [{{ draft.region[2] }}, {{ draft.region[3] }}]
        </span>
      </div>
      <div class="summary-item">
        <span class="label">Reach radius</span>
        <span class="value mono">{{ draft.radiusOfReach }} u</span>
      </div>
      <div class="summary-item">
        <span class="label">Interference radius</span>
        <span class="value mono">{{ draft.radiusOfInter }} u</span>
      </div>
      <div class="summary-item" v-if="draft.sink">
        <span class="label">Sink</span>
        <span class="value mono">({{ draft.sink.x }}, {{ draft.sink.y }})</span>
      </div>
      <div class="summary-item" v-if="draft.candidates.length > 0">
        <span class="label">Candidates</span>
        <span class="value mono">{{ draft.candidates.length }}</span>
      </div>
      <div class="summary-item" v-if="draft.targets.length > 0">
        <span class="label">Targets</span>
        <span class="value mono">{{ draft.targets.length }}</span>
      </div>
      <div class="summary-item" v-if="draft.mobileNodes.length > 0">
        <span class="label">Mobile nodes</span>
        <span class="value mono">{{ draft.mobileNodes.length }}</span>
      </div>
      <div class="summary-item" v-if="draft.name === 'problem1'">
        <span class="label">Num. sensors</span>
        <span class="value mono">{{ draft.numSensors }}</span>
      </div>
      <div class="summary-item">
        <span class="label">MAC protocol</span>
        <span class="value mono">{{ macProtocol }}</span>
      </div>
    </div>

    <p class="note">
      To change the problem, close this wizard and use the problem editor.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../../app/stores/problemStore'

const problemStore = useProblemStore()
const draft = computed(() => problemStore.draft)
const hasSink = computed(() => !!draft.value.sink)
const macProtocol = computed(() => draft.value.chromosome?.macProtocol ?? '—')
</script>

<style scoped>
.step-problem { display: flex; flex-direction: column; gap: 16px; }
.warn-banner {
  padding: 10px 14px; background: #fff7ed; border: 1px solid #fdba74;
  border-radius: var(--radius-md); font-size: 13px; color: #c2410c;
}
.summary-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}
.summary-item {
  display: flex; flex-direction: column; gap: 2px;
  padding: 10px 12px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-md);
}
.label { font-size: 11px; color: var(--color-text-muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }
.value { font-size: 13px; color: var(--color-text); font-weight: 600; }
.value.mono { font-family: 'SFMono-Regular', Consolas, monospace; }
.value.badge {
  display: inline-block; padding: 1px 8px;
  background: var(--color-primary-light); color: var(--color-primary);
  border-radius: 99px; font-size: 12px;
}
.note { font-size: 12px; color: var(--color-text-muted); }
</style>
