<template>
  <div class="occupancy" :title="summary">
    <div class="bar" role="img" :aria-label="summary">
      <span
        v-for="segment in segments"
        :key="segment.kind"
        :class="['segment', `segment--${segment.kind}`]"
        :style="{ width: `${segment.percent}%` }"
      />
      <span v-if="total === 0" class="segment segment--empty" style="width: 100%" />
    </div>
    <div class="legend">
      <span v-for="segment in segments" :key="segment.kind" class="legend-item">
        <i :class="['dot', `segment--${segment.kind}`]" />
        {{ segment.count }} {{ segment.label }}
      </span>
      <span v-if="total === 0" class="legend-item muted">no containers</span>
      <!-- The segments are a partition of the inventory. If they ever stop
           being one, the bar is describing something other than this host's
           containers, and saying so beats drawing it as if nothing were
           wrong. -->
      <span v-if="!consistent" class="legend-item mismatch">
        counts do not add up to {{ resourceTotal }} containers
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import {
  occupancySegments,
  occupancySummary,
  occupancyTotal,
  occupancyIsConsistent,
} from "../../services/occupancy";

/**
 * How a host's containers are being used, as one bar.
 *
 * The counts come straight from the scheduler's own snapshot, so what is
 * drawn here is what the placement decision saw — not a second tally that
 * can disagree with it. The rules for reading them are in
 * `services/occupancy.ts`, where they can be tested against the partition
 * the backend actually produces.
 */
const props = defineProps<{
  free: number;
  active: number;
  unavailable: number;
  disabled: number;
  /** The inventory. The segments partition it, so it is the denominator. */
  resourceTotal?: number;
  /** In use *and* administratively out — an attribute of `active`. */
  draining?: number;
}>();

const counts = computed(() => ({
  free: props.free,
  active: props.active,
  unavailable: props.unavailable,
  disabled: props.disabled,
  resourceTotal: props.resourceTotal,
  draining: props.draining,
}));

const total = computed(() => occupancyTotal(counts.value));
const segments = computed(() => occupancySegments(counts.value));
const summary = computed(() => occupancySummary(counts.value));
const consistent = computed(() => occupancyIsConsistent(counts.value));
</script>

<style scoped>
.occupancy {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bar {
  display: flex;
  height: 10px;
  border-radius: 999px;
  overflow: hidden;
  background: var(--color-surface-hover);
  border: 1px solid var(--color-border);
}

.segment {
  height: 100%;
  transition: width 0.25s ease;
}

.segment--active {
  background: var(--status-running);
}
.segment--free {
  background: var(--status-done);
}
.segment--unavailable {
  background: var(--status-error);
}
.segment--disabled {
  background: var(--status-cancelled);
}
.segment--empty {
  background: transparent;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 11px;
  color: var(--color-text-muted);
}

.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.legend-item.muted {
  font-style: italic;
}

.legend-item.mismatch {
  color: var(--status-error);
  font-weight: 600;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  display: inline-block;
}
</style>
