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
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";

/**
 * How a host's containers are being used, as one bar.
 *
 * The counts come straight from the scheduler's own snapshot, so what is
 * drawn here is what the placement decision saw — not a second tally that
 * can disagree with it.
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

/**
 * The denominator.
 *
 * `resource_total` when the API sends it, because the four segments are a
 * partition of exactly that. Summing them was the fallback and the bug: a
 * container that was busy and disabled for draining counted in two
 * categories, so one worker produced a total of two.
 */
const total = computed(() =>
  props.resourceTotal !== undefined
    ? props.resourceTotal
    : props.active + props.free + props.unavailable + props.disabled,
);

type Segment = { kind: string; label: string; count: number; percent: number };

const segments = computed<Segment[]>(() => {
  const all = total.value;
  // Order is meaningful: in use first, then idle capacity, then what cannot
  // be used — so the eye reads occupancy left to right.
  const rows: Array<Omit<Segment, "percent">> = [
    { kind: "active", label: "in use", count: props.active },
    { kind: "free", label: "free", count: props.free },
    { kind: "unavailable", label: "unavailable", count: props.unavailable },
    { kind: "disabled", label: "disabled", count: props.disabled },
  ];
  return rows
    .filter((row) => row.count > 0)
    .map((row) => ({ ...row, percent: all === 0 ? 0 : (row.count / all) * 100 }));
});

const summary = computed(() => {
  if (total.value === 0) return "No containers on this host";
  const parts = segments.value.map((s) => `${s.count} ${s.label}`);
  // Draining is not a category of its own: it says something extra about the
  // workers already counted as in use.
  if (props.draining) parts.push(`${props.draining} of them draining`);
  return parts.join(", ");
});
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

.dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  display: inline-block;
}
</style>
