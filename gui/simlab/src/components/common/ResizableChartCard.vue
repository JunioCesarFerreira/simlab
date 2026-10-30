<template>
  <div class="card chart-card" :style="{ height: height + 'px' }">
    <slot />
    <ResizeHandle
      v-model="height"
      class="chart-card__handle"
      :label="label"
      :min="min"
      :max="max"
    />
  </div>
</template>

<script setup lang="ts">
import ResizeHandle from "./ResizeHandle.vue";

withDefaults(
  defineProps<{
    /** Accessible name for the resize separator. */
    label?: string;
    min?: number;
    max?: number;
  }>(),
  { label: "chart panel", min: 180, max: 1400 },
);

const height = defineModel<number>({ required: true });
</script>

<style scoped>
.chart-card {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  /* height controlled by :style binding — min/max enforced by useResizable */
}

.chart-card__handle {
  /* Bleed to the card edges and absorb its bottom padding. Pages whose .card
     padding differs from the 16px global default override the vars. */
  margin: 4px calc(var(--card-pad-x, 16px) * -1) calc(var(--card-pad-b, 16px) * -1);
}
</style>
