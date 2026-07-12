<template>
  <button
    type="button"
    class="export-btn"
    title="Export chart as PNG image"
    aria-label="Export chart as PNG image"
    @click="$emit('click')"
  >
    ⬇ PNG
  </button>
</template>

<script setup lang="ts">
defineEmits<{ (e: "click"): void }>();
</script>

<style scoped>
/* Matches the .pin-btn / .reset-zoom-btn convention used across the chart
   components (ParetoFrontChart, ParetoFront3DChart, …) so the export action
   reads as part of the same toolbar family regardless of which chart hosts it. */
.export-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text-muted);
  transition: background 0.15s, border-color 0.15s, color 0.15s;
  cursor: pointer;
}

.export-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
}
</style>
