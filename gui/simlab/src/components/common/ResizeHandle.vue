<template>
  <div
    class="resize-handle"
    role="separator"
    aria-orientation="horizontal"
    :aria-label="`Resize ${label}`"
    :aria-valuenow="height"
    :aria-valuemin="min"
    :aria-valuemax="max"
    tabindex="0"
    title="Drag to resize (arrow keys also work)"
    @pointerdown="startResize"
    @keydown="onKeydown"
  />
</template>

<script setup lang="ts">
import { useResizable } from "../../composables/useResizable";

/**
 * The drag bar under a resizable chart, on its own so a chart that does not
 * live in its own card can have one too. Drives a height through `v-model`;
 * the arrow keys move it as well.
 */
const props = withDefaults(
  defineProps<{
    /** Accessible name for the resize separator. */
    label?: string;
    min?: number;
    max?: number;
  }>(),
  { label: "chart panel", min: 180, max: 1400 },
);

const height = defineModel<number>({ required: true });
const { startResize, nudge } = useResizable({
  height,
  min: props.min,
  max: props.max,
});

const KEY_STEP = 24;

function onKeydown(e: KeyboardEvent) {
  if (e.key === "ArrowUp") {
    nudge(-KEY_STEP);
    e.preventDefault();
  } else if (e.key === "ArrowDown") {
    nudge(KEY_STEP);
    e.preventDefault();
  }
}
</script>

<style scoped>
.resize-handle {
  flex-shrink: 0;
  height: 10px;
  cursor: ns-resize;
  touch-action: none; /* pointer-capture drag: don't let touch scroll steal it */
  display: flex;
  align-items: center;
  justify-content: center;
  border-top: 1px solid var(--color-border);
  background: transparent;
  transition: background 0.15s;
}

.resize-handle::after {
  content: "";
  width: 36px;
  height: 3px;
  border-radius: 99px;
  background: var(--color-border);
  transition:
    background 0.15s,
    transform 0.15s;
}

.resize-handle:hover,
.resize-handle:focus-visible {
  background: var(--color-surface-hover);
}

.resize-handle:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: -2px;
}

.resize-handle:hover::after,
.resize-handle:focus-visible::after {
  background: var(--color-text-muted);
  transform: scaleX(1.25);
}
</style>
