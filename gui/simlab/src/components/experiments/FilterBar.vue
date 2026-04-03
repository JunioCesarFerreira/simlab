<template>
  <div class="filter-bar">
    <button
      class="tab"
      :class="{ active: modelValue === null }"
      @click="$emit('update:modelValue', null)"
    >
      Todos
      <span class="count">{{ total }}</span>
    </button>
    <button
      v-for="s in STATUSES"
      :key="s"
      class="tab"
      :class="{ active: modelValue === s }"
      @click="$emit('update:modelValue', s)"
    >
      {{ s }}
      <span v-if="counts[s]" class="count">{{ counts[s] }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed } from "vue";
import type { ExperimentStatus } from "../../types/simlab";

const STATUSES: ExperimentStatus[] = ["Running", "Waiting", "Done", "Error"];

const props = defineProps<{
  modelValue: ExperimentStatus | null;
  counts: Record<string, number>;
}>();

defineEmits<{
  (e: "update:modelValue", value: ExperimentStatus | null): void;
}>();

const total = computed(() =>
  Object.values(props.counts).reduce((a, b) => a + b, 0),
);
</script>

<style scoped>
.filter-bar {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.tab {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-muted);
  border: 1px solid transparent;
  transition: background 0.15s, color 0.15s;
}

.tab:hover {
  background: var(--color-border);
  color: var(--color-text);
}

.tab.active {
  background: var(--color-primary-light);
  color: var(--color-primary);
  border-color: #bfdbfe;
}

.count {
  font-size: 11px;
  font-weight: 700;
  background: rgba(0, 0, 0, 0.07);
  border-radius: 999px;
  padding: 1px 6px;
}

.tab.active .count {
  background: #bfdbfe;
}
</style>
