<template>
  <div class="gen-row">
    <button class="gen-header" @click="open = !open">
      <span class="gen-index">Gen {{ generation.index }}</span>
      <StatusBadge :status="generation.status" />
      <span class="gen-times">
        <span v-if="generation.start_time">{{ formatTime(generation.start_time) }}</span>
        <span v-if="generation.start_time && generation.end_time"> → </span>
        <span v-if="generation.end_time">{{ formatTime(generation.end_time) }}</span>
        <span v-if="duration" class="duration">({{ duration }})</span>
      </span>
      <span class="pop-count">
        {{ generation.population.length }} individuals
        <span v-if="penalizedCount > 0" class="penalized-count" :title="`${penalizedCount} infeasible (penalized)`">
          · {{ penalizedCount }} infeasible
        </span>
      </span>
      <span class="chevron" :class="{ rotated: open }">▾</span>
    </button>

    <div v-if="open" class="gen-body">
      <div v-if="generation.population.length === 0" class="empty">
        No individuals registered yet.
      </div>
      <table v-else class="ind-table">
        <thead>
          <tr>
            <th>Hash</th>
            <th>Objectives</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <IndividualRow
            v-for="ind in generation.population"
            :key="ind.id"
            :individual="ind"
            :objective-names="objectiveNames"
            @select="$emit('select', $event)"
          />
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import StatusBadge from "../common/StatusBadge.vue";
import IndividualRow from "./IndividualRow.vue";
import type { GenerationDto, IndividualDto } from "../../types/simlab";
import { isPenalized } from "../../types/simlab";

const props = defineProps<{
  generation: GenerationDto;
  objectiveNames: string[];
  defaultOpen?: boolean;
}>();

defineEmits<{ (e: "select", individual: IndividualDto): void }>();

const open = ref(props.defaultOpen ?? false);

const penalizedCount = computed(
  () => props.generation.population.filter((ind) => isPenalized(ind.objectives)).length,
);

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

const duration = computed(() => {
  if (!props.generation.start_time || !props.generation.end_time) return null;
  const ms =
    new Date(props.generation.end_time).getTime() -
    new Date(props.generation.start_time).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
});
</script>

<style scoped>
.gen-row {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.gen-header {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: var(--color-surface);
  text-align: left;
  cursor: pointer;
  transition: background 0.1s;
}

.gen-header:hover {
  background: var(--color-bg);
}

.gen-index {
  font-weight: 700;
  font-size: 13px;
  min-width: 54px;
}

.gen-times {
  font-size: 12px;
  color: var(--color-text-muted);
  flex: 1;
}

.duration {
  margin-left: 4px;
  color: var(--color-text-muted);
}

.pop-count {
  font-size: 12px;
  color: var(--color-text-muted);
  white-space: nowrap;
}

.penalized-count {
  color: #b45309;
  font-weight: 600;
}

.chevron {
  font-size: 16px;
  color: var(--color-text-muted);
  transition: transform 0.2s;
  flex-shrink: 0;
}

.chevron.rotated {
  transform: rotate(180deg);
}

.gen-body {
  border-top: 1px solid var(--color-border);
  background: var(--color-bg);
  padding: 0;
}

.empty {
  padding: 16px;
  font-size: 13px;
  color: var(--color-text-muted);
  font-style: italic;
}

.ind-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.ind-table thead th {
  padding: 8px 10px;
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
}
</style>
