<template>
  <RouterLink :to="`/experiments/${experiment.id}`" class="card experiment-card">
    <div class="row">
      <div class="name">{{ experiment.name }}</div>
      <StatusBadge :status="experiment.status" />
    </div>
    <div v-if="experiment.system_message" class="message">
      {{ experiment.system_message }}
    </div>
    <div class="meta">
      <span v-if="experiment.start_time">
        Start: {{ formatDate(experiment.start_time) }}
      </span>
      <span v-if="experiment.end_time" class="sep">·</span>
      <span v-if="experiment.end_time">
        End: {{ formatDate(experiment.end_time) }}
      </span>
      <span v-if="!experiment.start_time && !experiment.end_time" class="muted">
        Waiting to start
      </span>
    </div>
  </RouterLink>
</template>

<script setup lang="ts">
import StatusBadge from "../common/StatusBadge.vue";
import type { ExperimentInfoDto } from "../../types/simlab";

defineProps<{ experiment: ExperimentInfoDto }>();

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
</script>

<style scoped>
.experiment-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: box-shadow 0.15s, border-color 0.15s;
}

.experiment-card:hover {
  border-color: #bfdbfe;
  box-shadow: var(--shadow-md);
}

.row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.name {
  font-weight: 600;
  font-size: 15px;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.message {
  font-size: 12px;
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.sep {
  opacity: 0.4;
}

.muted {
  color: var(--color-text-muted);
  font-style: italic;
}
</style>
