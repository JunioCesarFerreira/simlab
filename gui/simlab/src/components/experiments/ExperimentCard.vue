<template>
  <RouterLink :to="`/experiments/${experiment.id}`" class="card experiment-card">
    <div class="row">
      <div class="name">{{ experiment.name }}</div>
      <div class="badges">
        <span v-if="experiment.is_synthetic" class="badge-synthetic">
          ⬡ {{ experiment.synthetic_bench ?? 'Synthetic' }}
        </span>
        <StatusBadge :status="experiment.status" />
      </div>
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

.badges {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.badge-synthetic {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 700;
  color: #d97706;
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: 99px;
  white-space: nowrap;
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
