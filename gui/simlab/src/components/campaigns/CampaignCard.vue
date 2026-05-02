<template>
  <RouterLink :to="`/campaigns/${campaign.id}`" class="card campaign-card">
    <div class="row">
      <div class="name">{{ campaign.name }}</div>
      <span class="exp-badge">{{ campaign.experiment_count }} experiment{{ campaign.experiment_count !== 1 ? 's' : '' }}</span>
    </div>
    <div v-if="campaign.description" class="description">
      {{ campaign.description }}
    </div>
    <div class="meta">
      <span v-if="campaign.created_time">
        Created {{ formatDate(campaign.created_time) }}
      </span>
      <span v-else class="muted">No date</span>
    </div>
  </RouterLink>
</template>

<script setup lang="ts">
import type { CampaignInfoDto } from "../../types/simlab";

defineProps<{ campaign: CampaignInfoDto }>();

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
.campaign-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  transition: box-shadow 0.15s, border-color 0.15s;
}

.campaign-card:hover {
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

.exp-badge {
  font-size: 11px;
  font-weight: 700;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 2px 10px;
  white-space: nowrap;
  flex-shrink: 0;
}

.description {
  font-size: 12px;
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.meta {
  font-size: 12px;
  color: var(--color-text-muted);
}

.muted {
  font-style: italic;
}
</style>
