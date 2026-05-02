<template>
  <div class="list-page">
    <div class="page-header">
      <h1 class="page-title">Campaigns</h1>
      <button class="refresh-btn" :disabled="store.loading" @click="load">
        {{ store.loading ? "Loading…" : "Refresh" }}
      </button>
    </div>

    <div v-if="store.error" class="error-banner">
      Failed to load: {{ store.error }}
    </div>

    <div v-if="store.loading && store.campaigns.length === 0" class="loading">
      Loading campaigns…
    </div>

    <div v-else-if="store.campaigns.length === 0 && !store.loading" class="empty-state">
      No campaigns found.
    </div>

    <div v-else class="campaign-list">
      <CampaignCard
        v-for="c in store.campaigns"
        :key="c.id"
        :campaign="c"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from "vue";
import { useCampaignsStore } from "../app/stores/campaignsStore";
import CampaignCard from "../components/campaigns/CampaignCard.vue";

const store = useCampaignsStore();

async function load() {
  await store.fetchAll();
}

onMounted(load);
</script>

<style scoped>
.list-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.page-title {
  font-size: 22px;
  font-weight: 700;
}

.refresh-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  background: var(--color-primary-light);
  color: var(--color-primary);
  font-size: 13px;
  font-weight: 600;
  border: 1px solid #bfdbfe;
  transition: background 0.15s;
}

.refresh-btn:hover:not(:disabled) {
  background: #dbeafe;
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.error-banner {
  padding: 12px 16px;
  background: #fee2e2;
  color: var(--status-error);
  border-radius: var(--radius-md);
  font-size: 13px;
  border: 1px solid #fecaca;
}

.campaign-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.loading,
.empty-state {
  text-align: center;
  padding: 48px;
  color: var(--color-text-muted);
  font-style: italic;
}
</style>
