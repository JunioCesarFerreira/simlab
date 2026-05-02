<template>
  <div class="detail-page">
    <div v-if="loading && !campaign" class="loading">
      Loading campaign…
    </div>
    <div v-else-if="error" class="error-banner">
      Error: {{ error }}
    </div>

    <template v-else-if="campaign">
      <!-- Header -->
      <div class="header">
        <RouterLink to="/campaigns" class="back-link">← Campaigns</RouterLink>
        <div class="header-main">
          <div class="header-left">
            <h1 class="campaign-name">{{ campaign.name }}</h1>
            <div v-if="campaign.description" class="description">
              {{ campaign.description }}
            </div>
            <div class="header-meta">
              <span v-if="campaign.created_time" class="meta-date">
                Created {{ formatDate(campaign.created_time) }}
              </span>
              <span class="exp-count-pill">
                {{ campaign.experiments.length }} experiment{{ campaign.experiments.length !== 1 ? 's' : '' }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Stats row -->
      <div class="stat-grid" v-if="campaign.experiments.length > 0">
        <div class="stat-card stat-card--running">
          <div class="stat-value">{{ countByStatus['Running'] ?? 0 }}</div>
          <div class="stat-label">Running</div>
        </div>
        <div class="stat-card stat-card--waiting">
          <div class="stat-value">{{ countByStatus['Waiting'] ?? 0 }}</div>
          <div class="stat-label">Queued</div>
        </div>
        <div class="stat-card stat-card--done">
          <div class="stat-value">{{ countByStatus['Done'] ?? 0 }}</div>
          <div class="stat-label">Finished</div>
        </div>
        <div class="stat-card stat-card--error">
          <div class="stat-value">{{ countByStatus['Error'] ?? 0 }}</div>
          <div class="stat-label">Errored</div>
        </div>
      </div>

      <!-- Experiments list -->
      <div class="section-title">Experiments</div>

      <div v-if="campaign.experiments.length === 0" class="empty-state">
        No experiments in this campaign yet.
      </div>

      <div v-else class="exp-list">
        <ExperimentCard
          v-for="e in campaign.experiments"
          :key="e.id"
          :experiment="e"
        />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { getCampaignFull } from "../api/campaigns";
import type { CampaignFullDto } from "../types/simlab";
import ExperimentCard from "../components/experiments/ExperimentCard.vue";

const props = defineProps<{ id: string }>();

const campaign = ref<CampaignFullDto | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const countByStatus = computed(() => {
  const counts: Record<string, number> = {};
  for (const e of campaign.value?.experiments ?? []) {
    counts[e.status] = (counts[e.status] ?? 0) + 1;
  }
  return counts;
});

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

onMounted(async () => {
  loading.value = true;
  error.value = null;
  try {
    campaign.value = await getCampaignFull(props.id);
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
});
</script>

<style scoped>
.detail-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.back-link {
  font-size: 13px;
  color: var(--color-text-muted);
  text-decoration: none;
  margin-bottom: 4px;
  display: inline-block;
}

.back-link:hover {
  color: var(--color-primary);
}

.header {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.header-main {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.campaign-name {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.2;
}

.description {
  font-size: 14px;
  color: var(--color-text-muted);
}

.header-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

.meta-date {
  font-size: 12px;
  color: var(--color-text-muted);
}

.exp-count-pill {
  font-size: 11px;
  font-weight: 700;
  background: var(--color-primary-light);
  color: var(--color-primary);
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  padding: 2px 10px;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 10px;
}

.stat-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 14px 18px;
  box-shadow: var(--shadow-sm);
}

.stat-value {
  font-size: 28px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 11px;
  color: var(--color-text-muted);
  font-weight: 500;
}

.stat-card--running .stat-value { color: var(--status-running); }
.stat-card--waiting .stat-value { color: var(--status-waiting); }
.stat-card--done .stat-value { color: var(--status-done); }
.stat-card--error .stat-value { color: var(--status-error); }

.section-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.exp-list {
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

.error-banner {
  padding: 12px 16px;
  background: #fee2e2;
  color: var(--status-error);
  border-radius: var(--radius-md);
  font-size: 13px;
  border: 1px solid #fecaca;
}
</style>
