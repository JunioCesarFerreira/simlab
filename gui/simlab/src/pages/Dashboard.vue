<template>
  <div class="dashboard">
    <div class="page-header">
      <h1 class="page-title">Dashboard</h1>
      <button class="refresh-btn" :disabled="store.loading" @click="load">
        {{ store.loading ? "Carregando…" : "Atualizar" }}
      </button>
    </div>

    <div v-if="store.error" class="error-banner">
      Erro ao carregar: {{ store.error }}
    </div>

    <!-- Stat cards -->
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-value">{{ total }}</div>
        <div class="stat-label">Total</div>
      </div>
      <div class="stat-card stat-card--running">
        <div class="stat-value">{{ counts["Running"] ?? 0 }}</div>
        <div class="stat-label">Em execução</div>
      </div>
      <div class="stat-card stat-card--waiting">
        <div class="stat-value">{{ counts["Waiting"] ?? 0 }}</div>
        <div class="stat-label">Na fila</div>
      </div>
      <div class="stat-card stat-card--done">
        <div class="stat-value">{{ counts["Done"] ?? 0 }}</div>
        <div class="stat-label">Finalizados</div>
      </div>
      <div class="stat-card stat-card--error">
        <div class="stat-value">{{ counts["Error"] ?? 0 }}</div>
        <div class="stat-label">Com erro</div>
      </div>
    </div>

    <!-- Running experiments -->
    <section v-if="running.length > 0">
      <div class="section-title">Em execução</div>
      <div class="exp-list">
        <ExperimentCard
          v-for="e in running"
          :key="e.id"
          :experiment="e"
        />
      </div>
    </section>

    <!-- Waiting experiments -->
    <section v-if="waiting.length > 0">
      <div class="section-title">Na fila</div>
      <div class="exp-list">
        <ExperimentCard
          v-for="e in waiting"
          :key="e.id"
          :experiment="e"
        />
      </div>
    </section>

    <!-- Recent finished -->
    <section v-if="recentFinished.length > 0">
      <div class="section-title">Recentes finalizados</div>
      <div class="exp-list">
        <ExperimentCard
          v-for="e in recentFinished"
          :key="e.id"
          :experiment="e"
        />
      </div>
    </section>

    <div
      v-if="!store.loading && store.experiments.length === 0 && !store.error"
      class="empty-state"
    >
      Nenhum experimento encontrado.
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onBeforeUnmount } from "vue";
import { useExperimentsStore } from "../app/stores/experimentsStore";
import ExperimentCard from "../components/experiments/ExperimentCard.vue";

const store = useExperimentsStore();

const counts = computed(() => store.countByStatus);
const total = computed(() =>
  Object.values(counts.value).reduce((a, b) => a + b, 0),
);

const running = computed(() =>
  store.experiments.filter((e) => e.status === "Running"),
);
const waiting = computed(() =>
  store.experiments.filter((e) => e.status === "Waiting"),
);
const recentFinished = computed(() =>
  store.experiments
    .filter((e) => e.status === "Done")
    .slice(0, 8),
);

let pollInterval: ReturnType<typeof setInterval> | null = null;

async function load() {
  await store.fetchAll();
}

onMounted(async () => {
  await load();
  pollInterval = setInterval(() => {
    if (running.value.length > 0 || waiting.value.length > 0) {
      load();
    }
  }, 5000);
});

onBeforeUnmount(() => {
  if (pollInterval) clearInterval(pollInterval);
});
</script>

<style scoped>
.dashboard {
  display: flex;
  flex-direction: column;
  gap: 24px;
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

.stat-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
}

.stat-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 16px 20px;
  box-shadow: var(--shadow-sm);
}

.stat-value {
  font-size: 32px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 6px;
}

.stat-label {
  font-size: 12px;
  color: var(--color-text-muted);
  font-weight: 500;
}

.stat-card--running .stat-value { color: var(--status-running); }
.stat-card--waiting .stat-value { color: var(--status-waiting); }
.stat-card--done .stat-value { color: var(--status-done); }
.stat-card--error .stat-value { color: var(--status-error); }

section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.exp-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.empty-state {
  text-align: center;
  padding: 48px;
  color: var(--color-text-muted);
  font-style: italic;
}
</style>
