<template>
  <div class="detail-page">
    <!-- Loading / error states -->
    <div v-if="store.loading && !store.experiment" class="loading">
      Loading experiment…
    </div>
    <div v-else-if="store.error" class="error-banner">
      Error: {{ store.error }}
    </div>

    <!-- Individual detail panel -->
    <IndividualDetailPanel
      v-if="selectedIndividual"
      :individual="selectedIndividual"
      :objective-names="store.objectiveNames"
      :metric-columns="metricColumns"
      @close="selectedIndividual = null"
    />

    <template v-else-if="store.experiment">
      <!-- Header -->
      <div class="header">
        <RouterLink to="/experiments" class="back-link">← Experiments</RouterLink>
        <div class="header-main">
          <div class="header-left">
            <h1 class="exp-name">{{ store.experiment.name }}</h1>
            <div class="header-meta">
              <StatusBadge :status="store.experiment.status" />
              <span v-if="store.isRunning" class="live-pill">● LIVE</span>
              <span class="meta-date" v-if="store.experiment.created_time">
                Created {{ formatDate(store.experiment.created_time) }}
              </span>
            </div>
            <div v-if="store.experiment.system_message" class="sys-message">
              {{ store.experiment.system_message }}
            </div>
          </div>
          <div class="header-actions">
            <button
              v-if="hasAnalysisFiles"
              class="action-btn"
              :disabled="downloading.analysis"
              @click="doDownloadAnalysis"
            >
              {{ downloading.analysis ? "Downloading…" : "Download analyses" }}
            </button>
            <button
              class="action-btn"
              :disabled="downloading.topologies"
              @click="doDownloadTopologies"
            >
              {{ downloading.topologies ? "Downloading…" : "Download topologies" }}
            </button>
          </div>
        </div>
      </div>

      <!-- Timeline row -->
      <div class="timeline-bar" v-if="store.experiment.start_time">
        <span>Start: {{ formatDate(store.experiment.start_time) }}</span>
        <template v-if="store.experiment.end_time">
          <span class="sep">→</span>
          <span>End: {{ formatDate(store.experiment.end_time) }}</span>
          <span class="duration">({{ totalDuration }})</span>
        </template>
      </div>

      <!-- Overview grid: parameters + charts -->
      <div class="overview-grid">
        <!-- Parameters panel -->
        <div class="card params-panel">
          <div class="section-title">Parameters</div>

          <div class="param-group">
            <div class="param-label">Strategy</div>
            <div class="param-value">{{ store.experiment.parameters.strategy }}</div>
          </div>

          <details class="collapsible" open>
            <summary>Algorithm</summary>
            <div class="param-table">
              <div
                v-for="(val, key) in store.experiment.parameters.algorithm"
                :key="key"
                class="param-row"
              >
                <span class="pk">{{ key }}</span>
                <span class="pv">{{ val }}</span>
              </div>
            </div>
          </details>

          <details class="collapsible">
            <summary>Objectives</summary>
            <div class="param-table">
              <div
                v-for="(obj, i) in store.experiment.parameters.objectives"
                :key="i"
                class="param-row"
              >
                <span class="pk">{{ obj.metric_name }}</span>
                <span class="pv goal" :class="`goal--${obj.goal}`">{{ obj.goal }}</span>
              </div>
            </div>
          </details>

          <details class="collapsible">
            <summary class="summary-with-btn">
              Problem
              <button
                class="viz-btn"
                @click.prevent="showProblemViz = true"
                title="Visualize problem"
              >
                ⬡ Visualize
              </button>
            </summary>
            <div class="param-table">
              <div
                v-for="(val, key) in store.experiment.parameters.problem"
                :key="key"
                class="param-row"
              >
                <span class="pk">{{ key }}</span>
                <span class="pv mono">{{ formatParamVal(val) }}</span>
              </div>
            </div>
          </details>

          <details class="collapsible">
            <summary>Data conversion</summary>
            <div class="param-table">
              <div class="param-row">
                <span class="pk">node_col</span>
                <span class="pv">{{ store.experiment.data_conversion_config.node_col }}</span>
              </div>
              <div class="param-row">
                <span class="pk">time_col</span>
                <span class="pv">{{ store.experiment.data_conversion_config.time_col }}</span>
              </div>
              <div
                v-for="(m, i) in store.experiment.data_conversion_config.metrics"
                :key="i"
                class="param-row"
              >
                <span class="pk">{{ m.name }}</span>
                <span class="pv">{{ m.kind }} / {{ m.column }}</span>
              </div>
            </div>
          </details>
        </div>

        <!-- Charts panel -->
        <div class="charts-panel">
          <div class="card chart-card">
            <div class="section-title">Pareto Front</div>
            <ParetoFrontChart
              :pareto-front="store.experiment.pareto_front"
              :generations="store.experiment.generations"
              :objective-names="store.objectiveNames"
              @click-individual="openIndividual"
            />
          </div>
          <div class="card chart-card">
            <div class="section-title">Objectives evolution (best per generation)</div>
            <ObjectivesEvolutionChart
              :generations="store.experiment.generations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
            />
          </div>
        </div>
      </div>

      <!-- Progress bar for running experiments -->
      <div v-if="store.isRunning" class="progress-section card">
        <div class="progress-header">
          <span>Completed generations: {{ finishedCount }} / {{ totalGenerations }}</span>
          <span class="live-pill">● Updating every 3s</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" :style="{ width: `${progressPct}%` }" />
        </div>
      </div>

      <!-- Generations -->
      <div class="generations-section">
        <div class="section-title">
          Generations
          <span class="gen-count">({{ store.experiment.generations.length }})</span>
        </div>
        <div
          v-if="store.experiment.generations.length === 0"
          class="empty-state"
        >
          No generations started yet.
        </div>
        <div v-else class="gen-list">
          <GenerationRow
            v-for="gen in sortedGenerations"
            :key="gen.id"
            :generation="gen"
            :objective-names="store.objectiveNames"
            :default-open="gen.index === 0"
          />
        </div>
      </div>

      <!-- Problem visualization modal (Teleport renders to body) -->
      <ProblemVizModal
        v-if="showProblemViz"
        :problem="store.experiment.parameters.problem as JsonObject"
        @close="showProblemViz = false"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount } from "vue";
import { useExperimentDetailStore } from "../app/stores/experimentDetailStore";
import StatusBadge from "../components/common/StatusBadge.vue";
import GenerationRow from "../components/detail/GenerationRow.vue";
import ParetoFrontChart from "../components/charts/ParetoFrontChart.vue";
import ObjectivesEvolutionChart from "../components/charts/ObjectivesEvolutionChart.vue";
import IndividualDetailPanel from "../components/detail/IndividualDetailPanel.vue";
import ProblemVizModal from "../components/detail/ProblemVizModal.vue";
import { downloadAnalysisZip, downloadTopologiesZip } from "../api/files";
import type { IndividualDto, JsonObject } from "../types/simlab";

const props = defineProps<{ id: string }>();
const store = useExperimentDetailStore();

const sortedGenerations = computed(() =>
  [...(store.experiment?.generations ?? [])].sort((a, b) => a.index - b.index),
);

const finishedCount = computed(
  () =>
    sortedGenerations.value.filter((g) => g.status === "Done").length,
);

const totalGenerations = computed(
  () =>
    (store.experiment?.parameters?.algorithm?.["number_of_generations"] as number | undefined) ??
    store.experiment?.generations.length ??
    0,
);

const progressPct = computed(() => {
  if (!totalGenerations.value) return 0;
  return Math.round((finishedCount.value / totalGenerations.value) * 100);
});

const hasAnalysisFiles = computed(
  () => Object.keys(store.experiment?.analysis_files ?? {}).length > 0,
);

// Metric columns used in objective computation
const metricColumns = computed(() =>
  store.experiment?.data_conversion_config?.metrics?.map((m) => m.column) ?? [],
);

// Individual selected by chart click
const selectedIndividual = ref<IndividualDto | null>(null);

// Problem visualization modal
const showProblemViz = ref(false);

function openIndividual(individualId: string) {
  const all = store.experiment?.generations.flatMap((g) => g.population) ?? [];
  const found = all.find((ind) => ind.individual_id === individualId);
  if (found) selectedIndividual.value = found;
}

const downloading = reactive({ analysis: false, topologies: false });

async function doDownloadAnalysis() {
  downloading.analysis = true;
  try { await downloadAnalysisZip(props.id); }
  catch (e) { console.error("Error downloading analyses:", e); }
  finally { downloading.analysis = false; }
}

async function doDownloadTopologies() {
  downloading.topologies = true;
  try { await downloadTopologiesZip(props.id); }
  catch (e) { console.error("Error downloading topologies:", e); }
  finally { downloading.topologies = false; }
}

const totalDuration = computed(() => {
  const exp = store.experiment;
  if (!exp?.start_time || !exp?.end_time) return null;
  const ms =
    new Date(exp.end_time).getTime() - new Date(exp.start_time).getTime();
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
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

function formatParamVal(val: unknown): string {
  if (typeof val === "object" && val !== null) return JSON.stringify(val);
  return String(val);
}

onMounted(async () => {
  await store.fetch(props.id);
  store.startPolling(props.id);
});

onBeforeUnmount(() => {
  store.stopPolling();
});
</script>

<style scoped>
.detail-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* Header */
.back-link {
  font-size: 13px;
  color: var(--color-primary);
  font-weight: 500;
  display: inline-block;
  margin-bottom: 8px;
}

.back-link:hover {
  text-decoration: underline;
}

.header {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.header-main {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.exp-name {
  font-size: 24px;
  font-weight: 800;
  line-height: 1.2;
  margin-bottom: 6px;
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

.live-pill {
  font-size: 11px;
  font-weight: 700;
  color: var(--status-running);
  background: #dbeafe;
  padding: 2px 8px;
  border-radius: 999px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.sys-message {
  margin-top: 8px;
  font-size: 13px;
  color: var(--color-text-muted);
  font-style: italic;
}

.header-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
  flex-wrap: wrap;
}

.action-btn {
  padding: 7px 14px;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text);
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;
}

.action-btn:hover {
  background: var(--color-bg);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

/* Timeline */
.timeline-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--color-text-muted);
  padding: 8px 14px;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}

.sep { opacity: 0.4; }
.duration { font-weight: 600; color: var(--color-text); }

/* Overview grid */
.overview-grid {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 16px;
  align-items: start;
}

@media (max-width: 900px) {
  .overview-grid {
    grid-template-columns: 1fr;
  }
}

/* Params panel */
.params-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.param-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.param-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
}

.param-value {
  font-size: 14px;
  font-weight: 600;
}

.collapsible {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.collapsible summary {
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  background: var(--color-bg);
  user-select: none;
}

.summary-with-btn {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.viz-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 9px;
  border: 1px solid #bfdbfe;
  border-radius: var(--radius-sm);
  background: var(--color-primary-light);
  color: var(--color-primary);
  transition: background 0.15s;
  flex-shrink: 0;
}

.viz-btn:hover {
  background: #dbeafe;
}

.collapsible summary:hover {
  background: var(--color-border);
}

.param-table {
  padding: 4px 0;
}

.param-row {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  padding: 4px 12px;
  font-size: 12px;
}

.param-row:nth-child(even) {
  background: var(--color-bg);
}

.pk {
  color: var(--color-text-muted);
  white-space: nowrap;
  flex-shrink: 0;
}

.pv {
  font-weight: 500;
  text-align: right;
  word-break: break-all;
}

.goal {
  font-weight: 700;
  font-size: 11px;
  text-transform: uppercase;
}

.goal--min { color: var(--status-running); }
.goal--max { color: var(--status-done); }

/* Charts */
.charts-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chart-card {
  min-height: 320px;
  display: flex;
  flex-direction: column;
}

/* Progress */
.progress-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  font-weight: 500;
}

.progress-track {
  height: 8px;
  background: var(--color-border);
  border-radius: 999px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--color-primary);
  border-radius: 999px;
  transition: width 0.5s ease;
}

/* Generations */
.generations-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.gen-count {
  font-size: 13px;
  font-weight: 400;
  color: var(--color-text-muted);
  margin-left: 4px;
}

.gen-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.empty-state,
.loading {
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
