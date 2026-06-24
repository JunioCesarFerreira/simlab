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
            <div class="name-row">
              <template v-if="!editing">
                <h1 class="exp-name">{{ store.experiment.name }}</h1>
                <button class="edit-name-btn" title="Edit name" @click="startEdit">✎</button>
              </template>
              <template v-else>
                <input
                  ref="editInput"
                  v-model="draftName"
                  class="name-input"
                  @keydown.enter="saveEdit"
                  @keydown.escape="cancelEdit"
                />
                <button class="edit-confirm-btn" :disabled="savingName || !draftName.trim()" @click="saveEdit">✓</button>
                <button class="edit-cancel-btn" :disabled="savingName" @click="cancelEdit">✕</button>
              </template>
            </div>
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
              v-if="has3Objectives"
              class="action-btn pareto-btn"
              :class="{ 'pareto-running': plotParetoState === 'running', 'pareto-done': plotParetoState === 'done', 'pareto-error': plotParetoState === 'error' }"
              :disabled="plotParetoState === 'running'"
              :title="plotParetoState === 'error' ? plotParetoError : undefined"
              @click="doPlotPareto"
            >
              <span v-if="plotParetoState === 'running'" class="spinner" />
              {{ plotParetoState === 'running' ? 'Running analysis…' : plotParetoState === 'done' ? '✓ Analysis done' : plotParetoState === 'error' ? '✗ Analysis failed' : 'Plot Pareto results' }}
            </button>
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
            <summary>Simulation</summary>
            <div class="param-table">
              <div class="param-row">
                <span class="pk">total simulations</span>
                <span class="pv">
                  {{ totalSimulations }}
                  <span v-if="expectedSimulations !== null" class="muted-inline">
                    / expected {{ expectedSimulations }}
                  </span>
                </span>
              </div>
              <div class="param-row">
                <span class="pk">seeds count</span>
                <span class="pv">{{ simulationSeeds.length }}</span>
              </div>
              <div v-if="simulationSeeds.length > 0" class="param-row seeds-row">
                <span class="pk">seeds</span>
                <span class="pv seeds-list mono">
                  <span
                    v-for="seed in simulationSeeds"
                    :key="seed"
                    class="seed-pill"
                  >{{ seed }}</span>
                </span>
              </div>
              <div
                v-for="(val, key) in simulationOtherParams"
                :key="key"
                class="param-row"
              >
                <span class="pk">{{ key }}</span>
                <span class="pv mono">{{ formatParamVal(val) }}</span>
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
          <div class="card chart-card" :style="{ height: paretoH + 'px' }">
            <div class="section-title pareto-title">
              Pareto Front
              <div v-if="has3Objectives" class="view-toggle">
                <button
                  :class="['vt-btn', { active: chartView === '2d' }]"
                  @click="chartView = '2d'"
                >2D</button>
                <button
                  :class="['vt-btn', { active: chartView === '3d' }]"
                  @click="chartView = '3d'"
                >3D</button>
              </div>
            </div>
            <ParetoFrontChart
              v-if="chartView === '2d'"
              :pareto-front="store.experiment.pareto_front"
              :generations="store.experiment.generations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
              @click-individual="openIndividual"
            />
            <ParetoFront3DChart
              v-else
              :pareto-front="store.experiment.pareto_front"
              :generations="store.experiment.generations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
              @click-individual="openIndividual"
            />
            <div
              class="resize-handle"
              title="Arrastar para redimensionar"
              @mousedown="startParetoResize"
            />
          </div>
          <div class="card chart-card" :style="{ height: hvgdH + 'px' }">
            <div class="section-title">HV &amp; GD per generation</div>
            <HvGdChart
              :experiment-id="props.id"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
            />
            <div
              class="resize-handle"
              title="Arrastar para redimensionar"
              @mousedown="startHvGdResize"
            />
          </div>
          <div class="card chart-card" :style="{ height: evolutionH + 'px' }">
            <div class="section-title">Objectives evolution (best per generation)</div>
            <ObjectivesEvolutionChart
              :generations="store.experiment.generations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
            />
            <div
              class="resize-handle"
              title="Arrastar para redimensionar"
              @mousedown="startEvoResize"
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
            @select="selectedIndividual = $event"
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
import { ref, reactive, computed, nextTick, onMounted, onBeforeUnmount } from "vue";
import { useExperimentDetailStore } from "../app/stores/experimentDetailStore";
import StatusBadge from "../components/common/StatusBadge.vue";
import GenerationRow from "../components/detail/GenerationRow.vue";
import ParetoFrontChart from "../components/charts/ParetoFrontChart.vue";
import { defineAsyncComponent } from "vue";
const ParetoFront3DChart = defineAsyncComponent(
  () => import("../components/charts/ParetoFront3DChart.vue"),
);
import ObjectivesEvolutionChart from "../components/charts/ObjectivesEvolutionChart.vue";
import HvGdChart from "../components/charts/HvGdChart.vue";
import IndividualDetailPanel from "../components/detail/IndividualDetailPanel.vue";
import ProblemVizModal from "../components/detail/ProblemVizModal.vue";
import { downloadAnalysisZip, downloadTopologiesZip } from "../api/files";
import { updateExperiment, plotParetoResults } from "../api/experiments";
import { useResizable } from "../composables/useResizable";
import type { IndividualDto, JsonObject } from "../types/simlab";

const props = defineProps<{ id: string }>();
const store = useExperimentDetailStore();

// 2D / 3D toggle — only shown when there are ≥ 3 objectives
const chartView = ref<"2d" | "3d">("2d");
const has3Objectives = computed(() => (store.objectiveNames?.length ?? 0) >= 3);

// Per-card resizable height
const { height: paretoH, startResize: startParetoResize } = useResizable({ initial: 420 });
const { height: hvgdH, startResize: startHvGdResize } = useResizable({ initial: 280 });
const { height: evolutionH, startResize: startEvoResize } = useResizable({ initial: 380 });

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

// Simulation parameters derived view
const simulationSeeds = computed<number[]>(() => {
  const raw = store.experiment?.parameters?.simulation?.["random_seeds"];
  return Array.isArray(raw) ? raw.map((v) => Number(v)) : [];
});

const simulationOtherParams = computed(() => {
  const sim = store.experiment?.parameters?.simulation ?? {};
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(sim)) {
    if (k !== "random_seeds") out[k] = v;
  }
  return out;
});

const totalSimulations = computed(() => {
  const gens = store.experiment?.generations ?? [];
  let total = 0;
  for (const g of gens) {
    for (const ind of g.population) {
      total += ind.simulations_ids?.length ?? 0;
    }
  }
  return total;
});

const expectedSimulations = computed<number | null>(() => {
  const exp = store.experiment;
  if (!exp) return null;
  const popSize = Number(exp.parameters.algorithm?.["population_size"]);
  const numGens = Number(exp.parameters.algorithm?.["number_of_generations"]);
  const seedsCount =
    simulationSeeds.value.length ||
    Number(exp.parameters.simulation?.["random_seeds_count"]) ||
    0;
  if (!popSize || !numGens || !seedsCount) return null;
  return popSize * numGens * seedsCount;
});

// Inline name editing
const editing = ref(false);
const draftName = ref("");
const savingName = ref(false);
const editInput = ref<HTMLInputElement | null>(null);

function startEdit() {
  draftName.value = store.experiment!.name;
  editing.value = true;
  nextTick(() => editInput.value?.focus());
}

function cancelEdit() {
  editing.value = false;
}

async function saveEdit() {
  if (!draftName.value.trim() || savingName.value) return;
  savingName.value = true;
  try {
    await updateExperiment(props.id, { name: draftName.value.trim() });
    store.experiment!.name = draftName.value.trim();
    editing.value = false;
  } catch (e) {
    console.error("Failed to update experiment name:", e);
  } finally {
    savingName.value = false;
  }
}

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

const plotParetoState = ref<"idle" | "running" | "done" | "error">("idle");
const plotParetoError = ref("");

async function doPlotPareto() {
  if (plotParetoState.value === "running") return;
  const objectives = store.objectiveNames.slice(0, 3);
  const goals = store.objectiveGoals.slice(0, 3);
  const minimize = goals.map((g) => g === "min");
  plotParetoState.value = "running";
  plotParetoError.value = "";
  try {
    await plotParetoResults(props.id, objectives, minimize);
    plotParetoState.value = "done";
    await store.refresh(props.id);
    setTimeout(() => { plotParetoState.value = "idle"; }, 4000);
  } catch (e) {
    plotParetoError.value = e instanceof Error ? e.message : String(e);
    plotParetoState.value = "error";
    setTimeout(() => { plotParetoState.value = "idle"; }, 6000);
  }
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

.name-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.exp-name {
  font-size: 24px;
  font-weight: 800;
  line-height: 1.2;
  margin: 0;
}

.edit-name-btn {
  font-size: 16px;
  color: var(--color-text-muted);
  padding: 2px 6px;
  border-radius: var(--radius-sm);
  opacity: 0;
  transition: opacity 0.15s;
}

.name-row:hover .edit-name-btn {
  opacity: 1;
}

.edit-name-btn:hover {
  background: var(--color-border);
  color: var(--color-text);
}

.name-input {
  font-size: 22px;
  font-weight: 800;
  padding: 2px 8px;
  border: 2px solid var(--color-primary);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  outline: none;
  min-width: 240px;
}

.edit-confirm-btn {
  font-size: 14px;
  font-weight: 700;
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  background: var(--color-primary);
  color: #fff;
  border: none;
}

.edit-confirm-btn:disabled { opacity: 0.5; cursor: default; }

.edit-cancel-btn {
  font-size: 13px;
  padding: 3px 8px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
}

.edit-cancel-btn:hover:not(:disabled) { background: var(--color-border); }
.edit-cancel-btn:disabled { opacity: 0.5; cursor: default; }

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

.pareto-btn { display: flex; align-items: center; gap: 6px; }
.pareto-btn.pareto-running { border-color: var(--color-primary); color: var(--color-primary); opacity: 0.8; cursor: wait; }
.pareto-btn.pareto-done { border-color: #16a34a; color: #16a34a; }
.pareto-btn.pareto-error { border-color: #dc2626; color: #dc2626; }

.spinner {
  width: 12px;
  height: 12px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}

@keyframes spin { to { transform: rotate(360deg); } }

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

.muted-inline {
  color: var(--color-text-muted);
  font-weight: 400;
  font-size: 11px;
  margin-left: 4px;
}

.seeds-row {
  flex-direction: column;
  align-items: stretch;
  gap: 4px;
}

.seeds-list {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  text-align: left;
}

.seed-pill {
  display: inline-block;
  padding: 1px 7px;
  font-size: 11px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
}

/* Charts */
.charts-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chart-card {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  /* height controlled by :style binding — min enforced by useResizable */
}

.resize-handle {
  flex-shrink: 0;
  height: 10px;
  margin: 4px -16px -16px;  /* bleed to card edges, absorb card padding-bottom */
  cursor: ns-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  border-top: 1px solid var(--color-border);
  background: transparent;
  transition: background 0.15s;
}

.resize-handle::after {
  content: '';
  width: 36px;
  height: 3px;
  border-radius: 99px;
  background: var(--color-border);
  transition: background 0.15s, transform 0.15s;
}

.resize-handle:hover {
  background: var(--color-surface-hover);
}

.resize-handle:hover::after {
  background: var(--color-text-muted);
  transform: scaleX(1.25);
}

.pareto-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.view-toggle {
  display: flex;
  gap: 2px;
  padding: 2px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
}

.vt-btn {
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 600;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  color: var(--color-text-muted);
  background: transparent;
  transition: background 0.12s, color 0.12s;
  letter-spacing: 0.03em;
}

.vt-btn:hover {
  color: var(--color-text);
}

.vt-btn.active {
  background: var(--color-surface);
  color: var(--color-primary);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
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
