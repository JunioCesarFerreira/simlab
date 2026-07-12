<template>
  <div class="detail-page">
    <!-- Loading / error states -->
    <div v-if="store.loading && !store.experiment" class="loading">
      Loading experiment…
    </div>
    <div v-else-if="store.error" class="error-banner">
      Error: {{ store.error }}
    </div>

    <!-- Individual detail panel — fixed overlay, rendered on top of main content -->
    <IndividualDetailPanel
      v-if="selectedIndividual"
      :individual="selectedIndividual"
      :objective-names="store.objectiveNames"
      :metric-columns="metricColumns"
      :is-synthetic="isSynthetic"
      @close="selectedIndividual = null"
    />

    <template v-if="store.experiment">
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
              <span v-if="isSynthetic" class="badge-synthetic">
                ⬡ Synthetic — {{ syntheticBench }}
              </span>
              <span v-if="store.isRunning" class="live-pill">● LIVE</span>
              <span v-if="store.experiment.created_time" class="meta-date">
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
              v-if="!isSynthetic"
              class="action-btn"
              :disabled="downloading.topologies"
              @click="doDownloadTopologies"
            >
              {{ downloading.topologies ? "Downloading…" : "Download topologies" }}
            </button>
            <button
              class="action-btn danger-btn"
              :disabled="deleting"
              @click="doDelete"
            >
              {{ deleting ? "Deleting…" : "Delete experiment" }}
            </button>
          </div>
        </div>
      </div>

      <!-- Timeline row -->
      <div v-if="store.experiment.start_time" class="timeline-bar">
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
                title="Visualize problem"
                @click.prevent="showProblemViz = true"
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
          <ResizableChartCard v-model="paretoH" label="Pareto front chart">
            <div class="section-title pareto-title">
              Pareto Front
              <div class="pareto-toggles">
                <div class="view-toggle" title="Scope of the displayed Pareto front">
                  <button
                    :class="['vt-btn', { active: paretoScope === 'all' }]"
                    @click="paretoScope = 'all'"
                  >All solutions</button>
                  <button
                    :class="['vt-btn', { active: paretoScope === 'last' }]"
                    @click="paretoScope = 'last'"
                  >Last generation</button>
                </div>
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
            </div>
            <ParetoFrontChart
              v-if="chartView === '2d'"
              :pareto-front="displayedParetoFront"
              :generations="paretoGenerations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
              :rank-map="individualRankMap"
              :chromosome-map="store.chromosomeToIndividualId"
              :init-x="paretoXKey"
              :init-y="paretoYKey"
              @click-individual="openIndividual"
              @axis-change="onParetoAxisChange"
            />
            <ParetoFront3DChart
              v-else
              :pareto-front="displayedParetoFront"
              :generations="paretoGenerations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
              :rank-map="individualRankMap"
              :chromosome-map="store.chromosomeToIndividualId"
              :strategy="store.experiment.parameters.strategy"
              :reference-point-divisions="store.experiment.parameters.algorithm?.divisions"
              :init-x="pareto3dXKey"
              :init-y="pareto3dYKey"
              :init-z="pareto3dZKey"
              @click-individual="openIndividual"
              @axis-change="on3dAxisChange"
            />
          </ResizableChartCard>
          <ResizableChartCard v-model="hvgdH" label="HV and GD chart">
            <div class="section-title">HV &amp; GD per generation</div>
            <HvGdChart
              :experiment-id="props.id"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
            />
          </ResizableChartCard>
          <ResizableChartCard v-model="evolutionH" label="Objectives evolution chart">
            <div class="section-title">Objectives evolution (best per generation)</div>
            <ObjectivesEvolutionChart
              :generations="store.experiment.generations"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
            />
          </ResizableChartCard>
          <ResizableChartCard v-model="parallelH" label="Parallel coordinates chart" :max="800">
            <div class="section-title">Parallel coordinates — Pareto solutions</div>
            <ParetoParallelChart
              :pareto-front="store.experiment.pareto_front"
              :chromosome-map="store.chromosomeToIndividualId"
              :objective-names="store.objectiveNames"
              :objective-goals="store.objectiveGoals"
              @click-individual="openIndividual"
            />
          </ResizableChartCard>
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
            :is-synthetic="isSynthetic"
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
import { ref, reactive, computed, watch, nextTick, onMounted, onBeforeUnmount } from "vue";
import { useExperimentDetailStore } from "../app/stores/experimentDetailStore";
import { useExperimentViewState } from "../composables/useExperimentViewState";
import StatusBadge from "../components/common/StatusBadge.vue";
import GenerationRow from "../components/detail/GenerationRow.vue";
import ParetoFrontChart from "../components/charts/ParetoFrontChart.vue";
import { defineAsyncComponent } from "vue";
const ParetoFront3DChart = defineAsyncComponent(
  () => import("../components/charts/ParetoFront3DChart.vue"),
);
import ObjectivesEvolutionChart from "../components/charts/ObjectivesEvolutionChart.vue";
import HvGdChart from "../components/charts/HvGdChart.vue";
import ParetoParallelChart from "../components/charts/ParetoParallelChart.vue";
import IndividualDetailPanel from "../components/detail/IndividualDetailPanel.vue";
import ProblemVizModal from "../components/detail/ProblemVizModal.vue";
import { downloadAnalysisZip, downloadTopologiesZip } from "../api/files";
import { updateExperiment, plotParetoResults, deleteExperiment } from "../api/experiments";
import { useRouter } from "vue-router";
import ResizableChartCard from "../components/common/ResizableChartCard.vue";
import { confirmDialog } from "../composables/useConfirm";
import { reportRuntimeError } from "../composables/useRuntimeError";
import type { IndividualDto, JsonObject, ParetoFrontItemDto } from "../types/simlab";
import { isPenalized } from "../types/simlab";
import { computeRanks, computeRanksWithDuplicates } from "../utils/nonDominatedSort";

const props = defineProps<{ id: string }>();
const store = useExperimentDetailStore();
const router = useRouter();

// Delete experiment (and all its owned artifacts, server-side)
const deleting = ref(false);
async function doDelete() {
  const name = store.experiment?.name ?? "this experiment";
  const ok = await confirmDialog({
    title: `Delete experiment "${name}"?`,
    message:
      "This permanently removes the experiment and all its artifacts " +
      "(generations, individuals, simulations and their files). Shared source code is not affected. " +
      "This cannot be undone.",
    confirmLabel: "Delete",
    danger: true,
  });
  if (!ok) return;
  deleting.value = true;
  try {
    await deleteExperiment(props.id);
    router.push("/experiments");
  } catch (e) {
    reportRuntimeError(e, "Failed to delete experiment");
    deleting.value = false;
  }
}

// 2D / 3D toggle — only shown when there are ≥ 3 objectives
const has3Objectives = computed(() => (store.objectiveNames?.length ?? 0) >= 3);

const isSynthetic = computed(
  () => store.experiment?.parameters?.simulation?.synthetic?.enabled ?? false,
);
const syntheticBench = computed(
  () => store.experiment?.parameters?.simulation?.synthetic?.bench ?? "Synthetic",
);

// Persistent per-experiment view state (survives navigation away and back)
const viewState = useExperimentViewState(props.id);

// Two-way binding into the persisted view state: reads come from (and writes
// go straight to) `viewState`, so no ref+watch pair per field is needed.
function persisted<K extends keyof typeof viewState.value>(key: K) {
  return computed({
    get: () => viewState.value[key],
    set: (v: (typeof viewState.value)[K]) => { viewState.value[key] = v; },
  });
}

const chartView = persisted("chartView");

// Pareto front scope: over all explored solutions, or only the last generation
const paretoScope = persisted("paretoScope");
// Reset to 2D if objectives data loads and turns out there aren't 3
watch(has3Objectives, (has3) => { if (!has3 && chartView.value === '3d') chartView.value = '2d'; });

// Per-card resizable height
const paretoH = persisted("paretoH");
const hvgdH = persisted("hvgdH");
const evolutionH = persisted("evolutionH");
const parallelH = persisted("parallelH");

// Axis selections — passed to chart components as initial values
const paretoXKey = persisted("paretoXKey");
const paretoYKey = persisted("paretoYKey");
const pareto3dXKey = persisted("pareto3dXKey");
const pareto3dYKey = persisted("pareto3dYKey");
const pareto3dZKey = persisted("pareto3dZKey");

function onParetoAxisChange({ x, y }: { x: string; y: string }) {
  paretoXKey.value = x;
  paretoYKey.value = y;
}

function on3dAxisChange({ x, y, z }: { x: string; y: string; z: string }) {
  pareto3dXKey.value = x;
  pareto3dYKey.value = y;
  pareto3dZKey.value = z;
}

const sortedGenerations = computed(() =>
  [...(store.experiment?.generations ?? [])].sort((a, b) => a.index - b.index),
);

// ── Pareto front scope (all solutions vs. last generation) ────────────────────

// Highest-index generation that actually has a population
const lastGeneration = computed(() => {
  const gens = sortedGenerations.value;
  for (let i = gens.length - 1; i >= 0; i--) {
    if ((gens[i]?.population.length ?? 0) > 0) return gens[i];
  }
  return undefined;
});

// Generations fed to the Pareto charts — restricted to the last one when scoped.
// Both charts derive their non-dominated fronts and population from this list,
// so limiting it to the last generation makes them show that generation only.
const paretoGenerations = computed(() => {
  if (paretoScope.value === "last") {
    return lastGeneration.value ? [lastGeneration.value] : [];
  }
  return store.experiment?.generations ?? [];
});

// Non-dominated rank per individual (0 = best front), over the FULL objective
// vector, computed from `paretoGenerations` (so it respects the all/last-gen
// scope toggle above). This is O(n²) and used to be recomputed independently
// inside BOTH ParetoFrontChart and ParetoFront3DChart — every poll tick while
// the experiment is running, and again every time the user toggled between
// the 2D/3D views (which unmounts one and throws its cached result away).
// Computing it once here, in the parent that never unmounts across that
// toggle, means it's shared by whichever chart is currently visible.
const individualRankMap = computed<Map<string, number>>(() => {
  const goals = store.objectiveGoals;
  if (goals.length === 0) return new Map();
  const minimize = goals.map((g) => g === "min");
  const seen = new Set<string>();
  const points: { id: string; objectives: number[] }[] = [];
  for (const gen of paretoGenerations.value) {
    for (const ind of gen.population) {
      if (seen.has(ind.individual_id)) continue;
      seen.add(ind.individual_id);
      if (isPenalized(ind.objectives)) continue;
      points.push({ id: ind.individual_id, objectives: ind.objectives });
    }
  }
  return computeRanksWithDuplicates(points, minimize);
});

// Pareto front points passed to the charts. For "all" this is the global front
// computed server-side; for "last" it is the non-dominated set of the last
// generation, recomputed client-side.
const displayedParetoFront = computed<ParetoFrontItemDto[] | null | undefined>(() => {
  if (paretoScope.value === "all") return store.experiment?.pareto_front;

  const gen = lastGeneration.value;
  if (!gen) return [];

  const names = store.objectiveNames;
  const goals = store.objectiveGoals;

  const toItem = (ind: IndividualDto): ParetoFrontItemDto => ({
    chromosome: ind.chromosome,
    objectives: Object.fromEntries(
      ind.objectives.map((v, i) => [names[i] ?? `obj${i}`, v]),
    ),
  });

  const valid = gen.population.filter((ind) => !isPenalized(ind.objectives));
  // Without goals domination can't be evaluated — fall back to all feasible points
  if (goals.length === 0) return valid.map(toItem);

  const minimize = goals.map((g) => g === "min");
  const ranks = computeRanks(
    valid.map((ind) => ({ id: ind.individual_id, objectives: ind.objectives })),
    minimize,
  );
  return valid.filter((ind) => (ranks.get(ind.individual_id) ?? 0) === 0).map(toItem);
});

const finishedCount = computed(
  () =>
    sortedGenerations.value.filter((g) => g.status === "Done").length,
);

const totalGenerations = computed(
  () =>
    store.experiment?.parameters?.algorithm?.number_of_generations ??
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
const simulationSeeds = computed<number[]>(
  () => store.experiment?.parameters?.simulation?.random_seeds ?? [],
);

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
  const popSize = exp.parameters.algorithm?.population_size ?? 0;
  const numGens = exp.parameters.algorithm?.number_of_generations ?? 0;
  const seedsCount =
    simulationSeeds.value.length || exp.parameters.simulation?.random_seeds_count || 0;
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
    reportRuntimeError(e, "Failed to rename experiment");
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
  // Drop whichever experiment the singleton store still holds from a previous
  // visit, so the page shows a loading state instead of stale data.
  store.clear();
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

.badge-synthetic {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 10px;
  font-size: 11px;
  font-weight: 700;
  color: #d97706;
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.3);
  border-radius: 999px;
  white-space: nowrap;
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

.danger-btn { color: var(--status-error); border-color: #fecaca; background: #fee2e2; }
.danger-btn:hover:not(:disabled) { background: #fecaca; border-color: var(--status-error); color: var(--status-error); }
.danger-btn:disabled { opacity: 0.5; cursor: default; }

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

/* card chrome + resize handle live in ResizableChartCard */

.pareto-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.pareto-toggles {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
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
