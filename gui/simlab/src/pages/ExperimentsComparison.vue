<template>
  <div class="compare-page">
    <!-- ── Header ──────────────────────────────────────────────────────────── -->
    <div class="header">
      <RouterLink to="/campaigns" class="back-link">← Campaigns</RouterLink>
      <h1 class="page-title">Compare Experiments</h1>
    </div>

    <!-- ── Selectors ──────────────────────────────────────────────────────── -->
    <div class="card selectors-card">
      <div class="selectors-row">
        <div class="sel-group">
          <label class="sel-label">Campaign</label>
          <select v-model="campaignId" class="sel" @change="onCampaignChange">
            <option value="">Select campaign…</option>
            <option v-for="c in allCampaigns" :key="c.id" :value="c.id">{{ c.name }}</option>
          </select>
        </div>

        <div class="sel-group">
          <label class="sel-label">Experiment A</label>
          <select v-model="exp1Id" class="sel sel-a" :disabled="!doneExps.length">
            <option value="">Select…</option>
            <option
              v-for="e in doneExps"
              :key="e.id"
              :value="e.id"
              :disabled="e.id === exp2Id"
            >{{ e.name }}</option>
          </select>
        </div>

        <span class="vs-badge">vs</span>

        <div class="sel-group">
          <label class="sel-label">Experiment B</label>
          <select v-model="exp2Id" class="sel sel-b" :disabled="!doneExps.length">
            <option value="">Select…</option>
            <option
              v-for="e in doneExps"
              :key="e.id"
              :value="e.id"
              :disabled="e.id === exp1Id"
            >{{ e.name }}</option>
          </select>
        </div>

        <button class="compare-btn" :disabled="!canCompare || loading" @click="runComparison">
          {{ loading ? 'Loading…' : 'Compare' }}
        </button>
      </div>

      <div v-if="loadingCampaigns" class="sel-hint">Loading campaigns…</div>
      <div v-else-if="campaignId && !doneExps.length" class="sel-hint">
        No finished experiments in this campaign.
      </div>
    </div>

    <!-- ── Error ──────────────────────────────────────────────────────────── -->
    <div v-if="error" class="error-banner">{{ error }}</div>

    <!-- ── Results ────────────────────────────────────────────────────────── -->
    <template v-if="result">
      <!-- Objectives mismatch warning -->
      <div v-if="objectivesMismatch" class="warn-banner">
        Warning: experiments have different objectives. Metrics are computed on the
        {{ sharedObjectives.length }} shared objective(s) only.
      </div>

      <!-- ── Parameters table ─────────────────────────────────────────────── -->
      <div class="card">
        <h2 class="card-title">Parameters</h2>
        <div class="table-wrap">
          <table class="cmp-table">
            <thead>
              <tr>
                <th class="col-param">Parameter</th>
                <th class="col-a-head">{{ result.expA.name }}</th>
                <th class="col-b-head">{{ result.expB.name }}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Strategy</td>
                <td>{{ result.expA.parameters.strategy }}</td>
                <td>{{ result.expB.parameters.strategy }}</td>
              </tr>
              <tr v-for="k in algParamKeys" :key="k">
                <td>{{ k }}</td>
                <td>{{ fmtParam(result.expA.parameters.algorithm[k]) }}</td>
                <td>{{ fmtParam(result.expB.parameters.algorithm[k]) }}</td>
              </tr>
              <tr>
                <td>Objectives</td>
                <td>{{ fmtObjectives(result.expA.parameters.objectives) }}</td>
                <td>{{ fmtObjectives(result.expB.parameters.objectives) }}</td>
              </tr>
              <tr>
                <td>Seeds count</td>
                <td>{{ seedsCount(result.expA) }}</td>
                <td>{{ seedsCount(result.expB) }}</td>
              </tr>
              <tr>
                <td>Sim. duration (s)</td>
                <td>{{ fmtParam(result.expA.parameters.simulation?.duration) }}</td>
                <td>{{ fmtParam(result.expB.parameters.simulation?.duration) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ── Metrics summary ──────────────────────────────────────────────── -->
      <div class="card">
        <h2 class="card-title">Performance Metrics</h2>
        <p class="card-note">
          HV and GD are sourced from the backend per experiment (reference = experiment-specific
          worst point). C-metric, IGD+, ε-indicator and Spacing are computed client-side from
          final Pareto fronts. For cross-experiment metrics (IGD+, ε), lower means the annotated
          front is a better approximation of the other.
        </p>
        <div class="table-wrap">
          <table class="cmp-table metrics-table">
            <thead>
              <tr>
                <th class="col-metric">Metric</th>
                <th class="col-desc">Description</th>
                <th class="col-a-head">{{ result.expA.name }}</th>
                <th class="col-b-head">{{ result.expB.name }}</th>
                <th class="col-winner">Better</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in metricsRows" :key="row.name">
                <td><code class="metric-name">{{ row.name }}</code></td>
                <td class="desc-cell">{{ row.desc }}</td>
                <td :class="['val-cell', rowWinner(row) === 'A' ? 'winner-a' : '']">
                  {{ row.valA !== null ? fmtNum(row.valA) : '—' }}
                </td>
                <td :class="['val-cell', rowWinner(row) === 'B' ? 'winner-b' : '']">
                  {{ row.valB !== null ? fmtNum(row.valB) : '—' }}
                </td>
                <td class="winner-cell">
                  <span v-if="rowWinner(row) === 'A'" class="badge badge-a">
                    {{ result.expA.name }}
                  </span>
                  <span v-else-if="rowWinner(row) === 'B'" class="badge badge-b">
                    {{ result.expB.name }}
                  </span>
                  <span v-else class="badge-none">—</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ── Pareto front ─────────────────────────────────────────────────── -->
      <ResizableChartCard
        v-if="sharedObjectives.length >= 2"
        v-model="paretoH"
        class="cmp-chart-card"
        label="Pareto front comparison chart"
      >
        <div class="section-title pareto-title">
          Pareto Front
          <div class="pareto-title-right">
            <!-- 2D axis selectors (only in 2D when > 2 objectives) -->
            <div v-if="chartView === '2d' && sharedObjectives.length > 2" class="axis-sel-row">
              <label class="axis-label">X:
                <select v-model="axisX" class="axis-sel" @change="renderParetoChart">
                  <option v-for="o in sharedObjectives" :key="o.metric_name" :value="o.metric_name">
                    {{ o.metric_name }}
                  </option>
                </select>
              </label>
              <label class="axis-label">Y:
                <select v-model="axisY" class="axis-sel" @change="renderParetoChart">
                  <option v-for="o in sharedObjectives" :key="o.metric_name" :value="o.metric_name">
                    {{ o.metric_name }}
                  </option>
                </select>
              </label>
            </div>
            <!-- 2D / 3D toggle — only when ≥ 3 shared objectives -->
            <div v-if="sharedObjectives.length >= 3" class="view-toggle">
              <button
                :class="['vt-btn', { active: chartView === '2d' }]"
                @click="chartView = '2d'"
              >2D</button>
              <button
                :class="['vt-btn', { active: chartView === '3d' }]"
                @click="chartView = '3d'"
              >3D</button>
            </div>
            <!-- The 3D view (ParetoFront3DComparisonChart) carries its own export button -->
            <ChartExportButton v-if="chartView === '2d'" @click="handleExportParetoImage" />
          </div>
        </div>

        <!-- 2D scatter -->
        <div v-if="chartView === '2d'" ref="paretoEl" class="chart-fill" />

        <!-- 3D component -->
        <ParetoFront3DComparisonChart
          v-else
          :front-a="result.expA.pareto_front ?? []"
          :front-b="result.expB.pareto_front ?? []"
          :name-a="result.expA.name"
          :name-b="result.expB.name"
          :objectives="sharedObjectives"
        />

      </ResizableChartCard>

      <!-- ── Convergence charts ────────────────────────────────────────────── -->
      <ResizableChartCard
        v-if="result.hvgdA || result.hvgdB"
        v-model="convergenceH"
        class="cmp-chart-card"
        label="Convergence charts"
      >
        <div class="section-title">Convergence</div>
        <div class="evo-row">
          <div class="evo-block">
            <div class="evo-label-row">
              <span class="evo-label">Hypervolume (HV)</span>
              <ChartExportButton @click="handleExportEvoImage('hv')" />
            </div>
            <div ref="hvEl" class="chart-fill" />
          </div>
          <div class="evo-block">
            <div class="evo-label-row">
              <span class="evo-label">Generational Distance (GD)</span>
              <ChartExportButton @click="handleExportEvoImage('gd')" />
            </div>
            <div ref="gdEl" class="chart-fill" />
          </div>
        </div>
      </ResizableChartCard>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, nextTick, defineAsyncComponent } from 'vue';
import { RouterLink } from 'vue-router';
import type * as echarts from 'echarts';
import { useEChart } from '../composables/useEChart';
import { useTheme } from '../composables/useTheme';
import { chartPalette, chartExportBackground } from '../services/chartTheme';
import ResizableChartCard from '../components/common/ResizableChartCard.vue';
import ChartExportButton from '../components/charts/ChartExportButton.vue';
import { getAllCampaigns, getCampaignFull } from '../api/campaigns';
import { getExperiment } from '../api/experiments';
import client from '../api/client';
import type { CampaignInfoDto, ExperimentDto, ObjectiveItem } from '../types/simlab';
import { extractFront, coverage, epsilonIndicator, igdPlus, spacing } from '../utils/comparisonMetrics';
import { chartExportFilename } from '../utils/chartExport';

const ParetoFront3DComparisonChart = defineAsyncComponent(
  () => import('../components/charts/ParetoFront3DComparisonChart.vue'),
);

// ── Types ────────────────────────────────────────────────────────────────────

interface HvGdData {
  generations: number[];
  hv: number[];
  gd: number[];
  worst_point: Record<string, number>;
}

interface ComparisonResult {
  expA: ExperimentDto;
  expB: ExperimentDto;
  hvgdA: HvGdData | null;
  hvgdB: HvGdData | null;
}

interface MetricRow {
  name: string;
  desc: string;
  valA: number | null;
  valB: number | null;
  higherBetter: boolean | null;
}

// ── Theme ────────────────────────────────────────────────────────────────────

const { isDark } = useTheme();


// ── Resizable cards ──────────────────────────────────────────────────────────

const paretoH = ref(420);
const convergenceH = ref(300);

// ── State ────────────────────────────────────────────────────────────────────

const allCampaigns = ref<CampaignInfoDto[]>([]);
const loadingCampaigns = ref(false);
const campaignId = ref('');
const doneExps = ref<ExperimentDto[]>([]);
const exp1Id = ref('');
const exp2Id = ref('');
const loading = ref(false);
const error = ref('');
const result = ref<ComparisonResult | null>(null);
const chartView = ref<'2d' | '3d'>('2d');
const axisX = ref('');
const axisY = ref('');

// ── Derived ──────────────────────────────────────────────────────────────────

const canCompare = computed(
  () => exp1Id.value && exp2Id.value && exp1Id.value !== exp2Id.value,
);

const sharedObjectives = computed<ObjectiveItem[]>(() => {
  if (!result.value) return [];
  const { expA, expB } = result.value;
  const namesB = new Set(expB.parameters.objectives.map(o => o.metric_name));
  return expA.parameters.objectives.filter(o => namesB.has(o.metric_name));
});

const objectivesMismatch = computed(() => {
  if (!result.value) return false;
  const a = result.value.expA.parameters.objectives.length;
  const b = result.value.expB.parameters.objectives.length;
  return a !== b || sharedObjectives.value.length !== a;
});

const isMinArr = computed(() => sharedObjectives.value.map(o => o.goal === 'min'));

const frontA = computed(() => {
  if (!result.value?.expA.pareto_front || !sharedObjectives.value.length) return [];
  return extractFront(result.value.expA.pareto_front, sharedObjectives.value);
});

const frontB = computed(() => {
  if (!result.value?.expB.pareto_front || !sharedObjectives.value.length) return [];
  return extractFront(result.value.expB.pareto_front, sharedObjectives.value);
});

const algParamKeys = computed<string[]>(() => {
  if (!result.value) return [];
  const keys = new Set([
    ...Object.keys(result.value.expA.parameters.algorithm),
    ...Object.keys(result.value.expB.parameters.algorithm),
  ]);
  return [...keys];
});

const metricsRows = computed<MetricRow[]>(() => {
  if (!result.value) return [];
  const fa = frontA.value;
  const fb = frontB.value;
  const im = isMinArr.value;
  const hvgdA = result.value.hvgdA;
  const hvgdB = result.value.hvgdB;

  return [
    {
      name: 'HV',
      desc: 'Hypervolume — volume of objective space dominated by the front (↑ better). Reference point = experiment-specific worst point.',
      valA: hvgdA ? (hvgdA.hv[hvgdA.hv.length - 1] ?? null) : null,
      valB: hvgdB ? (hvgdB.hv[hvgdB.hv.length - 1] ?? null) : null,
      higherBetter: true,
    },
    {
      name: 'GD',
      desc: 'Generational Distance — average distance from each front solution to the nearest reference point (↓ better).',
      valA: hvgdA ? (hvgdA.gd[hvgdA.gd.length - 1] ?? null) : null,
      valB: hvgdB ? (hvgdB.gd[hvgdB.gd.length - 1] ?? null) : null,
      higherBetter: false,
    },
    {
      name: 'C(A→B)',
      desc: "Coverage: fraction of B's solutions dominated by at least one solution in A (↑ better for A).",
      valA: fa.length && fb.length ? coverage(fa, fb, im) : null,
      valB: null,
      higherBetter: null,
    },
    {
      name: 'C(B→A)',
      desc: "Coverage: fraction of A's solutions dominated by at least one solution in B (↑ better for B).",
      valA: null,
      valB: fa.length && fb.length ? coverage(fb, fa, im) : null,
      higherBetter: null,
    },
    {
      name: 'IGD+(A→B)',
      desc: 'Modified IGD from B to A — average min-distance from each B point to A (↓ better for A; measures how well A approximates B).',
      valA: fa.length && fb.length ? (v => isFinite(v) ? v : null)(igdPlus(fa, fb, im)) : null,
      valB: null,
      higherBetter: null,
    },
    {
      name: 'IGD+(B→A)',
      desc: 'Modified IGD from A to B — average min-distance from each A point to B (↓ better for B; measures how well B approximates A).',
      valA: null,
      valB: fa.length && fb.length ? (v => isFinite(v) ? v : null)(igdPlus(fb, fa, im)) : null,
      higherBetter: null,
    },
    {
      name: 'ε(A→B)',
      desc: 'Additive ε-indicator: smallest ε such that A ε-dominates B (↓ better for A; negative means A already strictly dominates B).',
      valA: fa.length && fb.length ? (v => isFinite(v) ? v : null)(epsilonIndicator(fa, fb, im)) : null,
      valB: null,
      higherBetter: null,
    },
    {
      name: 'ε(B→A)',
      desc: 'Additive ε-indicator: smallest ε such that B ε-dominates A (↓ better for B; negative means B already strictly dominates A).',
      valA: null,
      valB: fa.length && fb.length ? (v => isFinite(v) ? v : null)(epsilonIndicator(fb, fa, im)) : null,
      higherBetter: null,
    },
    {
      name: 'Spacing',
      desc: "Schott's spacing: std. dev. of nearest-neighbor distances — measures distribution uniformity (↓ better).",
      valA: fa.length >= 2 ? spacing(fa) : null,
      valB: fb.length >= 2 ? spacing(fb) : null,
      higherBetter: false,
    },
    {
      name: '|Front|',
      desc: 'Number of non-dominated solutions in the final Pareto front (↑ better).',
      valA: fa.length,
      valB: fb.length,
      higherBetter: true,
    },
  ];
});

function rowWinner(row: MetricRow): 'A' | 'B' | null {
  if (row.higherBetter === null) return null;
  if (row.valA === null || row.valB === null) return null;
  if (row.valA === row.valB) return null;
  return row.higherBetter
    ? (row.valA > row.valB ? 'A' : 'B')
    : (row.valA < row.valB ? 'A' : 'B');
}

// ── API ───────────────────────────────────────────────────────────────────────

async function loadCampaigns() {
  loadingCampaigns.value = true;
  try {
    allCampaigns.value = await getAllCampaigns();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loadingCampaigns.value = false;
  }
}

async function onCampaignChange() {
  doneExps.value = [];
  exp1Id.value = '';
  exp2Id.value = '';
  result.value = null;
  error.value = '';
  if (!campaignId.value) return;
  try {
    const full = await getCampaignFull(campaignId.value);
    doneExps.value = full.experiments.filter(e => e.status === 'Done');
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function fetchHvGd(id: string, objectives: ObjectiveItem[]): Promise<HvGdData | null> {
  if (objectives.length < 2) return null;
  const params = new URLSearchParams();
  objectives.forEach(o => params.append('objectives', o.metric_name));
  objectives.forEach(o => params.append('minimize', o.goal === 'min' ? 'true' : 'false'));
  try {
    const { data } = await client.get<HvGdData>(`/experiments/${id}/hv-gd?${params}`);
    return data.generations?.length ? data : null;
  } catch {
    return null;
  }
}

async function runComparison() {
  if (!canCompare.value) return;
  loading.value = true;
  error.value = '';
  result.value = null;
  chartView.value = '2d';

  try {
    const [expA, expB] = await Promise.all([
      getExperiment(exp1Id.value),
      getExperiment(exp2Id.value),
    ]);

    const [hvgdA, hvgdB] = await Promise.all([
      fetchHvGd(expA.id, expA.parameters.objectives),
      fetchHvGd(expB.id, expB.parameters.objectives),
    ]);

    result.value = { expA, expB, hvgdA, hvgdB };

    const shared = sharedObjectives.value;
    axisX.value = shared[0]?.metric_name ?? '';
    axisY.value = shared[1]?.metric_name ?? shared[0]?.metric_name ?? '';

    await nextTick();
    renderParetoChart();
    renderEvolutionCharts();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

// ── Charts ────────────────────────────────────────────────────────────────────
// useEChart owns init/dispose, element recreation across v-if toggles (the 2D
// pareto chart is destroyed while the 3D view is shown) and resize handling.

const paretoEl = ref<HTMLElement | null>(null);
const hvEl = ref<HTMLElement | null>(null);
const gdEl = ref<HTMLElement | null>(null);
const paretoChart = useEChart(paretoEl);
const hvChart = useEChart(hvEl);
const gdChart = useEChart(gdEl);

function handleExportParetoImage() {
  paretoChart.exportImage(chartExportFilename('pareto-front-comparison'), {
    backgroundColor: chartExportBackground(isDark.value),
  });
}

function handleExportEvoImage(kind: 'hv' | 'gd') {
  const target = kind === 'hv' ? hvChart : gdChart;
  target.exportImage(chartExportFilename(kind === 'hv' ? 'hv-comparison' : 'gd-comparison'), {
    backgroundColor: chartExportBackground(isDark.value),
  });
}

function renderParetoChart() {
  if (!result.value) return;

  const dark = isDark.value;
  const c = chartPalette(dark);
  const objs = sharedObjectives.value;
  const xi = objs.findIndex(o => o.metric_name === axisX.value);
  const yi = objs.findIndex(o => o.metric_name === axisY.value);
  if (xi < 0 || yi < 0) return;

  const ptA = frontA.value.map(p => [p[xi], p[yi]]);
  const ptB = frontB.value.map(p => [p[xi], p[yi]]);

  const xObj = objs[xi]!;
  const yObj = objs[yi]!;
  const xLabel = `${xObj.metric_name} (${xObj.goal === 'min' ? '↓' : '↑'})`;
  const yLabel = `${yObj.metric_name} (${yObj.goal === 'min' ? '↓' : '↑'})`;

  paretoChart.setOption(
    {
      backgroundColor: c.bg,
      legend: {
        data: [result.value.expA.name, result.value.expB.name],
        textStyle: { color: c.text, fontSize: 12 },
        top: 4,
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: c.tooltip,
        borderColor: c.tooltipBorder,
        textStyle: { color: c.text, fontSize: 12 },
        formatter: (p: unknown) => {
          const { seriesName, value } = p as { seriesName: string; value: number[] };
          return `${seriesName}<br/>${xLabel}: ${fmtNum(value[0]!)}<br/>${yLabel}: ${fmtNum(value[1]!)}`;
        },
      },
      grid: { top: 40, right: 20, bottom: 50, left: 60 },
      xAxis: {
        type: 'value',
        name: xLabel,
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { color: c.muted, fontSize: 11 },
        axisLabel: { color: c.muted, fontSize: 10 },
        axisLine: { lineStyle: { color: c.grid } },
        splitLine: { lineStyle: { color: c.grid, type: 'dashed' } },
      },
      yAxis: {
        type: 'value',
        name: yLabel,
        nameLocation: 'middle',
        nameGap: 42,
        nameTextStyle: { color: c.muted, fontSize: 11 },
        axisLabel: { color: c.muted, fontSize: 10 },
        axisLine: { lineStyle: { color: c.grid } },
        splitLine: { lineStyle: { color: c.grid, type: 'dashed' } },
      },
      series: [
        {
          name: result.value.expA.name,
          type: 'scatter',
          data: ptA,
          symbolSize: 7,
          itemStyle: { color: c.colorA, opacity: 0.85 },
        },
        {
          name: result.value.expB.name,
          type: 'scatter',
          data: ptB,
          symbolSize: 7,
          itemStyle: { color: c.colorB, opacity: 0.85 },
        },
      ],
    },
    true,
  );
}

function buildEvoOption(
  nameA: string,
  nameB: string,
  dataA: [number, number][],
  dataB: [number, number][],
  seriesName: string,
  dark: boolean,
): echarts.EChartsOption {
  const c = chartPalette(dark);
  return {
    backgroundColor: c.bg,
    legend: {
      data: [nameA, nameB],
      textStyle: { color: c.text, fontSize: 11 },
      top: 2,
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
    },
    grid: { top: 36, right: 16, bottom: 40, left: 56 },
    xAxis: {
      type: 'value',
      name: 'Generation',
      nameLocation: 'middle',
      nameGap: 26,
      nameTextStyle: { color: c.muted, fontSize: 11 },
      axisLabel: { color: c.muted, fontSize: 10 },
      axisLine: { lineStyle: { color: c.grid } },
      splitLine: { lineStyle: { color: c.grid, type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      name: seriesName,
      nameTextStyle: { color: c.muted, fontSize: 11 },
      axisLabel: { color: c.muted, fontSize: 10 },
      axisLine: { lineStyle: { color: c.grid } },
      splitLine: { lineStyle: { color: c.grid, type: 'dashed' } },
    },
    series: [
      {
        name: nameA,
        type: 'line',
        data: dataA,
        smooth: true,
        symbolSize: 4,
        itemStyle: { color: c.colorA },
        lineStyle: { color: c.colorA, width: 2 },
        areaStyle: { color: c.areaA },
      },
      {
        name: nameB,
        type: 'line',
        data: dataB,
        smooth: true,
        symbolSize: 4,
        itemStyle: { color: c.colorB },
        lineStyle: { color: c.colorB, width: 2 },
        areaStyle: { color: c.areaB },
      },
    ],
  };
}

function renderEvolutionCharts() {
  if (!result.value) return;
  const { expA, expB, hvgdA, hvgdB } = result.value;
  const dark = isDark.value;
  const nameA = expA.name;
  const nameB = expB.name;

  const hvA: [number, number][] = hvgdA
    ? hvgdA.generations.map((g, i) => [g, hvgdA.hv[i]!])
    : [];
  const hvB: [number, number][] = hvgdB
    ? hvgdB.generations.map((g, i) => [g, hvgdB.hv[i]!])
    : [];
  hvChart.setOption(buildEvoOption(nameA, nameB, hvA, hvB, 'HV', dark), true);

  const gdA: [number, number][] = hvgdA
    ? hvgdA.generations.map((g, i) => [g, hvgdA.gd[i]!])
    : [];
  const gdB: [number, number][] = hvgdB
    ? hvgdB.generations.map((g, i) => [g, hvgdB.gd[i]!])
    : [];
  gdChart.setOption(buildEvoOption(nameA, nameB, gdA, gdB, 'GD', dark), true);
}

// ── Formatters ────────────────────────────────────────────────────────────────

function fmtNum(v: number): string {
  if (!isFinite(v)) return '∞';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e6 || (a > 0 && a < 0.001)) return v.toExponential(3);
  if (a >= 100) return v.toFixed(2);
  if (a >= 10) return v.toFixed(3);
  return v.toFixed(4);
}

function fmtParam(v: unknown): string {
  if (v === undefined || v === null) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function fmtObjectives(objs: ObjectiveItem[]): string {
  return objs.map(o => `${o.metric_name} (${o.goal})`).join(', ');
}

function seedsCount(exp: ExperimentDto): string {
  const raw = exp.parameters.simulation?.random_seeds;
  return Array.isArray(raw) ? String(raw.length) : '—';
}

// ── Lifecycle & watchers ──────────────────────────────────────────────────────

onMounted(loadCampaigns);

watch(isDark, () => {
  if (!result.value) return;
  if (chartView.value === '2d') renderParetoChart();
  renderEvolutionCharts();
});

watch(chartView, async (view) => {
  if (view === '2d') {
    await nextTick();
    renderParetoChart();
  }
});
</script>

<style scoped>
/* ── Page layout ──────────────────────────────────────────────────────────── */

.compare-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 24px 28px;
  max-width: 1200px;
  margin: 0 auto;
  min-height: 100%;
}

.header {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.back-link {
  font-size: 13px;
  color: var(--color-primary);
  text-decoration: none;
  width: fit-content;
}

.back-link:hover { text-decoration: underline; }

.page-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--color-text);
  margin: 0;
}

/* ── Card ─────────────────────────────────────────────────────────────────── */

.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg, 10px);
  padding: 20px 22px;
}

/* Chart cards (ResizableChartCard): this page's .card padding differs from the
   16px global default, so tell the handle how far to bleed. */
.cmp-chart-card {
  --card-pad-x: 22px;
  --card-pad-b: 20px;
}

.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text);
  margin: 0 0 12px;
}

.card-note {
  font-size: 12px;
  color: var(--color-text-muted);
  margin: -6px 0 14px;
  line-height: 1.6;
}

/* ── Section title (inside chart-card) ───────────────────────────────────── */

.section-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text);
  margin-bottom: 10px;
  flex-shrink: 0;
}

.pareto-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.pareto-title-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

/* ── 2D/3D view toggle ────────────────────────────────────────────────────── */

.view-toggle {
  display: flex;
  gap: 2px;
  padding: 2px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm, 5px);
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

.vt-btn:hover { color: var(--color-text); }

.vt-btn.active {
  background: var(--color-surface);
  color: var(--color-primary);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

/* ── Selectors ────────────────────────────────────────────────────────────── */

.selectors-card { padding: 16px 22px; }

.selectors-row {
  display: flex;
  align-items: flex-end;
  gap: 14px;
  flex-wrap: wrap;
}

.sel-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  flex: 1;
}

.sel-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-muted);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.sel {
  padding: 8px 10px;
  border-radius: var(--radius-md, 7px);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 13px;
  cursor: pointer;
  min-width: 160px;
}

.sel-a { border-color: #3b82f6; }
.sel-b { border-color: #f97316; }

.vs-badge {
  padding: 8px 0 8px;
  font-size: 13px;
  font-weight: 700;
  color: var(--color-text-muted);
  align-self: flex-end;
  flex-shrink: 0;
}

.compare-btn {
  padding: 9px 20px;
  border-radius: var(--radius-md, 7px);
  background: var(--color-primary);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  border: none;
  white-space: nowrap;
  align-self: flex-end;
  flex-shrink: 0;
  transition: opacity 0.12s;
}

.compare-btn:disabled { opacity: 0.4; cursor: default; }
.compare-btn:not(:disabled):hover { opacity: 0.88; }

.sel-hint {
  font-size: 12px;
  color: var(--color-text-muted);
  margin-top: 8px;
}

/* ── Banners ──────────────────────────────────────────────────────────────── */

.error-banner {
  padding: 12px 16px;
  background: rgba(220, 38, 38, 0.1);
  border: 1px solid rgba(220, 38, 38, 0.3);
  border-radius: var(--radius-md, 7px);
  color: #dc2626;
  font-size: 13px;
}

.warn-banner {
  padding: 12px 16px;
  background: rgba(234, 179, 8, 0.1);
  border: 1px solid rgba(234, 179, 8, 0.3);
  border-radius: var(--radius-md, 7px);
  color: #ca8a04;
  font-size: 13px;
}

/* ── Tables ───────────────────────────────────────────────────────────────── */

.table-wrap { overflow-x: auto; }

.cmp-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.cmp-table th {
  padding: 8px 12px;
  text-align: left;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--color-text-muted);
  border-bottom: 1px solid var(--color-border);
}

.cmp-table td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border);
  color: var(--color-text);
  vertical-align: top;
}

.cmp-table tr:last-child td { border-bottom: none; }

.col-a-head { color: #3b82f6 !important; }
.col-b-head { color: #f97316 !important; }

.col-param { width: 160px; }
.col-metric { width: 120px; }
.col-desc { min-width: 260px; }
.col-winner { width: 130px; }

.desc-cell {
  color: var(--color-text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.val-cell {
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-size: 13px;
}

.winner-a { background: rgba(59, 130, 246, 0.07); font-weight: 600; }
.winner-b { background: rgba(249, 115, 22, 0.07); font-weight: 600; }

.metric-name {
  font-family: monospace;
  font-size: 12px;
  background: var(--color-surface-hover, rgba(0,0,0,0.04));
  padding: 2px 6px;
  border-radius: 4px;
}

.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.badge-a { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
.badge-b { background: rgba(249, 115, 22, 0.15); color: #f97316; }
.badge-none { color: var(--color-text-muted); font-size: 12px; }

/* ── Chart fill (flex child that fills remaining card height) ─────────────── */

.chart-fill {
  flex: 1;
  min-height: 0;
}

/* ── 2D axis selectors ────────────────────────────────────────────────────── */

.axis-sel-row {
  display: flex;
  gap: 12px;
  align-items: center;
}

.axis-label {
  font-size: 12px;
  color: var(--color-text-muted);
  display: flex;
  align-items: center;
  gap: 6px;
}

.axis-sel {
  padding: 4px 8px;
  border-radius: var(--radius-sm, 5px);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  font-size: 12px;
  cursor: pointer;
}

/* ── Convergence charts ───────────────────────────────────────────────────── */

.evo-row {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.evo-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}

.evo-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-shrink: 0;
}

.evo-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
  letter-spacing: 0.03em;
  flex-shrink: 0;
}

@media (max-width: 720px) {
  .evo-row { grid-template-columns: 1fr; }
  .selectors-row { flex-direction: column; }
  .sel { min-width: unset; }
}
</style>
