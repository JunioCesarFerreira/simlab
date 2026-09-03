<template>
  <div class="hvgd-root">
    <div v-if="state === 'loading'" class="hvgd-placeholder">
      <span class="spinner" />
      Computing HV, GD &amp; IGD…
    </div>
    <div v-else-if="state === 'error'" class="hvgd-placeholder hvgd-error">
      {{ errorMsg }}
    </div>
    <div v-else-if="state === 'empty'" class="hvgd-placeholder">
      No reference front available yet.
    </div>
    <div v-else class="hvgd-charts">
      <div class="hvgd-col">
        <div class="controls-bar">
          <div class="hv-mode-toggle" role="group" aria-label="Hypervolume mode">
            <button
              type="button"
              :class="['mode-btn', { active: hvMode === 'perGen' }]"
              :aria-pressed="hvMode === 'perGen'"
              title="Hypervolume of each generation's own Pareto front"
              @click="hvMode = 'perGen'"
            >
              Per generation
            </button>
            <button
              type="button"
              :class="['mode-btn', { active: hvMode === 'cumulative' }]"
              :aria-pressed="hvMode === 'cumulative'"
              title="Best-so-far hypervolume over every generation up to each point"
              @click="hvMode = 'cumulative'"
            >
              Cumulative
            </button>
          </div>
          <ChartExportButton @click="handleExportImage('hv')" />
        </div>
        <div ref="hvEl" class="hvgd-chart" role="img" :aria-label="hvAriaLabel" />
      </div>
      <div class="hvgd-col">
        <div class="controls-bar">
          <ChartExportButton @click="handleExportImage('gd')" />
        </div>
        <div ref="gdEl" class="hvgd-chart" role="img" aria-label="Generational distance per generation chart" />
      </div>
      <div class="hvgd-col">
        <div class="controls-bar">
          <ChartExportButton @click="handleExportImage('igd')" />
        </div>
        <div ref="igdEl" class="hvgd-chart" role="img" aria-label="Inverted generational distance per generation chart" />
      </div>
    </div>
    <p v-if="state === 'ready'" class="hvgd-caption" :class="{ 'is-warning': selfReferential }">
      {{ referenceCaption }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from "vue";
import * as echarts from "../../lib/echarts";
import type { EChartsOption, DefaultLabelFormatterCallbackParams } from "echarts";
import { useTheme } from "../../composables/useTheme";
import { chartPalette, chartExportBackground } from "../../services/chartTheme";
import client from "../../api/client";
import { exportChartImage, chartExportFilename } from "../../utils/chartExport";
import ChartExportButton from "./ChartExportButton.vue";

const props = defineProps<{
  experimentId: string;
  objectiveNames: string[];
  objectiveGoals: string[];
}>();

const { isDark } = useTheme();

// ── state ──────────────────────────────────────────────────────────────────
type State = "idle" | "loading" | "ready" | "empty" | "error";
const state = ref<State>("idle");
const errorMsg = ref("");

// HV can be viewed per generation (each gen's own front) or cumulatively
// (best-so-far front over all generations so far). Both curves come from a
// single fetch, so switching is instant and never re-hits the backend.
type HvMode = "perGen" | "cumulative";
const hvMode = ref<HvMode>("perGen");

// The distance indicators are null for a generation with no feasible
// individual — a gap in the curve, which ECharts renders as a break, rather
// than a zero that would read as "perfect convergence".
interface HvGdData {
  generations: number[];
  hv: number[];
  hv_cumulative: number[];
  gd: (number | null)[];
  igd: (number | null)[];
  igd_plus: (number | null)[];
  reference: "true_front" | "final_front" | null;
  reference_size: number;
  normalized: boolean;
  worst_point: Record<string, number>;
}
const data = ref<HvGdData | null>(null);

const hvAriaLabel = computed(() =>
  hvMode.value === "cumulative"
    ? "Cumulative hypervolume chart"
    : "Hypervolume per generation chart",
);

// Against the run's own final front, GD and IGD reach zero on the last
// generation by construction: they measure how much of that front had been
// found by generation k, not convergence to the real optimum. Saying so is the
// difference between a progress curve and a misread quality claim.
const selfReferential = computed(() => data.value?.reference === "final_front");

const referenceCaption = computed(() => {
  const d = data.value;
  if (!d) return "";
  const scale = d.normalized
    ? "normalized by the reference front's ideal-nadir range"
    : "in raw objective units";
  const size = `${d.reference_size} point${d.reference_size === 1 ? "" : "s"}`;
  return selfReferential.value
    ? `GD / IGD reference: this run's own final Pareto front (${size}) — self-referential, so both reach 0 on the last generation by construction. Distances ${scale}.`
    : `GD / IGD reference: the benchmark's analytical true front (${size}). Distances ${scale}.`;
});

// ── chart instances ─────────────────────────────────────────────────────────
const hvEl = ref<HTMLElement | null>(null);
const gdEl = ref<HTMLElement | null>(null);
const igdEl = ref<HTMLElement | null>(null);
let hvChart: echarts.EChartsType | null = null;
let gdChart: echarts.EChartsType | null = null;
let igdChart: echarts.EChartsType | null = null;
let ro: ResizeObserver | null = null;

type ChartKind = "hv" | "gd" | "igd";
const EXPORT_NAMES: Record<ChartKind, string> = {
  hv: "hypervolume",
  gd: "generational-distance",
  igd: "inverted-generational-distance",
};

function handleExportImage(kind: ChartKind) {
  const chart = kind === "hv" ? hvChart : kind === "gd" ? gdChart : igdChart;
  exportChartImage(chart, chartExportFilename(EXPORT_NAMES[kind]), {
    backgroundColor: chartExportBackground(isDark.value),
  });
}

// ── fetch ───────────────────────────────────────────────────────────────────
async function fetchData() {
  if (!props.experimentId || props.objectiveNames.length < 2) return;

  state.value = "loading";
  errorMsg.value = "";

  const minimize = props.objectiveGoals.map((g) => (g === "min" ? "true" : "false"));
  const params = new URLSearchParams();
  props.objectiveNames.forEach((o) => params.append("objectives", o));
  minimize.forEach((m) => params.append("minimize", m));

  try {
    const { data: res } = await client.get<HvGdData>(
      `/experiments/${props.experimentId}/hv-gd?${params.toString()}`,
    );
    if (!res.generations || res.generations.length === 0) {
      state.value = "empty";
      return;
    }
    data.value = res;
    state.value = "ready";
  } catch (e) {
    errorMsg.value = e instanceof Error ? e.message : String(e);
    state.value = "error";
  }
}

// ── chart init & rendering ──────────────────────────────────────────────────

function buildHvOption(d: HvGdData, dark: boolean, mode: HvMode): EChartsOption {
  const c = chartPalette(dark);
  const xLabels = d.generations.map((g) => `Gen ${g}`);
  const cumulative = mode === "cumulative";
  const series = cumulative ? d.hv_cumulative : d.hv;
  const axisLabel = cumulative ? "HV (cumulative)" : "HV";
  const seriesName = cumulative ? "Cumulative hypervolume" : "Hypervolume";

  return {
    backgroundColor: c.bg,
    tooltip: {
      trigger: "axis",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params) => {
        const list = params as Array<DefaultLabelFormatterCallbackParams & { axisValueLabel?: string }>;
        const p = list[0];
        if (!p) return "";
        return `${p.axisValueLabel ?? p.name}<br/><b>${axisLabel}</b>: ${(p.value as number).toExponential(3)}`;
      },
    },
    grid: { top: 30, right: 20, bottom: 40, left: 60, containLabel: false },
    xAxis: {
      type: "category",
      data: xLabels,
      axisLine: { lineStyle: { color: c.grid } },
      axisLabel: { color: c.muted, fontSize: 11 },
      axisTick: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: "value",
      name: axisLabel,
      nameTextStyle: { color: c.muted, fontSize: 11 },
      axisLabel: {
        color: c.muted,
        fontSize: 10,
        formatter: (v: number) => {
          if (Math.abs(v) >= 1e9) return (v / 1e9).toFixed(1) + "B";
          if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + "M";
          return String(v);
        },
      },
      splitLine: { lineStyle: { color: c.grid, type: "dashed" } },
    },
    series: [
      {
        name: seriesName,
        type: "line",
        data: series,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        itemStyle: { color: c.hv },
        lineStyle: { color: c.hv, width: 2 },
        areaStyle: { color: c.hvArea },
      },
    ],
  };
}

function buildGdOption(d: HvGdData, dark: boolean): EChartsOption {
  const c = chartPalette(dark);
  const xLabels = d.generations.map((g) => `Gen ${g}`);

  return {
    backgroundColor: c.bg,
    tooltip: {
      trigger: "axis",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params) => {
        const list = params as Array<DefaultLabelFormatterCallbackParams & { axisValueLabel?: string }>;
        const p = list[0];
        if (!p) return "";
        const v = p.value as number | null;
        const shown = v === null || v === undefined ? "—" : v.toFixed(4);
        return `${p.axisValueLabel ?? p.name}<br/><b>GD</b>: ${shown}`;
      },
    },
    grid: { top: 30, right: 20, bottom: 40, left: 60, containLabel: false },
    xAxis: {
      type: "category",
      data: xLabels,
      axisLine: { lineStyle: { color: c.grid } },
      axisLabel: { color: c.muted, fontSize: 11 },
      axisTick: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: "value",
      name: "GD",
      nameTextStyle: { color: c.muted, fontSize: 11 },
      axisLabel: { color: c.muted, fontSize: 10 },
      splitLine: { lineStyle: { color: c.grid, type: "dashed" } },
    },
    series: [
      {
        name: "Generational Distance",
        type: "line",
        data: d.gd,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        itemStyle: { color: c.gd },
        lineStyle: { color: c.gd, width: 2 },
        areaStyle: { color: c.gdArea },
      },
    ],
  };
}

function buildIgdOption(d: HvGdData, dark: boolean): EChartsOption {
  const c = chartPalette(dark);
  const xLabels = d.generations.map((g) => `Gen ${g}`);

  // IGD and IGD+ answer the same question and live on the same scale, so they
  // share one panel. GD stays on its own: a population converged onto a single
  // corner of the front scores a near-zero GD and a large IGD, and putting the
  // two on one axis would flatten whichever is smaller into the baseline.
  return {
    backgroundColor: c.bg,
    tooltip: {
      trigger: "axis",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params) => {
        const list = params as Array<DefaultLabelFormatterCallbackParams & { axisValueLabel?: string }>;
        const first = list[0];
        if (!first) return "";
        const rows = list
          .map((p) => {
            const v = p.value as number | null;
            const shown = v === null || v === undefined ? "—" : v.toFixed(4);
            return `<b>${p.seriesName}</b>: ${shown}`;
          })
          .join("<br/>");
        return `${first.axisValueLabel ?? first.name}<br/>${rows}`;
      },
    },
    legend: {
      top: 0,
      right: 0,
      itemWidth: 14,
      itemHeight: 8,
      textStyle: { color: c.muted, fontSize: 10 },
      data: ["IGD", "IGD+"],
    },
    grid: { top: 30, right: 20, bottom: 40, left: 60, containLabel: false },
    xAxis: {
      type: "category",
      data: xLabels,
      axisLine: { lineStyle: { color: c.grid } },
      axisLabel: { color: c.muted, fontSize: 11 },
      axisTick: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: "value",
      name: "IGD",
      nameTextStyle: { color: c.muted, fontSize: 11 },
      axisLabel: { color: c.muted, fontSize: 10 },
      splitLine: { lineStyle: { color: c.grid, type: "dashed" } },
    },
    series: [
      {
        name: "IGD",
        type: "line",
        data: d.igd,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        itemStyle: { color: c.igd },
        lineStyle: { color: c.igd, width: 2 },
        areaStyle: { color: c.igdArea },
      },
      {
        // Pareto-compliant variant (Ishibuchi et al., 2015). It bounds IGD from
        // below, so it is drawn as a line only — an area would sit on top of
        // the IGD area and muddy both.
        name: "IGD+",
        type: "line",
        data: d.igd_plus,
        smooth: true,
        symbol: "triangle",
        symbolSize: 6,
        itemStyle: { color: c.igdPlus },
        lineStyle: { color: c.igdPlus, width: 2, type: "dashed" },
      },
    ],
  };
}

function initCharts() {
  if (!hvEl.value || !gdEl.value || !igdEl.value) return;
  hvChart = echarts.init(hvEl.value, null, { renderer: "svg" });
  gdChart = echarts.init(gdEl.value, null, { renderer: "svg" });
  igdChart = echarts.init(igdEl.value, null, { renderer: "svg" });

  ro = new ResizeObserver(() => {
    // Skip collapsed/hidden passes — resizing to 0×0 blanks the chart.
    if (hvEl.value && hvEl.value.clientHeight > 0) hvChart?.resize();
    if (gdEl.value && gdEl.value.clientHeight > 0) gdChart?.resize();
    if (igdEl.value && igdEl.value.clientHeight > 0) igdChart?.resize();
  });
  ro.observe(hvEl.value);
  ro.observe(gdEl.value);
  ro.observe(igdEl.value);
}

function renderCharts() {
  if (!data.value || !hvChart || !gdChart || !igdChart) return;
  const dark = isDark.value;
  hvChart.setOption(buildHvOption(data.value, dark, hvMode.value), true);
  gdChart.setOption(buildGdOption(data.value, dark), true);
  igdChart.setOption(buildIgdOption(data.value, dark), true);
}

function destroyCharts() {
  ro?.disconnect();
  ro = null;
  hvChart?.dispose();
  gdChart?.dispose();
  igdChart?.dispose();
  hvChart = null;
  gdChart = null;
  igdChart = null;
}

// ── lifecycle ────────────────────────────────────────────────────────────────
onMounted(async () => {
  await fetchData();
});

onBeforeUnmount(destroyCharts);

// When data arrives, init + render charts
watch(state, async (s) => {
  if (s !== "ready") return;
  // Wait for DOM update so the chart divs are visible
  await new Promise((r) => setTimeout(r, 0));
  if (!hvChart) initCharts();
  renderCharts();
});

// Re-render on theme change
watch(isDark, () => {
  if (state.value === "ready") renderCharts();
});

// Toggle per-generation ↔ cumulative — reuse the already-fetched data, only the
// HV chart changes so the GD chart is left untouched.
watch(hvMode, (mode) => {
  if (state.value === "ready" && hvChart && data.value) {
    hvChart.setOption(buildHvOption(data.value, isDark.value, mode), true);
  }
});

// Refetch if experiment changes
watch(
  () => props.experimentId,
  () => {
    destroyCharts();
    fetchData();
  },
);
</script>

<style scoped>
.hvgd-root {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  padding: 0 4px;
}

.hvgd-charts {
  display: flex;
  flex: 1;
  gap: 12px;
  min-height: 0;
}

.hvgd-col {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
  min-height: 0;
}

.hvgd-chart {
  flex: 1;
  min-width: 0;
  min-height: 0;
}

/* States which front GD/IGD are measured against — without it the two curves
   are unreadable, since a self-referential run drives both to zero for reasons
   that have nothing to do with solution quality. */
.hvgd-caption {
  margin: 4px 0 0;
  font-size: 11px;
  line-height: 1.4;
  color: var(--color-text-muted);
  text-align: center;
}

.hvgd-caption.is-warning {
  color: var(--color-warning, #b45309);
}

.controls-bar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

/* Toggle sits at the far left; the export button stays flush right. */
.hv-mode-toggle {
  margin-right: auto;
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.mode-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border: none;
  background: var(--color-surface);
  color: var(--color-text-muted);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.mode-btn + .mode-btn {
  border-left: 1px solid var(--color-border);
}

.mode-btn:hover {
  background: var(--color-surface-hover);
  color: var(--color-text);
}

.mode-btn.active {
  background: var(--color-primary);
  color: #fff;
}

.hvgd-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex: 1;
  font-size: 13px;
  color: var(--color-text-muted);
}

.hvgd-error {
  color: #dc2626;
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>
