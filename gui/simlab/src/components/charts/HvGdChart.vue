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
    <div v-else class="hvgd-body">
      <div class="population-bar">
        <span class="population-label">Measured set</span>
        <div class="population-toggle" role="group" aria-label="Measured population">
          <button
            v-for="opt in POPULATION_OPTIONS"
            :key="opt.value"
            type="button"
            :class="['mode-btn', { active: population === opt.value }]"
            :aria-pressed="population === opt.value"
            :title="opt.hint"
            @click="population = opt.value"
          >
            {{ opt.label }}
          </button>
        </div>
      </div>
      <div class="hvgd-charts">
      <div class="hvgd-col">
        <div class="controls-bar">
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
    </div>
    <p
      v-if="state === 'ready'"
      class="hvgd-caption"
      :class="{ 'is-warning': selfReferential || populationFallback }"
    >
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

// Which set each generation is measured on. This used to be an HV-only
// "per generation / cumulative" toggle, which left GD and IGD on the offspring
// whatever the user picked. All three indicators now follow one selector, and
// the backend computes it — the survivor set cannot be derived client-side.
type Population = "survivors" | "offspring" | "archive";
const POPULATION_OPTIONS: { value: Population; label: string; hint: string }[] = [
  {
    value: "survivors",
    label: "Survivors",
    hint: "The population environmental selection kept (P_t) — what the search carries forward",
  },
  {
    value: "offspring",
    label: "Offspring",
    hint: "Only the children evaluated in that generation (Q_t) — swings with each batch",
  },
  {
    value: "archive",
    label: "Archive",
    hint: "Best-so-far: the non-dominated set of everything evaluated up to that generation",
  },
];
const population = ref<Population>("survivors");

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
  population: Population | null;
  population_source: Population | null;
  gd_method: "analytical" | "reference_front" | null;
  gd_formula: string | null;
  normalization: string | null;
}
const data = ref<HvGdData | null>(null);

const measuredSet = computed<Population>(
  () => data.value?.population_source ?? population.value,
);

const measuredSetLabel = computed(
  () => POPULATION_OPTIONS.find((o) => o.value === measuredSet.value)?.label ?? "",
);

const hvAriaLabel = computed(
  () => `Hypervolume per generation chart, measured on the ${measuredSetLabel.value.toLowerCase()}`,
);

// The backend degrades to the offspring for runs recorded before survivor sets
// were persisted. Say so rather than letting the two curves be read as one.
const populationFallback = computed(
  () => data.value != null && data.value.population !== data.value.population_source,
);

// Against the run's own final front, GD and IGD measure progress towards this
// run's own result, not convergence to the real optimum, and so cannot be
// compared across runs. They do NOT reach zero on the last generation: the
// engine builds the stored front from the merged pool (surviving parents ∪ last
// offspring), while a generation document records only that generation's
// offspring — so each set holds points the other lacks.
const selfReferential = computed(() => data.value?.reference === "final_front");

const referenceCaption = computed(() => {
  const d = data.value;
  if (!d) return "";
  const scale = d.normalized
    ? "normalized by the reference front's ideal-nadir range"
    : "in raw objective units";
  const size = `${d.reference_size} point${d.reference_size === 1 ? "" : "s"}`;
  const measured = populationFallback.value
    ? "Measured on the offspring (Q_t): this run predates persisted survivor sets, so the population kept by environmental selection cannot be recovered."
    : `Measured on the ${measuredSetLabel.value.toLowerCase()}.`;
  const reference = selfReferential.value
    ? `GD / IGD reference: this run's own final Pareto front (${size}) — self-referential, so these measure progress towards this run's own result, not convergence to the true optimum, and are not comparable across runs. Distances ${scale}.`
    : `IGD / IGD+ reference: the benchmark's analytical true front (${size}). Distances ${scale}.`;
  // A sampled reference cannot measure GD below its own fill distance — points
  // exactly on the DTLZ2 front score 0.19 at M=6 against 500 reference points.
  // Say when GD escaped that, since it makes the two curves read differently.
  const gd = d.gd_method === "analytical"
    ? " GD is the exact distance to the true front, free of the reference front's discretisation error; IGD and IGD+ still carry it."
    : "";
  return `${measured} ${reference}${gd}`;
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
  params.append("population", population.value);

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

function buildHvOption(d: HvGdData, dark: boolean): EChartsOption {
  const c = chartPalette(dark);
  const xLabels = d.generations.map((g) => `Gen ${g}`);
  const label = POPULATION_OPTIONS.find((o) => o.value === d.population_source)?.label ?? "";
  const series = d.hv;
  const axisLabel = "HV";
  const seriesName = `Hypervolume (${label.toLowerCase()})`;

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
  hvChart.setOption(buildHvOption(data.value, dark), true);
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

// Switching the measured set refetches: GD / IGD over the survivors or the
// archive cannot be recomputed from the offspring series already in hand.
watch(population, () => {
  if (state.value === "ready" || state.value === "empty") fetchData();
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

.hvgd-body {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  gap: 6px;
}

.hvgd-charts {
  display: flex;
  flex: 1;
  gap: 12px;
  min-height: 0;
}

/* The measured set drives all three charts, so it sits above them rather than
   inside the hypervolume column. */
.population-bar {
  display: flex;
  align-items: center;
  gap: 8px;
}

.population-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-muted);
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

.population-toggle {
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
