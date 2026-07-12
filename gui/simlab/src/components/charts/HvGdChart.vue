<template>
  <div class="hvgd-root">
    <div v-if="state === 'loading'" class="hvgd-placeholder">
      <span class="spinner" />
      Computing HV &amp; GD…
    </div>
    <div v-else-if="state === 'error'" class="hvgd-placeholder hvgd-error">
      {{ errorMsg }}
    </div>
    <div v-else-if="state === 'empty'" class="hvgd-placeholder">
      No reference front available yet.
    </div>
    <div v-else class="hvgd-charts">
      <div ref="hvEl" class="hvgd-chart" />
      <div ref="gdEl" class="hvgd-chart" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onBeforeUnmount } from "vue";
import * as echarts from "echarts";
import { useTheme } from "../../composables/useTheme";
import client from "../../api/client";

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

interface HvGdData {
  generations: number[];
  hv: number[];
  gd: number[];
  worst_point: Record<string, number>;
}
const data = ref<HvGdData | null>(null);

// ── chart instances ─────────────────────────────────────────────────────────
const hvEl = ref<HTMLElement | null>(null);
const gdEl = ref<HTMLElement | null>(null);
let hvChart: echarts.ECharts | null = null;
let gdChart: echarts.ECharts | null = null;
let ro: ResizeObserver | null = null;

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
function palette(dark: boolean) {
  return {
    bg: "transparent",
    text: dark ? "#cdd6f4" : "#334155",
    muted: dark ? "#6c7086" : "#94a3b8",
    grid: dark ? "#313244" : "#e2e8f0",
    hv: dark ? "#89b4fa" : "#2563eb",
    gd: dark ? "#f38ba8" : "#dc2626",
    hvArea: dark ? "rgba(137,180,250,0.12)" : "rgba(37,99,235,0.08)",
    gdArea: dark ? "rgba(243,139,168,0.12)" : "rgba(220,38,38,0.08)",
    tooltip: dark ? "#1e1e2e" : "#ffffff",
    tooltipBorder: dark ? "#313244" : "#e2e8f0",
  };
}

function buildHvOption(d: HvGdData, dark: boolean): echarts.EChartsOption {
  const c = palette(dark);
  const xLabels = d.generations.map((g) => `Gen ${g}`);

  return {
    backgroundColor: c.bg,
    tooltip: {
      trigger: "axis",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params) => {
        const list = params as Array<echarts.DefaultLabelFormatterCallbackParams & { axisValueLabel?: string }>;
        const p = list[0];
        if (!p) return "";
        return `${p.axisValueLabel ?? p.name}<br/><b>HV</b>: ${(p.value as number).toExponential(3)}`;
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
      name: "HV",
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
        name: "Hypervolume",
        type: "line",
        data: d.hv,
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

function buildGdOption(d: HvGdData, dark: boolean): echarts.EChartsOption {
  const c = palette(dark);
  const xLabels = d.generations.map((g) => `Gen ${g}`);

  return {
    backgroundColor: c.bg,
    tooltip: {
      trigger: "axis",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params) => {
        const list = params as Array<echarts.DefaultLabelFormatterCallbackParams & { axisValueLabel?: string }>;
        const p = list[0];
        if (!p) return "";
        const v = p.value as number;
        return `${p.axisValueLabel ?? p.name}<br/><b>GD</b>: ${v.toFixed(4)}`;
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

function initCharts() {
  if (!hvEl.value || !gdEl.value) return;
  hvChart = echarts.init(hvEl.value, null, { renderer: "svg" });
  gdChart = echarts.init(gdEl.value, null, { renderer: "svg" });

  ro = new ResizeObserver(() => {
    // Skip collapsed/hidden passes — resizing to 0×0 blanks the chart.
    if (hvEl.value && hvEl.value.clientHeight > 0) hvChart?.resize();
    if (gdEl.value && gdEl.value.clientHeight > 0) gdChart?.resize();
  });
  ro.observe(hvEl.value);
  ro.observe(gdEl.value);
}

function renderCharts() {
  if (!data.value || !hvChart || !gdChart) return;
  const dark = isDark.value;
  hvChart.setOption(buildHvOption(data.value, dark), true);
  gdChart.setOption(buildGdOption(data.value, dark), true);
}

function destroyCharts() {
  ro?.disconnect();
  ro = null;
  hvChart?.dispose();
  gdChart?.dispose();
  hvChart = null;
  gdChart = null;
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

.hvgd-chart {
  flex: 1;
  min-width: 0;
  min-height: 0;
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
