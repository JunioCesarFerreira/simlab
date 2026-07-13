<template>
  <div class="card rm-card">
    <div class="section-title rm-title">
      <span>
        Runtime Metrics
        <span v-if="statusLabel" :class="['rm-status', `rm-status--${metrics.status}`]">
          {{ statusLabel }}
        </span>
      </span>
      <button
        v-if="hasArtifact"
        class="rm-toggle-btn"
        :disabled="state === 'loading'"
        @click="toggleCharts"
      >
        <span v-if="state === 'loading'" class="spinner" />
        {{ chartsOpen ? "Hide charts" : state === "loading" ? "Loading…" : "Show charts" }}
      </button>
    </div>

    <!-- Summary tiles — always visible, fed by the embedded summary block -->
    <div class="rm-summary">
      <div class="rm-tile">
        <div class="rm-tile-label">Duration</div>
        <div class="rm-tile-value">{{ formatDuration(summary.duration_seconds) }}</div>
      </div>
      <div class="rm-tile">
        <div class="rm-tile-label">CPU avg</div>
        <div class="rm-tile-value">{{ formatPercent(summary.cpu?.average_percent) }}</div>
      </div>
      <div class="rm-tile">
        <div class="rm-tile-label">CPU peak</div>
        <div class="rm-tile-value">{{ formatPercent(summary.cpu?.maximum_percent) }}</div>
      </div>
      <div class="rm-tile">
        <div class="rm-tile-label">Memory avg</div>
        <div class="rm-tile-value">{{ formatBytes(summary.memory?.average_bytes) }}</div>
      </div>
      <div class="rm-tile">
        <div class="rm-tile-label">Memory peak</div>
        <div class="rm-tile-value">{{ formatBytes(summary.memory?.maximum_bytes) }}</div>
      </div>
    </div>

    <div v-if="metrics.status === 'collecting'" class="rm-note">
      Collecting telemetry from Prometheus — the summary will appear shortly.
    </div>
    <div v-else-if="metrics.status === 'no_data'" class="rm-note">
      No telemetry samples were captured for this execution window.
    </div>
    <div v-else-if="metrics.status === 'failed'" class="rm-note rm-note--error">
      Telemetry collection failed{{ metrics.error ? `: ${metrics.error}` : "." }}
    </div>

    <!-- Full time series — loaded on demand only -->
    <template v-if="chartsOpen">
      <div v-if="state === 'error'" class="rm-note rm-note--error">{{ errorMsg }}</div>
      <template v-else-if="state === 'ready'">
        <div class="rm-charts">
          <div class="rm-chart-block">
            <div class="rm-chart-title">CPU usage (%)</div>
            <div ref="cpuEl" class="rm-chart" role="img" aria-label="CPU usage over the experiment execution" />
          </div>
          <div class="rm-chart-block">
            <div class="rm-chart-title">Memory usage</div>
            <div ref="memEl" class="rm-chart" role="img" aria-label="Memory usage over the experiment execution" />
          </div>
        </div>
        <div v-if="data?.downsampled" class="rm-note">
          Series downsampled to at most {{ MAX_POINTS }} points per line for display —
          the full-resolution artifact ({{ data.total_samples }} samples) is preserved in GridFS.
        </div>
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { EChartsOption } from "echarts";
import { useEChart } from "../../composables/useEChart";
import { useTheme } from "../../composables/useTheme";
import { chartPalette } from "../../services/chartTheme";
import { getRuntimeMetricsSeries } from "../../api/experiments";
import type {
  RuntimeMetricsDto,
  RuntimeMetricsSeriesDto,
  RuntimeMetricsSeriesResponseDto,
} from "../../types/simlab";

const props = defineProps<{
  experimentId: string;
  metrics: RuntimeMetricsDto;
}>();

const MAX_POINTS = 1000;

const { isDark } = useTheme();

const summary = computed(() => props.metrics.summary ?? {});
const hasArtifact = computed(
  () => props.metrics.status === "completed" && !!props.metrics.artifact,
);

const statusLabel = computed(() => {
  switch (props.metrics.status) {
    case "collecting": return "collecting…";
    case "no_data": return "no data";
    case "failed": return "failed";
    default: return "";
  }
});

// ── on-demand series loading ────────────────────────────────────────────────
type State = "idle" | "loading" | "ready" | "error";
const state = ref<State>("idle");
const errorMsg = ref("");
const chartsOpen = ref(false);
const data = ref<RuntimeMetricsSeriesResponseDto | null>(null);

async function toggleCharts() {
  if (chartsOpen.value) {
    chartsOpen.value = false;
    return;
  }
  if (!data.value) {
    state.value = "loading";
    errorMsg.value = "";
    try {
      data.value = await getRuntimeMetricsSeries(props.experimentId, MAX_POINTS);
      state.value = "ready";
    } catch (e) {
      errorMsg.value = e instanceof Error ? e.message : String(e);
      state.value = "error";
    }
  }
  chartsOpen.value = true;
}

// ── charts ──────────────────────────────────────────────────────────────────
const cpuEl = ref<HTMLElement | null>(null);
const memEl = ref<HTMLElement | null>(null);
const cpuChart = useEChart(cpuEl);
const memChart = useEChart(memEl);

function buildOption(
  series: RuntimeMetricsSeriesDto[],
  valueFormatter: (v: number) => string,
): EChartsOption {
  const c = chartPalette(isDark.value);
  return {
    backgroundColor: c.bg,
    grid: { left: 64, right: 16, top: 12, bottom: 52 },
    tooltip: {
      trigger: "axis",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      valueFormatter: (v) => (typeof v === "number" ? valueFormatter(v) : String(v ?? "")),
    },
    legend: {
      type: "scroll",
      bottom: 0,
      textStyle: { color: c.muted, fontSize: 11 },
      pageTextStyle: { color: c.muted },
    },
    xAxis: {
      type: "time",
      axisLabel: { color: c.muted, fontSize: 11 },
      axisLine: { lineStyle: { color: c.grid } },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      axisLabel: {
        color: c.muted,
        fontSize: 11,
        formatter: (v: number) => valueFormatter(v),
      },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: series.map((s) => ({
      name: s.name,
      type: "line",
      showSymbol: false,
      smooth: false,
      // Whole-stack aggregate gets visual priority over per-container lines
      lineStyle: { width: s.scope === "aggregate" ? 2.5 : 1 },
      emphasis: { focus: "series" },
      z: s.scope === "aggregate" ? 3 : 2,
      data: s.points.map(([ts, v]) => [ts * 1000, v]),
    })),
  };
}

function renderCharts() {
  const all = data.value?.series ?? [];
  const cpu = all.filter((s) => s.metric === "cpu_percent");
  const mem = all.filter((s) => s.metric === "memory_bytes");
  cpuChart.setOption(buildOption(cpu, (v) => `${v.toFixed(1)}%`));
  memChart.setOption(buildOption(mem, (v) => formatBytes(v)));
}

// Containers only exist while chartsOpen && ready; render once both charts
// attach, and re-render on data/theme change.
watch(
  [() => data.value, chartsOpen, isDark, cpuChart.ready, memChart.ready],
  () => {
    if (chartsOpen.value && state.value === "ready" && cpuChart.ready.value) {
      renderCharts();
    }
  },
  { flush: "post" },
);

// ── formatting ──────────────────────────────────────────────────────────────
function formatPercent(v: number | undefined): string {
  return v == null ? "—" : `${v.toFixed(1)}%`;
}

function formatBytes(v: number | undefined): string {
  if (v == null) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let val = v;
  let i = 0;
  while (val >= 1024 && i < units.length - 1) {
    val /= 1024;
    i++;
  }
  return `${val.toFixed(val >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDuration(seconds: number | undefined): string {
  if (seconds == null) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
</script>

<style scoped>
.rm-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.rm-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.rm-status {
  margin-left: 8px;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.rm-status--collecting {
  color: var(--status-running);
  background: rgba(37, 99, 235, 0.1);
  animation: rm-pulse 2s infinite;
}

.rm-status--no_data {
  color: var(--color-text-muted);
  background: var(--color-bg);
}

.rm-status--failed {
  color: var(--status-error);
  background: rgba(220, 38, 38, 0.1);
}

@keyframes rm-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.rm-toggle-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: var(--radius-md);
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  font-size: 12px;
  font-weight: 500;
  color: var(--color-text);
  white-space: nowrap;
  transition: background 0.15s, border-color 0.15s;
}

.rm-toggle-btn:hover:not(:disabled) {
  background: var(--color-bg);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.rm-toggle-btn:disabled { opacity: 0.7; cursor: wait; }

.spinner {
  width: 11px;
  height: 11px;
  border: 2px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: rm-spin 0.7s linear infinite;
  flex-shrink: 0;
}

@keyframes rm-spin { to { transform: rotate(360deg); } }

.rm-summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px;
}

.rm-tile {
  padding: 10px 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
}

.rm-tile-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
}

.rm-tile-value {
  margin-top: 2px;
  font-size: 18px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.rm-note {
  font-size: 12px;
  color: var(--color-text-muted);
  font-style: italic;
}

.rm-note--error {
  color: var(--status-error);
  font-style: normal;
}

.rm-charts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

@media (max-width: 1100px) {
  .rm-charts { grid-template-columns: 1fr; }
}

.rm-chart-block {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.rm-chart-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
}

.rm-chart {
  height: 300px;
  min-width: 0;
}
</style>
