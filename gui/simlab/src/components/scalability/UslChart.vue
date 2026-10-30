<template>
  <div class="usl-chart">
    <div class="usl-chart__bar">
      <span class="usl-chart__hint">
        Scroll to zoom · drag to pan · drag the bar below the chart to resize
      </span>
      <button
        v-if="isZoomed"
        type="button"
        class="reset-zoom-btn"
        title="Reset zoom to full view"
        @click="resetZoom"
      >
        ↺ Reset zoom
      </button>
      <ChartExportButton @click="exportPng" />
    </div>
    <div
      ref="el"
      class="usl-chart__canvas"
      :style="{ height: `${height}px` }"
      role="img"
      :aria-label="ariaLabel"
    />
    <ResizeHandle v-model="height" :label="`${name || 'USL'} chart`" :min="220" :max="1000" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { EChartsOption, LineSeriesOption } from "echarts";
import { useEChart } from "../../composables/useEChart";
import { useTheme } from "../../composables/useTheme";
import { chartExportBackground, chartPalette } from "../../services/chartTheme";
import { chartExportFilename } from "../../utils/chartExport";
import { formatThroughput, yAxisMax, type ChartData } from "../../services/usl";
import { usePersistedHeight } from "../../composables/usePersistedHeight";
import ChartExportButton from "../charts/ChartExportButton.vue";
import ResizeHandle from "../common/ResizeHandle.vue";

/**
 * Throughput against concurrency: what was measured, the USL fitted to it,
 * linear scaling for reference, and the peak when the curve has one.
 *
 * The model is drawn past the largest measured level and that part is
 * shaded, because a curve is worth showing where nothing was measured only
 * if the chart says that it is a prediction.
 *
 * Zoomable — the wheel zooms both axes, a drag pans, the slider under the
 * chart picks the range of N — because the interesting part is often a few
 * levels around the knee, squeezed between a long extrapolation and the
 * origin. Resizable by the bar underneath, remembered for the session.
 */
const props = defineProps<{
  data: ChartData;
  /** Used for the export file name. A new name is a different chart, so it
   *  starts zoomed out again. */
  name?: string;
  /** Which remembered height to use: charts with different roles keep
   *  their own. */
  heightKey?: string;
}>();

const el = ref<HTMLElement | null>(null);
const chart = useEChart(el);
const { isDark } = useTheme();
const height = usePersistedHeight(
  `usl:${props.heightKey ?? "pool"}`,
  typeof window !== "undefined" && window.innerWidth < 640 ? 300 : 380,
);
const isZoomed = ref(false);

const ariaLabel = computed(() => {
  const peak = props.data.peak;
  return (
    `Throughput against concurrency: ${props.data.measured.length} measured waves` +
    (peak
      ? `, model peaks at ${peak.n} concurrent simulations (${formatThroughput(peak.x)}/min)` +
        (peak.withinTested ? "" : ", extrapolated")
      : "")
  );
});

function option(data: ChartData, dark: boolean): EChartsOption {
  const c = chartPalette(dark);
  const xMax = Math.max(data.testedMax, ...data.model.map((p) => p[0]), 1);
  const peak = data.peak;
  // The x-axis stops a few times past what was measured: a peak twenty times
  // further out is a long extrapolation, and stretching the axis to it would
  // flatten everything that was actually measured into the left edge. The
  // chart still has to say the peak exists, or the tile above contradicts it.
  const peakOffChart = peak !== null && peak.n > xMax;

  // Typed here so the mark pairs below are read as the tuples ECharts
  // expects rather than as arrays of a union.
  const model: LineSeriesOption = {
    name: "USL model",
    type: "line",
    data: data.model,
    showSymbol: false,
    lineStyle: { width: 2.5, color: c.hv },
    itemStyle: { color: c.hv },
    z: 3,
    // Beyond the largest level that was run, this is a prediction.
    markArea:
      xMax > data.testedMax || peakOffChart
        ? {
            silent: true,
            itemStyle: { color: dark ? "rgba(166,173,200,0.06)" : "rgba(100,116,139,0.06)" },
            label: {
              show: true,
              color: c.muted,
              fontSize: 11,
              position: "insideTop",
              formatter: peakOffChart
                ? `extrapolated · peak at N* ≈ ${peak!.n}, past this chart →`
                : "extrapolated",
            },
            data: [[{ name: "extrapolated", xAxis: data.testedMax }, { xAxis: xMax }]],
          }
        : undefined,
    markPoint:
      peak && !peakOffChart
        ? {
            symbol: "pin",
            symbolSize: 46,
            itemStyle: { color: peak.withinTested ? c.gd : c.muted },
            label: { color: "#fff", fontSize: 11, fontWeight: 600, formatter: `N*=${peak.n}` },
            data: [{ name: "peak", coord: [peak.n, peak.x], value: peak.n }],
          }
        : undefined,
    markLine:
      peak && !peakOffChart
        ? {
            silent: true,
            symbol: "none",
            lineStyle: { type: "dotted", color: peak.withinTested ? c.gd : c.muted },
            label: {
              color: c.muted,
              fontSize: 11,
              formatter: peak.withinTested ? "peak" : "peak (extrapolated)",
            },
            data: [{ xAxis: peak.n }],
          }
        : undefined,
  };

  return {
    backgroundColor: c.bg,
    animation: false,
    // Room under the axis name for the zoom slider.
    grid: { left: 64, right: 24, top: 40, bottom: 88 },
    dataZoom: [
      // Wheel zooms, drag pans. `filterMode: none` keeps the lines whole
      // instead of cutting them at the edge of the window.
      { type: "inside", xAxisIndex: 0, yAxisIndex: 0, filterMode: "none" },
      {
        type: "slider",
        xAxisIndex: 0,
        filterMode: "none",
        height: 18,
        bottom: 8,
        borderColor: c.grid,
        textStyle: { color: c.muted, fontSize: 11 },
        labelFormatter: (value: number) => String(Math.round(value)),
      },
    ],
    legend: {
      type: "scroll",
      top: 4,
      textStyle: { color: c.text, fontSize: 12 },
      data: ["Measured", "Mean per level", "USL model", "Linear (λN)"],
    },
    tooltip: {
      trigger: "item",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (params: unknown) => {
        const p = params as { seriesName: string; value: [number, number] };
        return `${p.seriesName}<br/>N = ${p.value[0]}: <b>${formatThroughput(p.value[1])}</b> sim/min`;
      },
    },
    xAxis: {
      type: "value",
      name: "Concurrent simulations (N)",
      nameLocation: "middle",
      nameGap: 32,
      nameTextStyle: { color: c.text, fontSize: 12 },
      min: 0,
      max: xMax,
      minInterval: 1,
      axisLabel: { color: c.muted, hideOverlap: true },
      axisLine: { lineStyle: { color: c.grid } },
      splitLine: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: "value",
      name: "Throughput (simulations / min)",
      nameLocation: "middle",
      nameGap: 46,
      nameTextStyle: { color: c.text, fontSize: 12 },
      min: 0,
      max: yAxisMax(data),
      axisLabel: { color: c.muted },
      axisLine: { lineStyle: { color: c.grid } },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: [
      {
        name: "Linear (λN)",
        type: "line",
        data: data.ideal,
        showSymbol: false,
        lineStyle: { type: "dashed", width: 1.5, color: c.muted },
        itemStyle: { color: c.muted },
        z: 1,
      },
      model,
      {
        // Every repetition, not only the mean: the spread is part of the
        // result, and hiding it would make a loose fit look tight.
        name: "Measured",
        type: "scatter",
        data: data.measured,
        symbolSize: 7,
        itemStyle: { color: c.colorB, opacity: 0.55 },
        z: 4,
      },
      {
        name: "Mean per level",
        type: "scatter",
        data: data.means.map((m) => [m.n, m.mean]),
        symbol: "diamond",
        symbolSize: 12,
        itemStyle: { color: c.colorB, borderColor: c.text, borderWidth: 1 },
        z: 5,
      },
    ],
  };
}

// After the first render, updates replace only the series: everything else
// merges, so the viewer's zoom survives the page's refresh of a running
// benchmark instead of snapping back every few seconds.
let initialized = false;

function render() {
  const next = option(props.data, isDark.value);
  if (!initialized) {
    chart.setOption(next, true);
    initialized = true;
    isZoomed.value = false;
  } else {
    chart.setOption(next, { replaceMerge: ["series"] });
  }
}

function resetZoom() {
  for (const dataZoomIndex of [0, 1]) {
    chart.dispatch({ type: "dataZoom", dataZoomIndex, start: 0, end: 100 });
  }
  isZoomed.value = false;
}

chart.on("datazoom", () => {
  isZoomed.value = true;
});

watch(() => props.data, render, { deep: true, immediate: true });
watch(isDark, render);
watch(
  () => props.name,
  () => {
    initialized = false;
    render();
  },
);
// A new chart instance (its element was recreated) starts from scratch.
watch(chart.ready, (ready) => {
  if (!ready) return;
  initialized = false;
  render();
});

function exportPng() {
  chart.exportImage(chartExportFilename(`usl-${props.name || "benchmark"}`), {
    backgroundColor: chartExportBackground(isDark.value),
  });
}
</script>

<style scoped>
.usl-chart {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.usl-chart__bar {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.usl-chart__hint {
  margin-right: auto;
  font-size: 11px;
  color: var(--color-text-muted);
}

.usl-chart__canvas {
  width: 100%;
  /* height from the :style binding — resizable, bounded by the handle */
}

.reset-zoom-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border: 1px solid var(--color-primary);
  border-radius: var(--radius-sm);
  background: var(--color-primary-light);
  color: var(--color-primary);
  cursor: pointer;
  transition: background 0.15s;
}

.reset-zoom-btn:hover {
  background: var(--color-primary);
  color: #fff;
}
</style>
