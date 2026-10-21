<template>
  <div class="usl-chart">
    <div class="usl-chart__bar">
      <ChartExportButton @click="exportPng" />
    </div>
    <div ref="el" class="usl-chart__canvas" role="img" :aria-label="ariaLabel" />
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
import ChartExportButton from "../charts/ChartExportButton.vue";

/**
 * Throughput against concurrency: what was measured, the USL fitted to it,
 * linear scaling for reference, and the peak when the curve has one.
 *
 * The model is drawn past the largest measured level and that part is
 * shaded, because a curve is worth showing where nothing was measured only
 * if the chart says that it is a prediction.
 */
const props = defineProps<{
  data: ChartData;
  /** Used for the export file name. */
  name?: string;
}>();

const el = ref<HTMLElement | null>(null);
const chart = useEChart(el);
const { isDark } = useTheme();

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
    grid: { left: 64, right: 24, top: 40, bottom: 56 },
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

function render() {
  chart.setOption(option(props.data, isDark.value));
}

watch(() => props.data, render, { deep: true, immediate: true });
watch(isDark, render);
watch(chart.ready, (ready) => ready && render());

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
}

.usl-chart__canvas {
  width: 100%;
  height: 380px;
}

@media (max-width: 640px) {
  .usl-chart__canvas {
    height: 300px;
  }
}
</style>
