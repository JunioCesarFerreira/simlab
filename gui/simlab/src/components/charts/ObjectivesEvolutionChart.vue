<template>
  <div class="chart-wrap">
    <div v-if="!hasData" class="empty">No generation data to display</div>
    <div v-else ref="chartEl" class="chart" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from "vue";
import type { TopLevelFormatterParams } from "echarts/types/dist/shared";
import { useEChart } from "../../composables/useEChart";
import type { GenerationDto } from "../../types/simlab";

const props = defineProps<{
  generations: GenerationDto[];
  objectiveNames: string[];
  objectiveGoals: string[];
}>();

interface TooltipPoint {
  seriesName?: string;
  value?: number;
  axisValue?: string;
  data?: { raw?: number } | null;
}

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready } = useEChart(chartEl);

const finishedGens = computed(() =>
  props.generations
    .filter((g) => g.status === "Done" && g.population.length > 0)
    .sort((a, b) => a.index - b.index),
);

const hasData = computed(() => finishedGens.value.length > 0);

const PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

function isMaximizationGoal(goal: string | undefined): boolean {
  return goal?.toLowerCase() === "max";
}

function formatTooltip(params: TopLevelFormatterParams): string {
  const points = (Array.isArray(params) ? params : [params]) as TooltipPoint[];
  const header = `<b>${points[0]?.axisValue ?? ""}</b><br>`;
  const rows = points
    .map((p) => {
      const raw = p.data?.raw;
      const rawStr = raw !== undefined ? raw.toFixed(4) : "—";
      const normStr = p.value !== undefined ? `${(p.value * 100).toFixed(1)}%` : "—";
      return `${p.seriesName ?? ""}: <b>${rawStr}</b> <span style="color:#9ca3af">(${normStr})</span>`;
    })
    .join("<br>");

  return header + rows;
}

function buildOption() {
  if (!hasData.value || !ready.value) return;

  const names = props.objectiveNames;
  const goals = props.objectiveGoals;
  const xData = finishedGens.value.map((g) => `Gen ${g.index}`);

  const rawBest: (number | null)[][] = names.map((_, idx) =>
    finishedGens.value.map((gen) => {
      const valid = gen.population
        .map((ind) => ind.objectives[idx])
        .filter((v): v is number => v !== undefined && !isNaN(v));

      if (valid.length === 0) return null;

      return isMaximizationGoal(goals[idx])
        ? Math.max(...valid)
        : Math.min(...valid);
    }),
  );

  const globalMin = names.map((_, idx) => {
    const values = (rawBest[idx] ?? []).filter((v): v is number => v !== null);
    return values.length > 0 ? Math.min(...values) : 0;
  });

  const globalMax = names.map((_, idx) => {
    const values = (rawBest[idx] ?? []).filter((v): v is number => v !== null);
    return values.length > 0 ? Math.max(...values) : 1;
  });

  function normalize(v: number | null, idx: number): number | null {
    if (v === null) return null;

    const min = globalMin[idx] ?? 0;
    const max = globalMax[idx] ?? 1;
    const range = max - min;

    if (range === 0) return 1;

    if (isMaximizationGoal(goals[idx])) {
      return (v - min) / range;
    }

    return (max - v) / range;
  }

  const series = names.map((name, idx) => ({
    name,
    type: "line" as const,
    data: (rawBest[idx] ?? []).map((v) => {
      const norm = normalize(v, idx);
      if (norm === null) return null;

      return {
        value: norm,
        raw: v,
      };
    }),
    smooth: true,
    symbol: "circle",
    symbolSize: 6,
    itemStyle: { color: PALETTE[idx % PALETTE.length] },
    lineStyle: { color: PALETTE[idx % PALETTE.length], width: 2 },
  }));

  setOption({
    tooltip: {
      trigger: "axis",
      formatter: formatTooltip,
    },
    legend: {
      bottom: 0,
      textStyle: { fontSize: 12 },
    },
    grid: { left: 52, right: 24, top: 24, bottom: 48 },
    xAxis: {
      type: "category",
      data: xData,
      axisLabel: { fontSize: 11 },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: {
        fontSize: 11,
        formatter: (v: number) => `${(v * 100).toFixed(0)}%`,
      },
      splitLine: { lineStyle: { color: "#f0f0f0" } },
    },
    series,
  });
}

watch(
  [ready, () => props.generations, () => props.objectiveNames, () => props.objectiveGoals],
  buildOption,
);

onMounted(buildOption);
</script>

<style scoped>
.chart-wrap {
  width: 100%;
  height: 100%;
}

.chart {
  width: 100%;
  height: 280px;
}

.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 200px;
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
}
</style>
