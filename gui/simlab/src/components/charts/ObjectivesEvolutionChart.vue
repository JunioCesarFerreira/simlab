<template>
  <div class="chart-wrap">
    <div v-if="!hasData" class="empty">Sem dados de gerações para exibir</div>
    <div v-else ref="chartEl" class="chart" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from "vue";
import { useEChart } from "../../composables/useEChart";
import type { GenerationDto } from "../../types/simlab";

const props = defineProps<{
  generations: GenerationDto[];
  objectiveNames: string[];
}>();

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready } = useEChart(chartEl);

const finishedGens = computed(() =>
  props.generations
    .filter(
      (g) =>
        g.status === "Done" &&
        g.population.length > 0,
    )
    .sort((a, b) => a.index - b.index),
);

const hasData = computed(() => finishedGens.value.length > 0);

const PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

function buildOption() {
  if (!hasData.value || !ready.value) return;

  const names = props.objectiveNames;
  const xData = finishedGens.value.map((g) => `Gen ${g.index}`);

  const series = names.map((name, idx) => {
    const bestValues = finishedGens.value.map((gen) => {
      const values = gen.population.map((ind) => ind.objectives[idx] ?? NaN);
      const valid = values.filter((v) => !isNaN(v));
      return valid.length > 0 ? Math.min(...valid) : null;
    });

    return {
      name,
      type: "line" as const,
      data: bestValues,
      smooth: true,
      symbol: "circle",
      symbolSize: 6,
      itemStyle: { color: PALETTE[idx % PALETTE.length] },
      lineStyle: { color: PALETTE[idx % PALETTE.length], width: 2 },
    };
  });

  setOption({
    tooltip: { trigger: "axis" },
    legend: {
      bottom: 0,
      textStyle: { fontSize: 12 },
    },
    grid: { left: 56, right: 24, top: 24, bottom: 48 },
    xAxis: {
      type: "category",
      data: xData,
      axisLabel: { fontSize: 11 },
    },
    yAxis: {
      type: "value",
      splitLine: { lineStyle: { color: "#f0f0f0" } },
      axisLabel: { fontSize: 11 },
    },
    series,
  });
}

watch([ready, () => props.generations, () => props.objectiveNames], buildOption);
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
