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
    .filter((g) => g.status === "Done" && g.population.length > 0)
    .sort((a, b) => a.index - b.index),
);

const hasData = computed(() => finishedGens.value.length > 0);

const PALETTE = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

function buildOption() {
  if (!hasData.value || !ready.value) return;

  const names = props.objectiveNames;
  const xData = finishedGens.value.map((g) => `Gen ${g.index}`);

  // Para cada objetivo, coleta o melhor valor por geração (mínimo da população)
  const rawBest: (number | null)[][] = names.map((_, idx) =>
    finishedGens.value.map((gen) => {
      const valid = gen.population
        .map((ind) => ind.objectives[idx])
        .filter((v) => v !== undefined && !isNaN(v));
      return valid.length > 0 ? Math.min(...valid) : null;
    }),
  );

  // Min/max globais por objetivo para normalização [0, 1]
  const globalMin: number[] = names.map((_, idx) => {
    const vals = rawBest[idx].filter((v): v is number => v !== null);
    return vals.length > 0 ? Math.min(...vals) : 0;
  });
  const globalMax: number[] = names.map((_, idx) => {
    const vals = rawBest[idx].filter((v): v is number => v !== null);
    return vals.length > 0 ? Math.max(...vals) : 1;
  });

  function normalize(v: number | null, idx: number): number | null {
    if (v === null) return null;
    const range = globalMax[idx] - globalMin[idx];
    if (range === 0) return 0;
    return (v - globalMin[idx]) / range;
  }

  const series = names.map((name, idx) => ({
    name,
    type: "line" as const,
    data: rawBest[idx].map((v) => {
      const norm = normalize(v, idx);
      if (norm === null) return null;
      // Tooltip mostra valor original; nome inclui range para contexto
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
      formatter: (params: { seriesName: string; value: number; data: { raw: number } | null; axisValue: string }[]) => {
        const header = `<b>${params[0]?.axisValue}</b><br>`;
        const rows = params
          .map((p) => {
            const raw = p.data?.raw;
            const rawStr = raw !== undefined ? raw.toFixed(4) : "—";
            const normStr = p.value !== undefined ? (p.value * 100).toFixed(1) + "%" : "—";
            return `${p.seriesName}: <b>${rawStr}</b> <span style="color:#9ca3af">(${normStr})</span>`;
          })
          .join("<br>");
        return header + rows;
      },
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
