<template>
  <div class="study-chart">
    <div class="study-chart__bar">
      <span class="study-chart__title">{{ title }}</span>
      <ChartExportButton @click="exportPng" />
    </div>
    <div
      ref="el"
      class="study-chart__canvas"
      :style="{ height: `${height}px` }"
      role="img"
      :aria-label="title"
    />
    <ResizeHandle v-model="height" :label="`${title} chart`" :min="200" :max="900" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import type { EChartsOption } from "echarts";
import { useEChart } from "../../composables/useEChart";
import { usePersistedHeight } from "../../composables/usePersistedHeight";
import { useTheme } from "../../composables/useTheme";
import { chartExportBackground, chartPalette } from "../../services/chartTheme";
import { chartExportFilename } from "../../utils/chartExport";
import ChartExportButton from "../charts/ChartExportButton.vue";
import ResizeHandle from "../common/ResizeHandle.vue";

/**
 * One chart of a seed study. The options are built in
 * `services/seedStudyCharts.ts`; this draws them, follows the theme, resizes
 * and exports, the same way as the other charts.
 */
type Palette = ReturnType<typeof chartPalette>;

const props = defineProps<{
  title: string;
  /** Used for the export file name. */
  name: string;
  /** Called with the palette, so a change of theme redraws the chart. */
  build: (c: Palette) => EChartsOption;
}>();

const el = ref<HTMLElement | null>(null);
const chart = useEChart(el);
const { isDark } = useTheme();
const height = usePersistedHeight("seed-study", 320);
const option = computed(() => props.build(chartPalette(isDark.value)));

watch(
  [option, chart.ready],
  ([next, ready]) => {
    if (ready) chart.setOption(next, true);
  },
  { immediate: true },
);

function exportPng() {
  chart.exportImage(chartExportFilename(`seed-study-${props.name}`), {
    backgroundColor: chartExportBackground(isDark.value),
  });
}
</script>

<style scoped>
.study-chart {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.study-chart__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.study-chart__title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
}

.study-chart__canvas {
  width: 100%;
}
</style>
