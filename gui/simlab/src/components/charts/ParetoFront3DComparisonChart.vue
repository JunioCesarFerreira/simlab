<template>
  <div class="chart-wrap">
    <div v-if="!hasSufficientData" class="empty">
      <span v-if="!frontA?.length && !frontB?.length">No Pareto front data available</span>
      <span v-else>At least 3 shared objectives are required for 3D visualization</span>
    </div>
    <template v-else>
      <div class="controls-bar">
        <div class="axis-selectors">
          <label>
            X:
            <select v-model="xKey">
              <option v-for="k in keys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
          <label>
            Y:
            <select v-model="yKey">
              <option v-for="k in keys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
          <label>
            Z:
            <select v-model="zKey">
              <option v-for="k in keys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
        </div>
        <button
          :class="['pan-btn', { active: panMode }]"
          :title="panMode ? 'Switch back to rotate mode (left drag rotates)' : 'Switch to pan mode (left drag moves the scene)'"
          @click="panMode = !panMode"
        >
          ✋ {{ panMode ? 'Panning' : 'Pan' }}
        </button>
      </div>
      <div ref="chartEl" :class="['chart', { 'chart--pan': panMode }]" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import * as echarts from 'echarts';
// Side-effect import registers scatter3D, grid3D, xAxis3D, yAxis3D, zAxis3D
import 'echarts-gl';
import { useTheme } from '../../composables/useTheme';
import type { ParetoFrontItemDto, ObjectiveItem } from '../../types/simlab';

const props = defineProps<{
  frontA: ParetoFrontItemDto[];
  frontB: ParetoFrontItemDto[];
  nameA: string;
  nameB: string;
  objectives: ObjectiveItem[];
}>();

const { isDark } = useTheme();

// ── Chart lifecycle ──────────────────────────────────────────────────────────

const chartEl = ref<HTMLElement | null>(null);
let chart: echarts.ECharts | null = null;
let ro: ResizeObserver | null = null;
// Preserved across re-renders so the user's camera position is not reset
let cameraInitialized = false;

onMounted(() => {
  if (!chartEl.value) return;
  // echarts-gl requires canvas (WebGL) renderer — never use svg here
  chart = echarts.init(chartEl.value);
  ro = new ResizeObserver(() => chart?.resize());
  ro.observe(chartEl.value);
  buildOption();
});

onBeforeUnmount(() => {
  ro?.disconnect();
  ro = null;
  chart?.dispose();
  chart = null;
  cameraInitialized = false;
});

// ── Pan mode ─────────────────────────────────────────────────────────────────

const panMode = ref(false);

// ── Axis state ───────────────────────────────────────────────────────────────

const keys = computed(() => props.objectives.map(o => o.metric_name));

const hasSufficientData = computed(() =>
  (props.frontA.length > 0 || props.frontB.length > 0) && keys.value.length >= 3,
);

const xKey = ref('');
const yKey = ref('');
const zKey = ref('');

watch(
  keys,
  (k) => {
    if (k.length >= 3) {
      if (!xKey.value || !k.includes(xKey.value)) xKey.value = k[0] ?? '';
      if (!yKey.value || !k.includes(yKey.value)) yKey.value = k[1] ?? '';
      if (!zKey.value || !k.includes(zKey.value)) zKey.value = k[2] ?? '';
    }
  },
  { immediate: true },
);

// ── Data types ───────────────────────────────────────────────────────────────

interface Point3D {
  value: [number, number, number];
  allObjectives: Record<string, number>;
}

function toPoints(front: ParetoFrontItemDto[]): Point3D[] {
  return front.flatMap((item) => {
    const x = item.objectives[xKey.value];
    const y = item.objectives[yKey.value];
    const z = item.objectives[zKey.value];
    if (x === undefined || y === undefined || z === undefined) return [];
    if (isNaN(x) || isNaN(y) || isNaN(z)) return [];
    return [{ value: [x, y, z] as [number, number, number], allObjectives: item.objectives }];
  });
}

// ── Tooltip ──────────────────────────────────────────────────────────────────

function formatTooltip(params: unknown): string {
  const p = params as { seriesName: string; data: Point3D };
  const dark = isDark.value;
  const mutedColor = dark ? '#7f849c' : '#6b7280';
  const rows = props.objectives
    .map((o) =>
      `<tr><td style="color:${mutedColor};padding-right:12px">${o.metric_name}</td>` +
      `<td style="font-weight:600">${(p.data.allObjectives[o.metric_name] ?? 0).toFixed(6)}</td></tr>`,
    )
    .join('');
  return (
    `<div style="font-size:12px;font-weight:600;margin-bottom:4px">${p.seriesName}</div>` +
    `<table style="font-size:12px;border-spacing:0">${rows}</table>`
  );
}

// ── Option builder ────────────────────────────────────────────────────────────

function buildOption() {
  if (!chart || !hasSufficientData.value) return;

  const dark = isDark.value;
  const colorA = dark ? '#89b4fa' : '#3b82f6';
  const colorB = dark ? '#fab387' : '#f97316';
  const axisColor  = dark ? '#45475a' : '#c0cad8';
  const labelColor = dark ? '#7f849c' : '#64748b';
  const nameColor  = dark ? '#a6adc8' : '#374151';
  const splitColor = dark ? '#313244' : '#dde3eb';
  const bgColor    = dark ? '#181825' : '#ffffff';

  const xObj = props.objectives.find((o) => o.metric_name === xKey.value);
  const yObj = props.objectives.find((o) => o.metric_name === yKey.value);
  const zObj = props.objectives.find((o) => o.metric_name === zKey.value);
  const axisLabel = (o: ObjectiveItem | undefined) =>
    o ? `${o.metric_name} (${o.goal === 'min' ? '↓' : '↑'})` : '';

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const axisConfig = (name: string): any => ({
    name,
    nameTextStyle: { color: nameColor, fontSize: 12, fontWeight: 600 },
    type: 'value',
    axisLine: { lineStyle: { color: axisColor, width: 1.5 } },
    axisTick: { lineStyle: { color: axisColor } },
    axisLabel: { color: labelColor, fontSize: 11 },
    splitLine: { lineStyle: { color: splitColor, opacity: 0.6 } },
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const option: any = {
    backgroundColor: bgColor,
    legend: {
      bottom: 4,
      data: [props.nameA, props.nameB],
      textStyle: { fontSize: 12, color: labelColor },
      inactiveColor: dark ? '#45475a' : '#d1d5db',
    },
    tooltip: {
      trigger: 'item',
      formatter: formatTooltip,
      padding: [8, 12],
      backgroundColor: dark ? '#1e1e2e' : '#ffffff',
      borderColor: dark ? '#313244' : '#e5e7eb',
      textStyle: { color: dark ? '#cdd6f4' : '#111827', fontSize: 12 },
      extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.15);',
    },
    grid3D: {
      show: true,
      boxWidth: 100,
      boxHeight: 100,
      boxDepth: 100,
      axisLine: { lineStyle: { color: axisColor, width: 1.5 } },
      axisPointer: {
        show: true,
        lineStyle: { color: dark ? '#89b4fa' : '#3b82f6', width: 1.5, opacity: 0.7 },
      },
      splitLine: {
        show: true,
        lineStyle: { color: splitColor, opacity: 0.5, width: 1 },
      },
      splitArea: { show: false },
      light: {
        main: {
          intensity: dark ? 0.7 : 1.0,
          shadow: dark,
          shadowQuality: 'medium',
        },
        ambient: { intensity: dark ? 0.4 : 0.8 },
      },
      environment: bgColor,
      viewControl: {
        autoRotate: false,
        rotateMouseButton: panMode.value ? 'middle' : 'left',
        panMouseButton:    panMode.value ? 'left'   : 'middle',
        rotateSensitivity: 1.5,
        zoomSensitivity: 1.5,
        panSensitivity: 1,
        minDistance: 5,
        maxDistance: 2000,
        // Initial camera included only on first render; omitting preserves user state
        ...(!cameraInitialized && { distance: 220, alpha: 20, beta: 40 }),
      },
      postEffect: {
        enable: dark,
        SSAO: { enable: dark, radius: 2, intensity: dark ? 1.0 : 0, quality: 'medium' },
      },
    },
    xAxis3D: axisConfig(axisLabel(xObj)),
    yAxis3D: axisConfig(axisLabel(yObj)),
    zAxis3D: axisConfig(axisLabel(zObj)),
    series: [
      {
        name: props.nameA,
        type: 'scatter3D',
        data: toPoints(props.frontA),
        symbolSize: 7,
        itemStyle: { color: colorA, opacity: 0.88 },
        emphasis: { itemStyle: { color: colorA, opacity: 1 } },
      },
      {
        name: props.nameB,
        type: 'scatter3D',
        data: toPoints(props.frontB),
        symbolSize: 7,
        itemStyle: { color: colorB, opacity: 0.88 },
        emphasis: { itemStyle: { color: colorB, opacity: 1 } },
      },
    ],
  };

  if (!cameraInitialized) {
    chart.setOption(option, true);
    cameraInitialized = true;
  } else {
    // replaceMerge keeps camera state while updating series and theme properties
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (chart as any).setOption(option, { replaceMerge: ['series'] });
  }
}

// ── Reactivity ────────────────────────────────────────────────────────────────

watch(
  [
    () => props.frontA,
    () => props.frontB,
    () => props.nameA,
    () => props.nameB,
    () => props.objectives,
    xKey, yKey, zKey,
    panMode,
    isDark,
  ],
  () => buildOption(),
  { deep: true },
);
</script>

<style scoped>
.chart-wrap {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart {
  flex: 1;
  min-height: 280px;
}

.chart--pan {
  cursor: grab;
}

.chart--pan:active {
  cursor: grabbing;
}

.empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
  min-height: 200px;
}

.controls-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.axis-selectors {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.axis-selectors select {
  margin-left: 6px;
  font-size: 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 2px 6px;
  background: var(--color-surface);
  color: var(--color-text);
}

.pan-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text-muted);
  transition: background 0.15s, border-color 0.15s, color 0.15s;
  cursor: pointer;
}

.pan-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

.pan-btn.active {
  background: #ede9fe;
  border-color: #a78bfa;
  color: #5b21b6;
}
</style>
