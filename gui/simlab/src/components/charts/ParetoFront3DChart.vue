<template>
  <div class="chart-wrap">
    <div v-if="!hasSufficientData" class="empty">
      <span v-if="!paretoFront || paretoFront.length === 0">Pareto front not available</span>
      <span v-else>At least 3 objectives are required for 3D visualization</span>
    </div>
    <template v-else>
      <div class="controls-bar">
        <div class="axis-selectors">
          <label>
            X:
            <select v-model="xKey">
              <option v-for="k in availableKeys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
          <label>
            Y:
            <select v-model="yKey">
              <option v-for="k in availableKeys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
          <label>
            Z:
            <select v-model="zKey">
              <option v-for="k in availableKeys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
        </div>

        <div class="pin-controls">
          <button
            :class="['pin-btn', { active: markMode }]"
            :title="markMode ? 'Exit pin mode' : 'Enter pin mode — click a Pareto point to pin it'"
            @click="markMode = !markMode"
          >
            📌 {{ markMode ? 'Pinning…' : 'Pin' }}
          </button>
          <div v-if="markedId" class="marked-badge">
            <span class="marked-id" :title="markedId">📍 {{ markedId.slice(0, 14) }}…</span>
            <button class="clear-pin" title="Clear pin" @click="markedId = ''">✕</button>
          </div>
        </div>
      </div>

      <div ref="chartEl" class="chart" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from "vue";
import * as echarts from "echarts";
// Side-effect import registers scatter3D, grid3D, xAxis3D, yAxis3D, zAxis3D
import "echarts-gl";
import { useTheme } from "../../composables/useTheme";
import type { ParetoFrontItemDto, GenerationDto } from "../../types/simlab";
import { isPenalized } from "../../types/simlab";

const props = defineProps<{
  paretoFront: ParetoFrontItemDto[] | null | undefined;
  generations?: GenerationDto[];
  objectiveNames?: string[];
}>();

const emit = defineEmits<{
  (e: "click-individual", individualId: string): void;
}>();

const { isDark } = useTheme();

// ── Chart lifecycle ─────────────────────────────────────────────────────────

const chartEl = ref<HTMLElement | null>(null);
let chart: echarts.ECharts | null = null;
let ro: ResizeObserver | null = null;

onMounted(() => {
  if (!chartEl.value) return;
  // echarts-gl requires canvas (WebGL) renderer — never use svg here
  chart = echarts.init(chartEl.value);
  ro = new ResizeObserver(() => chart?.resize());
  ro.observe(chartEl.value);
  buildOption();
  chart.on("click", (params: Record<string, unknown>) => {
    const data = params.data as Point3D | undefined;
    if (!data?.individualId) return;
    if (markMode.value) {
      markedId.value = markedId.value === data.individualId ? "" : data.individualId;
    } else {
      emit("click-individual", data.individualId);
    }
  });
});

onBeforeUnmount(() => {
  ro?.disconnect();
  ro = null;
  chart?.dispose();
  chart = null;
});

// ── Data types ───────────────────────────────────────────────────────────────

interface Point3D {
  value: [number, number, number];
  individualId: string;
  allObjectives: number[] | Record<string, number>;
}

// ── Pin state ─────────────────────────────────────────────────────────────────

const markMode = ref(false);
const markedId = ref("");

// ── Axis state ────────────────────────────────────────────────────────────────

const availableKeys = computed<string[]>(() => {
  if (props.objectiveNames && props.objectiveNames.length >= 3) return props.objectiveNames;
  const first = props.paretoFront?.[0];
  return first ? Object.keys(first.objectives) : [];
});

const hasSufficientData = computed(
  () => (props.paretoFront?.length ?? 0) > 0 && availableKeys.value.length >= 3,
);

const xKey = ref("");
const yKey = ref("");
const zKey = ref("");

watch(
  availableKeys,
  (keys) => {
    if (keys.length >= 3) {
      xKey.value = keys[0] ?? "";
      yKey.value = keys[1] ?? "";
      zKey.value = keys[2] ?? "";
    }
  },
  { immediate: true },
);

const xIdx = computed(() => availableKeys.value.indexOf(xKey.value));
const yIdx = computed(() => availableKeys.value.indexOf(yKey.value));
const zIdx = computed(() => availableKeys.value.indexOf(zKey.value));

// ── Chromosome → individual_id lookup ────────────────────────────────────────

function stableStringify(val: unknown): string {
  if (Array.isArray(val)) return `[${val.map(stableStringify).join(",")}]`;
  if (val !== null && typeof val === "object") {
    const entries = Object.keys(val as object)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify((val as Record<string, unknown>)[k])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(val);
}

const chromosomeMap = computed(() => {
  const map = new Map<string, string>();
  for (const gen of props.generations ?? []) {
    for (const ind of gen.population) {
      map.set(stableStringify(ind.chromosome), ind.individual_id);
    }
  }
  return map;
});

// ── Series data ───────────────────────────────────────────────────────────────

const populationData = computed<Point3D[]>(() => {
  const seen = new Set<string>();
  const pts: Point3D[] = [];
  for (const gen of props.generations ?? []) {
    for (const ind of gen.population) {
      if (seen.has(ind.individual_id)) continue;
      seen.add(ind.individual_id);
      if (isPenalized(ind.objectives)) continue;
      const x = ind.objectives[xIdx.value];
      const y = ind.objectives[yIdx.value];
      const z = ind.objectives[zIdx.value];
      if (x === undefined || y === undefined || z === undefined) continue;
      if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
      pts.push({ value: [x, y, z], individualId: ind.individual_id, allObjectives: ind.objectives });
    }
  }
  return pts;
});

const paretoData = computed<Point3D[]>(() =>
  (props.paretoFront ?? []).flatMap((item) => {
    const x = item.objectives[xKey.value];
    const y = item.objectives[yKey.value];
    const z = item.objectives[zKey.value];
    if (x === undefined || y === undefined || z === undefined) return [];
    if (isNaN(x) || isNaN(y) || isNaN(z)) return [];
    const individualId =
      chromosomeMap.value.get(stableStringify(item.chromosome)) ?? "";
    return [{ value: [x, y, z], individualId, allObjectives: item.objectives }];
  }),
);

const markedPoint = computed<Point3D | null>(() => {
  if (!markedId.value) return null;
  return paretoData.value.find((p) => p.individualId === markedId.value) ?? null;
});

// ── Tooltip ───────────────────────────────────────────────────────────────────

function formatTooltip(params: Record<string, unknown>): string {
  const d = params.data as Point3D | undefined;
  if (!d) return "";

  const dark = isDark.value;
  const mutedColor = dark ? "#7f849c" : "#6b7280";
  const sepColor = dark ? "#313244" : "#f3f4f6";

  const idHtml = d.individualId
    ? `<div style="font-family:monospace;font-size:11px;color:${mutedColor};margin-bottom:4px">${d.individualId.slice(0, 16)}…</div>`
    : "";

  let objRows = "";
  if (Array.isArray(d.allObjectives)) {
    objRows = d.allObjectives
      .map((v, i) => {
        const name = availableKeys.value[i] ?? `obj${i}`;
        return `<tr><td style="color:${mutedColor};padding-right:12px">${name}</td><td style="font-weight:600">${v.toFixed(6)}</td></tr>`;
      })
      .join("");
  } else {
    objRows = Object.entries(d.allObjectives)
      .map(([k, v]) => `<tr><td style="color:${mutedColor};padding-right:12px">${k}</td><td style="font-weight:600">${v.toFixed(6)}</td></tr>`)
      .join("");
  }

  const hint = markMode.value
    ? `<div style="margin-top:6px;font-size:10px;color:${mutedColor};border-top:1px solid ${sepColor};padding-top:4px">${markedId.value === d.individualId ? "Click to unpin" : "Click to pin"}</div>`
    : "";

  return `${idHtml}<table style="font-size:12px;border-spacing:0">${objRows}</table>${hint}`;
}

// ── Option builder ────────────────────────────────────────────────────────────

function buildOption() {
  if (!chart || !hasSufficientData.value) return;

  const dark = isDark.value;

  const axisColor  = dark ? "#45475a" : "#c0cad8";
  const labelColor = dark ? "#7f849c" : "#64748b";
  const nameColor  = dark ? "#a6adc8" : "#374151";
  const splitColor = dark ? "#313244" : "#dde3eb";
  const bgColor    = dark ? "#181825" : "#ffffff";
  const envColor   = dark ? "#181825" : "#ffffff";

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const axisConfig = (name: string): any => ({
    name,
    nameTextStyle: { color: nameColor, fontSize: 12, fontWeight: 600 },
    type: "value",
    axisLine: { lineStyle: { color: axisColor, width: 1.5 } },
    axisTick: { lineStyle: { color: axisColor } },
    axisLabel: { color: labelColor, fontSize: 11 },
    splitLine: { lineStyle: { color: splitColor, opacity: 0.6 } },
  });

  const series = [];

  if (populationData.value.length > 0) {
    series.push({
      name: "Population",
      type: "scatter3D",
      data: populationData.value,
      symbolSize: 3,
      itemStyle: {
        color: dark ? "#585b70" : "#b0bac8",
        opacity: dark ? 0.45 : 0.5,
      },
      emphasis: {
        itemStyle: { color: dark ? "#7f849c" : "#64748b", opacity: 0.9 },
      },
    });
  }

  series.push({
    name: "Pareto Front",
    type: "scatter3D",
    data: paretoData.value.filter((p) => p.individualId !== markedId.value),
    symbolSize: 6,
    itemStyle: {
      color: "#3b82f6",
      opacity: 0.92,
      borderWidth: 0,
    },
    emphasis: {
      itemStyle: { color: "#1d4ed8", opacity: 1 },
      label: { show: false },
    },
  });

  if (markedPoint.value) {
    series.push({
      name: "Pinned",
      type: "scatter3D",
      data: [markedPoint.value],
      symbolSize: 10,
      itemStyle: { color: "#f59e0b", opacity: 1 },
      emphasis: { itemStyle: { color: "#d97706", opacity: 1 } },
    });
  }

  const legendData = [
    ...(populationData.value.length > 0 ? ["Population"] : []),
    "Pareto Front",
    ...(markedPoint.value ? ["Pinned"] : []),
  ];

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const option: any = {
    backgroundColor: bgColor,
    tooltip: {
      trigger: "item",
      formatter: formatTooltip,
      padding: [8, 12],
      backgroundColor: dark ? "#1e1e2e" : "#ffffff",
      borderColor: dark ? "#313244" : "#e5e7eb",
      textStyle: { color: dark ? "#cdd6f4" : "#111827", fontSize: 12 },
      extraCssText: "box-shadow: 0 4px 12px rgba(0,0,0,0.15);",
    },
    legend: {
      bottom: 4,
      textStyle: { fontSize: 12, color: labelColor },
      inactiveColor: dark ? "#45475a" : "#d1d5db",
      data: legendData,
    },
    grid3D: {
      show: true,
      boxWidth: 100,
      boxHeight: 100,
      boxDepth: 100,
      axisLine: { lineStyle: { color: axisColor, width: 1.5 } },
      axisPointer: {
        show: true,
        lineStyle: { color: dark ? "#89b4fa" : "#3b82f6", width: 1.5, opacity: 0.7 },
      },
      splitLine: {
        show: true,
        lineStyle: { color: splitColor, opacity: 0.5, width: 1 },
      },
      splitArea: { show: false },
      light: {
        main: {
          intensity: dark ? 0.7 : 1.0,
          // Shadows look dramatic and dark on white backgrounds — disable in light
          shadow: dark,
          shadowQuality: "medium",
        },
        ambient: {
          // High ambient in light mode fills shadows and keeps the scene bright
          intensity: dark ? 0.4 : 0.8,
        },
      },
      environment: envColor,
      viewControl: {
        autoRotate: false,
        rotateSensitivity: 1.5,
        zoomSensitivity: 1.2,
        panSensitivity: 1,
        distance: 220,
        alpha: 20,
        beta: 40,
        minDistance: 40,
        maxDistance: 600,
      },
      postEffect: {
        enable: dark,
        // SSAO (ambient occlusion) darkens concavities — looks dirty on white
        SSAO: { enable: dark, radius: 2, intensity: dark ? 1.0 : 0, quality: "medium" },
      },
    },
    xAxis3D: axisConfig(xKey.value),
    yAxis3D: axisConfig(yKey.value),
    zAxis3D: axisConfig(zKey.value),
    series,
  };

  chart.setOption(option, true);
}

// ── Reactivity ────────────────────────────────────────────────────────────────

watch(
  [
    () => props.paretoFront,
    () => props.generations,
    xKey, yKey, zKey,
    markedPoint,
    isDark,
  ],
  () => buildOption(),
  { deep: true },
);
</script>

<style scoped>
.chart-wrap {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart {
  flex: 1;
  min-height: 340px;
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

.pin-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.pin-btn {
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

.pin-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

.pin-btn.active {
  background: #fef3c7;
  border-color: #fcd34d;
  color: #92400e;
}

.marked-badge {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px 2px 10px;
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: var(--radius-sm);
  font-size: 11px;
  color: #92400e;
}

.marked-id {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 10px;
}

.clear-pin {
  font-size: 11px;
  color: #b45309;
  line-height: 1;
  padding: 0 2px;
  border: none;
  background: transparent;
  cursor: pointer;
  opacity: 0.7;
  transition: opacity 0.1s;
}

.clear-pin:hover {
  opacity: 1;
}
</style>
