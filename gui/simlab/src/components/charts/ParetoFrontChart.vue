<template>
  <div class="chart-wrap">
    <div v-if="!hasSufficientData" class="empty">
      <span v-if="!paretoFront || paretoFront.length === 0">
        Pareto front not available
      </span>
      <span v-else>
        At least 2 objectives are required for visualization
      </span>
    </div>
    <template v-else>
      <div class="controls-bar">
        <div v-if="availableKeys.length > 2" class="axis-selectors">
          <label>
            X Axis:
            <select v-model="xKey">
              <option v-for="k in availableKeys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
          <label>
            Y Axis:
            <select v-model="yKey">
              <option v-for="k in availableKeys" :key="k" :value="k">{{ k }}</option>
            </select>
          </label>
        </div>

        <div class="pin-controls">
          <button
            :class="['pin-btn', { active: markMode }]"
            :title="markMode ? 'Exit pin mode' : 'Enter pin mode — click Pareto points to pin them'"
            @click="markMode = !markMode"
          >
            📌 {{ markMode ? 'Pinning…' : 'Pin' }}
          </button>
          <div v-if="markedIds.length > 0" class="marked-badges">
            <div v-for="(id, i) in markedIds" :key="id" class="marked-badge">
              <span class="pin-swatch" :style="{ background: pinColorAt(i) }" />
              <span class="marked-id" :title="id">{{ id.slice(0, 10) }}…</span>
              <button class="clear-pin" title="Unpin" @click="unpin(id)">✕</button>
            </div>
            <button v-if="markedIds.length > 1" class="clear-all-btn" title="Clear all pins" @click="clearPins">
              Clear all
            </button>
          </div>
          <button
            v-if="isZoomed"
            class="reset-zoom-btn"
            title="Reset zoom to full view"
            @click="resetZoom"
          >↺ Reset zoom</button>
          <ChartExportButton @click="handleExportImage" />
        </div>
      </div>

      <div ref="chartEl" class="chart" role="img" aria-label="Pareto front scatter chart" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from "vue";
import type * as echarts from "echarts";
import type { TopLevelFormatterParams } from "echarts/types/dist/shared";
import { useEChart } from "../../composables/useEChart";
import { usePinnedPoints, pinColorAt } from "../../composables/usePinnedPoints";
import type { ParetoFrontItemDto, GenerationDto } from "../../types/simlab";
import { isPenalized } from "../../types/simlab";
import { stableStringify } from "../../utils/stableStringify";
import { chartExportFilename } from "../../utils/chartExport";
import ChartExportButton from "./ChartExportButton.vue";

const props = defineProps<{
  paretoFront: ParetoFrontItemDto[] | null | undefined;
  generations?: GenerationDto[];
  objectiveNames?: string[];
  objectiveGoals?: string[];
  initX?: string;
  initY?: string;
  /** individual_id -> non-dominated rank (0 = best), over the full objective
   *  vector. Computed once at the page level (experimentDetailStore) and
   *  shared across charts — see that store for why this is hoisted. */
  rankMap?: Map<string, number>;
  /** stable-stringified chromosome -> individual_id, also hoisted to the store. */
  chromosomeMap?: Map<string, string>;
}>();

const emit = defineEmits<{
  (e: "click-individual", individualId: string): void;
  (e: "axis-change", axes: { x: string; y: string }): void;
}>();

// Pin state models — bound by the parent to the SAME two refs passed to
// ParetoFront3DChart, so pins (and pin mode) survive the 2D/3D toggle
// instead of resetting when this component unmounts. Falls back to local,
// unbound defaults when no parent v-model is provided.
const markMode = defineModel<boolean>("markMode", { default: false });
const markedIds = defineModel<string[]>("markedIds", { default: () => [] });

interface ChartPoint {
  value: [number, number];
  individualId: string;
  allObjectives: number[] | Record<string, number>;
}

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready, on, dispatch, exportImage } = useEChart(chartEl);

function handleExportImage() {
  exportImage(chartExportFilename("pareto-front"));
}

// True after the first successful setOption; later rebuilds use replaceMerge
// so dataZoom (the user's current zoom) survives instead of resetting on
// every pin/axis-change/new-generation rebuild.
let chartInitialized = false;

const isZoomed = ref(false);

function resetZoom() {
  dispatch({ type: 'dataZoom', dataZoomIndex: 0, start: 0, end: 100 });
  isZoomed.value = false;
}

// Pin logic over the markMode/markedIds models above — see usePinnedPoints
// for why pinned points are never filtered out of their source series (that
// is what used to make them "disappear").
const { togglePin, unpin, clearPins } = usePinnedPoints(markMode, markedIds);

const availableKeys = computed<string[]>(() => {
  if (props.objectiveNames && props.objectiveNames.length >= 2) return props.objectiveNames;
  const first = props.paretoFront?.[0];
  return first ? Object.keys(first.objectives) : [];
});

const hasSufficientData = computed(
  () => (props.paretoFront?.length ?? 0) > 0 && availableKeys.value.length >= 2,
);

const xKey = ref(props.initX ?? "");
const yKey = ref(props.initY ?? "");

watch(availableKeys, (keys) => {
  if (keys.length >= 2) {
    if (!xKey.value || !keys.includes(xKey.value)) xKey.value = keys[0] ?? "";
    if (!yKey.value || !keys.includes(yKey.value)) yKey.value = keys[1] ?? "";
  }
}, { immediate: true });

watch([xKey, yKey], ([x, y]) => {
  emit("axis-change", { x, y });
});

const xIdx = computed(() => availableKeys.value.indexOf(xKey.value));
const yIdx = computed(() => availableKeys.value.indexOf(yKey.value));

const populationData = computed<ChartPoint[]>(() => {
  const seen = new Set<string>();
  const pts: ChartPoint[] = [];

  for (const gen of props.generations ?? []) {
    for (const ind of gen.population) {
      if (seen.has(ind.individual_id)) continue;
      seen.add(ind.individual_id);

      if (isPenalized(ind.objectives)) continue;

      const x = ind.objectives[xIdx.value];
      const y = ind.objectives[yIdx.value];
      if (x === undefined || y === undefined || isNaN(x) || isNaN(y)) continue;

      pts.push({
        value: [x, y],
        individualId: ind.individual_id,
        allObjectives: ind.objectives,
      });
    }
  }

  return pts;
});

const paretoData = computed<ChartPoint[]>(() =>
  (props.paretoFront ?? []).flatMap((item) => {
    const x = item.objectives[xKey.value];
    const y = item.objectives[yKey.value];
    if (x === undefined || y === undefined || isNaN(x) || isNaN(y)) return [];

    // Lookup by chromosome — independent of float precision or objective name differences
    const individualId = props.chromosomeMap?.get(stableStringify(item.chromosome)) ?? "";

    return [{
      value: [x, y],
      individualId,
      allObjectives: item.objectives,
    }];
  }),
);

// ── Rank computation ─────────────────────────────────────────────────────────

const MAX_LABELED_RANKS = 5;

// palette: index = rank (0-based), last entry = "other"
const RANK_PALETTE = [
  "#3b82f6", // Front 1 — blue
  "#10b981", // Front 2 — emerald
  "#f59e0b", // Front 3 — amber
  "#8b5cf6", // Front 4 — violet
  "#f43f5e", // Front 5 — rose
  "#94a3b8", // Other   — gray
] as const;

const RANK_SIZES = [11, 10, 9, 8, 7, 6] as const;

/** Map individualId → 0-indexed rank, capped at MAX_LABELED_RANKS.
 *  The expensive O(n²) dominance sort itself lives in the store
 *  (individualRankMap) — this is just a cheap O(n) capping pass. */
const rankMap = computed<Map<string, number>>(() => {
  const raw = props.rankMap;
  if (!raw || raw.size === 0) return new Map();
  const capped = new Map<string, number>();
  for (const [id, r] of raw) {
    capped.set(id, Math.min(r, MAX_LABELED_RANKS));
  }
  return capped;
});

/** populationData grouped by rank; group MAX_LABELED_RANKS = "Other" */
const rankedGroups = computed<ChartPoint[][]>(() => {
  if (rankMap.value.size === 0) return [];
  const groups: ChartPoint[][] = Array.from({ length: MAX_LABELED_RANKS + 1 }, () => []);
  for (const p of populationData.value) {
    const r = rankMap.value.get(p.individualId) ?? MAX_LABELED_RANKS;
    groups[r]!.push(p);
  }
  return groups;
});

// All points, keyed by individualId, re-projected onto the current axes.
// Population data is preferred (covers every generation); paretoData fills
// in ids that are on the front but weren't matched in populationData (e.g.
// no `generations` prop supplied).
const pointsById = computed<Map<string, ChartPoint>>(() => {
  const map = new Map<string, ChartPoint>();
  const base = rankedGroups.value.length > 0 ? rankedGroups.value.flat() : populationData.value;
  for (const p of base) if (p.individualId) map.set(p.individualId, p);
  for (const p of paretoData.value) if (p.individualId && !map.has(p.individualId)) map.set(p.individualId, p);
  return map;
});

// Every currently-pinned point that still resolves on the current axes, in
// pin order (order = badge/color order).
const markedPoints = computed<ChartPoint[]>(() =>
  markedIds.value.flatMap((id) => {
    const p = pointsById.value.get(id);
    return p ? [p] : [];
  }),
);

function formatTooltip(params: TopLevelFormatterParams): string {
  const point = (Array.isArray(params) ? params[0] : params) as { data?: ChartPoint };
  const d = point.data;
  if (!d) return "";

  const idHtml = d.individualId
    ? `<div style="font-family:monospace;font-size:11px;color:#6b7280;margin-bottom:4px">${d.individualId.slice(0, 16)}…</div>`
    : "";

  let objRows = "";
  if (Array.isArray(d.allObjectives)) {
    objRows = d.allObjectives
      .map((v, i) => {
        const name = availableKeys.value[i] ?? `obj${i}`;
        return `<tr><td style="color:#6b7280;padding-right:12px">${name}</td><td style="font-weight:600">${v.toFixed(6)}</td></tr>`;
      })
      .join("");
  } else {
    objRows = Object.entries(d.allObjectives)
      .map(([k, v]) => `<tr><td style="color:#6b7280;padding-right:12px">${k}</td><td style="font-weight:600">${v.toFixed(6)}</td></tr>`)
      .join("");
  }

  const hint = markMode.value
    ? `<div style="margin-top:6px;font-size:10px;color:#9ca3af;border-top:1px solid #f3f4f6;padding-top:4px">${markedIds.value.includes(d.individualId) ? "Click to unpin" : "Click to pin"}</div>`
    : "";

  return `${idHtml}<table style="font-size:12px;border-spacing:0">${objRows}</table>${hint}`;
}

function buildOption() {
  if (!hasSufficientData.value || !ready.value) return;

  const useRanks = rankedGroups.value.length > 0;
  const series = [];

  if (useRanks) {
    // ── Ranked view ──────────────────────────────────────────────────────────
    const presentRanks = rankedGroups.value
      .map((g, r) => ({ r, g }))
      .filter(({ g }) => g.length > 0);

    for (const { r, g } of presentRanks) {
      const isOther = r === MAX_LABELED_RANKS;
      const name = isOther ? "Other" : `Front ${r + 1}`;
      const color = RANK_PALETTE[r] ?? RANK_PALETTE[RANK_PALETTE.length - 1];
      const size  = RANK_SIZES[r]  ?? RANK_SIZES[RANK_SIZES.length - 1];

      series.push({
        name,
        type: "scatter" as const,
        // Pinned points stay in their rank group — see usePinnedPoints — so
        // they never disappear; the "Pinned" series below only adds a ring
        // highlight on top of the point that's already drawn here.
        data: g,
        symbolSize: size,
        // Canvas fast-path once a series crosses the threshold — draws points
        // without per-symbol graphic elements, which is what keeps a large
        // accumulated population (many generations) responsive.
        large: true,
        largeThreshold: 200,
        itemStyle: {
          color,
          borderColor: isOther ? "transparent" : "#fff",
          borderWidth: isOther ? 0 : 1,
          opacity: isOther ? 0.45 : 0.9,
        },
        emphasis: { itemStyle: { color, opacity: 1, borderWidth: 1.5 } },
        z: MAX_LABELED_RANKS + 2 - r,
      });
    }
  } else {
    // ── Fallback (no goals provided) ─────────────────────────────────────────
    if (populationData.value.length > 0) {
      series.push({
        name: "Population",
        type: "scatter" as const,
        data: populationData.value,
        symbolSize: 7,
        large: true,
        largeThreshold: 200,
        itemStyle: {
          color: "#94a3b8",
          borderColor: "rgba(255,255,255,0.6)",
          borderWidth: 1,
          opacity: 0.55,
        },
        emphasis: { itemStyle: { color: "#64748b", opacity: 1 } },
        z: 1,
      });
    }
    series.push({
      name: "Pareto Front",
      type: "scatter" as const,
      data: paretoData.value,
      symbolSize: 11,
      large: true,
      largeThreshold: 200,
      itemStyle: { color: "#3b82f6", borderColor: "#fff", borderWidth: 1.5 },
      emphasis: { itemStyle: { color: "#1d4ed8", borderColor: "#fff", borderWidth: 2 } },
      z: 2,
    });
  }

  // Pinned points — a hollow ring drawn on top of each pinned point, one
  // color per pin (cycled from PIN_COLORS). The ring never replaces the
  // point's own marker in its source series above, so the point itself never
  // disappears; the ring is purely an additive highlight, click-to-unpin.
  if (markedPoints.value.length > 0) {
    series.push({
      name: "Pinned",
      type: "scatter" as const,
      symbol: "circle",
      data: markedPoints.value.map((p, i) => ({
        ...p,
        symbolSize: 20,
        itemStyle: { color: "transparent", borderColor: pinColorAt(i), borderWidth: 3 },
        emphasis: { itemStyle: { borderWidth: 4 } },
      })),
      z: 10,
    });
  }

  const legendData = [
    ...(useRanks
      ? rankedGroups.value
          .map((g, r) => ({ r, g }))
          .filter(({ g }) => g.length > 0)
          .map(({ r }) => (r === MAX_LABELED_RANKS ? "Other" : `Front ${r + 1}`))
      : [
          ...(populationData.value.length > 0 ? ["Population"] : []),
          "Pareto Front",
        ]),
    ...(markedPoints.value.length > 0 ? ["Pinned"] : []),
  ];

  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: "item",
      formatter: formatTooltip,
      padding: [8, 12],
    },
    legend: {
      bottom: 0,
      textStyle: { fontSize: 12 },
      data: legendData,
    },
    grid: { left: 60, right: 24, top: 24, bottom: 72 },
    xAxis: {
      name: xKey.value,
      nameLocation: "middle",
      nameGap: 36,
      type: "value",
      splitLine: { lineStyle: { color: "#f0f0f0" } },
    },
    yAxis: {
      name: yKey.value,
      nameLocation: "middle",
      nameGap: 44,
      type: "value",
      splitLine: { lineStyle: { color: "#f0f0f0" } },
    },
    dataZoom: [{ type: "inside", xAxisIndex: 0, yAxisIndex: 0, filterMode: "none" }],
    series,
  };

  if (!chartInitialized) {
    // First render: full replace so every component (axes, dataZoom, legend…)
    // starts from a clean slate.
    isZoomed.value = false;
    setOption(option, true);
    chartInitialized = true;
  } else {
    // Later rebuilds (pin, axis swap, new generation…): only touch the series.
    // Everything else — notably dataZoom — is left alone via default merge,
    // so the user's current zoom survives instead of resetting every time.
    setOption(option, { replaceMerge: ["series"] });
  }
}

// markedPoints is a computed that returns a fresh array on every recompute
// (flatMap), so plain reference comparison already catches every pin change
// — no need for a deep watch here (would also deep-traverse `generations`).
watch(
  [ready, () => props.paretoFront, () => props.generations, () => props.objectiveGoals, xKey, yKey, markedPoints],
  buildOption,
);

onMounted(() => {
  buildOption();
  on("click", (params: { data?: ChartPoint }) => {
    const id = params.data?.individualId;
    if (!id) return;
    if (markMode.value) {
      togglePin(id);
    } else {
      emit("click-individual", id);
    }
  });
  on("datazoom", () => { isZoomed.value = true; });
});
</script>

<style scoped>
.chart-wrap {
  /* flex + min-height 0 (not height: 100%): the card that hosts this chart is
     user-resizable with overflow hidden — a rigid min-height here pushes the
     card's resize handle out of the clipped area, making it unreachable. */
  flex: 1;
  min-height: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chart {
  flex: 1;
  min-height: 0;
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

.marked-badges {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.marked-badge {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 2px 8px 2px 8px;
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: var(--radius-sm);
  font-size: 11px;
  color: #92400e;
}

.pin-swatch {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.15);
}

.marked-id {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 10px;
}

.clear-all-btn {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text-muted);
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.clear-all-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
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
