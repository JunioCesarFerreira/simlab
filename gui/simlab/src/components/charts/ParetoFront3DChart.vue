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
            :class="['pan-btn', { active: panMode }]"
            :title="panMode ? 'Switch back to rotate mode (left drag rotates)' : 'Switch to pan mode (left drag moves the scene)'"
            @click="panMode = !panMode"
          >
            ✋ {{ panMode ? 'Panning' : 'Pan' }}
          </button>
          <button
            v-if="hasNicheLineData"
            :class="['niche-btn', { active: showNicheLines }]"
            :title="showNicheLines ? 'Hide niche lines' : 'Show niche lines (NSGA-III reference directions)'"
            @click="showNicheLines = !showNicheLines"
          >
            ⟁ Niche Lines
          </button>
          <button
            :class="['dom-btn', { active: dominanceMode }]"
            :title="dominanceMode ? 'Exit dominance mode' : 'Enter dominance mode — click a point to highlight the region it dominates'"
            @click="dominanceMode = !dominanceMode"
          >
            ▣ {{ dominanceMode ? (dominancePoint ? 'Clear…' : 'Click a point…') : 'Dominance' }}
          </button>
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

      <div ref="chartEl" :class="['chart', { 'chart--pan': panMode }]" />
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
import { computeRanksWithDuplicates } from "../../utils/nonDominatedSort";

const props = defineProps<{
  paretoFront: ParetoFrontItemDto[] | null | undefined;
  generations?: GenerationDto[];
  objectiveNames?: string[];
  objectiveGoals?: string[];
  strategy?: string;
  referencePointDivisions?: number;
}>();

const emit = defineEmits<{
  (e: "click-individual", individualId: string): void;
}>();

const { isDark } = useTheme();

// ── Chart lifecycle ─────────────────────────────────────────────────────────

const chartEl = ref<HTMLElement | null>(null);
let chart: echarts.ECharts | null = null;
let ro: ResizeObserver | null = null;
// True after the first successful setOption; subsequent calls use replaceMerge
// to preserve the user's camera state (zoom, rotation, pan).
let cameraInitialized = false;

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
    } else if (dominanceMode.value) {
      dominancePoint.value =
        dominancePoint.value?.individualId === data.individualId ? null : data;
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

// ── Niche lines (NSGA-III reference directions) ───────────────────────────────

const NSGA3_STRATEGIES = ['nsga3', 'nsga3_deap', 'nsga3_pymoo'] as const;
const MAX_NICHE_LINES = 500;

const showNicheLines = ref(true);

const hasNicheLineData = computed(() =>
  NSGA3_STRATEGIES.includes(props.strategy as never) &&
  !!props.referencePointDivisions &&
  props.referencePointDivisions >= 1 &&
  hasSufficientData.value &&
  paretoData.value.length > 0,
);

function dassDennisRefPoints(m: number, p: number): number[][] {
  const points: number[][] = [];
  function gen(dim: number, remaining: number, current: number[]) {
    if (dim === 1) { points.push([...current, remaining / p]); return; }
    for (let k = 0; k <= remaining; k++) gen(dim - 1, remaining - k, [...current, k / p]);
  }
  gen(m, p, []);
  return points;
}

// ── Pan mode ──────────────────────────────────────────────────────────────────

const panMode = ref(false);

// ── Pin state ─────────────────────────────────────────────────────────────────

const markMode = ref(false);
const markedId = ref("");

// ── Dominance region ──────────────────────────────────────────────────────────

const dominanceMode = ref(false);
const dominancePoint = ref<Point3D | null>(null);

watch(dominanceMode, (active) => { if (!active) dominancePoint.value = null; });

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

// ── Rank computation ─────────────────────────────────────────────────────────

const MAX_LABELED_RANKS = 5;

const RANK_PALETTE = [
  "#3b82f6", // Front 1 — blue
  "#10b981", // Front 2 — emerald
  "#f59e0b", // Front 3 — amber
  "#8b5cf6", // Front 4 — violet
  "#f43f5e", // Front 5 — rose
  "#94a3b8", // Other   — gray
] as const;

const RANK_SIZES_3D = [6, 5, 5, 4, 4, 3] as const;

const minimize3D = computed<boolean[]>(() =>
  (props.objectiveGoals ?? []).map((g) => g === "min"),
);

const rankMap = computed<Map<string, number>>(() => {
  if (!props.objectiveGoals?.length || populationData.value.length === 0) {
    return new Map();
  }
  const pts = populationData.value.map((p) => ({
    id: p.individualId,
    objectives: p.allObjectives as number[],
  }));
  const raw = computeRanksWithDuplicates(pts, minimize3D.value);
  const capped = new Map<string, number>();
  for (const [id, r] of raw) {
    capped.set(id, Math.min(r, MAX_LABELED_RANKS));
  }
  return capped;
});

const rankedGroups = computed<Point3D[][]>(() => {
  if (rankMap.value.size === 0) return [];
  const groups: Point3D[][] = Array.from({ length: MAX_LABELED_RANKS + 1 }, () => []);
  for (const p of populationData.value) {
    const r = rankMap.value.get(p.individualId) ?? MAX_LABELED_RANKS;
    groups[r]!.push(p);
  }
  return groups;
});

const markedPoint = computed<Point3D | null>(() => {
  if (!markedId.value) return null;
  const all = rankedGroups.value.length > 0
    ? rankedGroups.value.flat()
    : paretoData.value;
  return all.find((p) => p.individualId === markedId.value) ?? null;
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

  const useRanks = rankedGroups.value.length > 0;
  const series = [];

  if (useRanks) {
    const presentRanks = rankedGroups.value
      .map((g, r) => ({ r, g }))
      .filter(({ g }) => g.length > 0);

    for (const { r, g } of presentRanks) {
      const isOther = r === MAX_LABELED_RANKS;
      const name  = isOther ? "Other" : `Front ${r + 1}`;
      const color = RANK_PALETTE[r] ?? RANK_PALETTE[RANK_PALETTE.length - 1];
      const size  = RANK_SIZES_3D[r] ?? RANK_SIZES_3D[RANK_SIZES_3D.length - 1];
      const data  = g.filter((p) => p.individualId !== markedId.value);

      series.push({
        name,
        type: "scatter3D",
        data,
        symbolSize: size,
        itemStyle: {
          color,
          opacity: isOther ? (dark ? 0.35 : 0.4) : 0.9,
        },
        emphasis: { itemStyle: { color, opacity: 1 } },
      });
    }
  } else {
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
        emphasis: { itemStyle: { color: dark ? "#7f849c" : "#64748b", opacity: 0.9 } },
      });
    }
    series.push({
      name: "Pareto Front",
      type: "scatter3D",
      data: paretoData.value.filter((p) => p.individualId !== markedId.value),
      symbolSize: 6,
      itemStyle: { color: "#3b82f6", opacity: 0.92 },
      emphasis: { itemStyle: { color: "#1d4ed8", opacity: 1 }, label: { show: false } },
    });
  }

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

  // ── Niche lines (NSGA-III) ────────────────────────────────────────────────
  if (showNicheLines.value && hasNicheLineData.value) {
    const m = availableKeys.value.length;
    const p = props.referencePointDivisions!;
    const refPoints = dassDennisRefPoints(m, p);

    if (refPoints.length <= MAX_NICHE_LINES) {
      const xi = xIdx.value;
      const yi = yIdx.value;
      const zi = zIdx.value;

      // Use full population bounding box (wider than Pareto front alone)
      const bboxPts = populationData.value.length > 0
        ? populationData.value
        : paretoData.value;

      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      for (const pt of bboxPts) {
        const [x, y, z] = pt.value;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }

      const goals = props.objectiveGoals ?? [];
      const isMaxX = goals[xi] === 'max';
      const isMaxY = goals[yi] === 'max';
      const isMaxZ = goals[zi] === 'max';

      // Ideal point: best value per axis according to goal
      const ox = isMaxX ? maxX : minX;
      const oy = isMaxY ? maxY : minY;
      const oz = isMaxZ ? maxZ : minZ;

      // Sign: min objectives extend toward higher values; max toward lower
      const sx = isMaxX ? -1 : 1;
      const sy = isMaxY ? -1 : 1;
      const sz = isMaxZ ? -1 : 1;

      const rangeX = (maxX - minX) || 1;
      const rangeY = (maxY - minY) || 1;
      const rangeZ = (maxZ - minZ) || 1;

      const lineColor = dark ? '#74c7ec' : '#0284c7';
      const lineOpacity = dark ? 0.28 : 0.32;

      let added = 0;
      for (const ref of refPoints) {
        let wx = ref[xi] ?? 0;
        let wy = ref[yi] ?? 0;
        let wz = ref[zi] ?? 0;
        const wsum = wx + wy + wz;
        // Skip directions with no component in the displayed axes (only relevant for m > 3)
        if (wsum < 1e-10) continue;
        // L1-normalise the 3D projection so all lines reach the same scale
        // regardless of how many objectives are outside the 3 displayed axes
        wx /= wsum;
        wy /= wsum;
        wz /= wsum;

        // Extend the line until it exits the population bounding box:
        // in normalised space each axis is [0,1], so t_max = min(1/wi) for wi > 0.
        // This guarantees every Pareto point (which is inside the box) lies within the line.
        // An extra 10% pushes the tip visibly past the outermost data points.
        let tMax = Infinity;
        if (wx > 1e-10) tMax = Math.min(tMax, 1 / wx);
        if (wy > 1e-10) tMax = Math.min(tMax, 1 / wy);
        if (wz > 1e-10) tMax = Math.min(tMax, 1 / wz);
        if (!isFinite(tMax)) continue;
        tMax *= 1.1;

        series.push({
          name: `__niche_${added++}`,
          type: 'line3D',
          data: [
            [ox, oy, oz],
            [ox + sx * wx * tMax * rangeX, oy + sy * wy * tMax * rangeY, oz + sz * wz * tMax * rangeZ],
          ],
          lineStyle: { color: lineColor, width: 1, opacity: lineOpacity },
          silent: true,
        });
      }
    }
  }

  // ── Dominance region ─────────────────────────────────────────────────────────
  if (dominancePoint.value) {
    const dp = dominancePoint.value;
    const [dpx, dpy, dpz] = dp.value;

    const xi = xIdx.value, yi = yIdx.value, zi = zIdx.value;
    const goals = props.objectiveGoals ?? [];
    const isMaxX = goals[xi] === 'max';
    const isMaxY = goals[yi] === 'max';
    const isMaxZ = goals[zi] === 'max';

    // Bounding box from all available data (population or Pareto)
    const bboxPts = populationData.value.length > 0 ? populationData.value : paretoData.value;
    let bMinX = Infinity, bMinY = Infinity, bMinZ = Infinity;
    let bMaxX = -Infinity, bMaxY = -Infinity, bMaxZ = -Infinity;
    for (const pt of bboxPts) {
      const [x, y, z] = pt.value;
      if (x < bMinX) bMinX = x; if (x > bMaxX) bMaxX = x;
      if (y < bMinY) bMinY = y; if (y > bMaxY) bMaxY = y;
      if (z < bMinZ) bMinZ = z; if (z > bMaxZ) bMaxZ = z;
    }

    // Dominance box: from selected point to the "worst" corner per axis
    // min objective → dominated region extends toward higher values (nadir)
    // max objective → dominated region extends toward lower values (nadir)
    const lx = isMaxX ? bMinX : dpx;
    const ly = isMaxY ? bMinY : dpy;
    const lz = isMaxZ ? bMinZ : dpz;
    const hx = isMaxX ? dpx : bMaxX;
    const hy = isMaxY ? dpy : bMaxY;
    const hz = isMaxZ ? dpz : bMaxZ;

    const domColor = '#ef4444';

    // 6 faces of the dominance box rendered as parametric surfaces
    type PFn = (u: number, v: number) => number;
    const faces: Array<{ x: PFn; y: PFn; z: PFn }> = [
      { x: ()  => lx,              y: (u)    => ly + u*(hy-ly), z: (_u, v) => lz + v*(hz-lz) }, // x=lx
      { x: ()  => hx,              y: (u)    => ly + u*(hy-ly), z: (_u, v) => lz + v*(hz-lz) }, // x=hx
      { x: (u) => lx + u*(hx-lx), y: ()     => ly,             z: (_u, v) => lz + v*(hz-lz) }, // y=ly
      { x: (u) => lx + u*(hx-lx), y: ()     => hy,             z: (_u, v) => lz + v*(hz-lz) }, // y=hy
      { x: (u) => lx + u*(hx-lx), y: (_u,v) => ly + v*(hy-ly), z: ()     => lz             }, // z=lz
      { x: (u) => lx + u*(hx-lx), y: (_u,v) => ly + v*(hy-ly), z: ()     => hz             }, // z=hz
    ];

    faces.forEach((face, i) => {
      series.push({
        name: `__dom_face_${i}`,
        type: 'surface',
        coordinateSystem: 'cartesian3D',
        parametric: true,
        parametricEquation: {
          u: { min: 0, max: 1, step: 1 },
          v: { min: 0, max: 1, step: 1 },
          x: face.x,
          y: face.y,
          z: face.z,
        },
        itemStyle: { color: domColor, opacity: dark ? 0.18 : 0.15 },
        wireframe: { show: false },
        silent: true,
      });
    });

    // Highlight the dominance source point in red
    series.push({
      name: '__dom_point',
      type: 'scatter3D',
      data: [dp],
      symbolSize: 11,
      itemStyle: { color: domColor, opacity: 1 },
      silent: true,
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
        // In pan mode swap mouse buttons: left drag pans, middle drag rotates
        rotateMouseButton: panMode.value ? 'middle' : 'left',
        panMouseButton:    panMode.value ? 'left'   : 'middle',
        rotateSensitivity: 1.5,
        zoomSensitivity: 1.5,
        panSensitivity: 1,
        minDistance: 5,
        maxDistance: 2000,
        // Initial camera position — included ONLY on the first render.
        // Omitting distance/alpha/beta on updates means echarts-gl never
        // re-applies them, so the user's current zoom/rotation is preserved.
        ...(!cameraInitialized && { distance: 220, alpha: 20, beta: 40 }),
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

  if (!cameraInitialized) {
    chart.setOption(option, true);
    cameraInitialized = true;
  } else {
    // replaceMerge replaces only the series array (correct add/remove of niche
    // lines, dominance faces, etc.) while merging everything else — the
    // echarts-gl viewControl state (zoom, rotation, pan) is untouched.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (chart as any).setOption(option, { replaceMerge: ['series'] });
  }
}

// ── Reactivity ────────────────────────────────────────────────────────────────

watch(
  [
    () => props.paretoFront,
    () => props.generations,
    () => props.objectiveGoals,
    () => props.strategy,
    () => props.referencePointDivisions,
    xKey, yKey, zKey,
    markedPoint,
    showNicheLines,
    dominancePoint,
    panMode,
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

.pin-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.pan-btn,
.niche-btn,
.dom-btn,
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

.pan-btn:hover,
.niche-btn:hover,
.dom-btn:hover,
.pin-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

.pan-btn.active {
  background: #ede9fe;
  border-color: #a78bfa;
  color: #5b21b6;
}

.niche-btn.active {
  background: #e0f2fe;
  border-color: #38bdf8;
  color: #0369a1;
}

.dom-btn.active {
  background: #fee2e2;
  border-color: #fca5a5;
  color: #991b1b;
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
