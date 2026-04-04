<template>
  <Teleport to="body">
    <div class="backdrop" @click.self="$emit('close')" />
    <div class="modal">
      <div class="modal-header">
        <div class="header-left">
          <span class="modal-title">Problem: {{ problemName }}</span>
          <div class="header-pills">
            <span class="pill">{{ numberOfRelays }} relay slots</span>
            <span class="pill">reach {{ radiusOfReach }}m</span>
            <span class="pill">inter {{ radiusOfInter }}m</span>
            <span v-if="candidateCount > 0" class="pill pill--blue">{{ candidateCount }} candidates</span>
            <span v-if="mobileNodeCount > 0" class="pill pill--green">{{ mobileNodeCount }} mobile nodes</span>
          </div>
        </div>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </div>
      <div ref="chartEl" class="chart" />
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from "vue";
import { useEChart } from "../../composables/useEChart";
import type { JsonObject } from "../../types/simlab";

const props = defineProps<{
  problem: JsonObject;
}>();

defineEmits<{ (e: "close"): void }>();

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready, resize } = useEChart(chartEl);

// -------------------------------------------------------
// Parsed problem fields
// -------------------------------------------------------

interface MobileNode {
  name: string;
  path_segments: [string, string][];
  is_closed: boolean;
  is_round_trip: boolean;
}

const problemName = computed(() => String(props.problem.name ?? ""));
const numberOfRelays = computed(() => props.problem.number_of_relays ?? "?");
const radiusOfReach = computed(() => props.problem.radius_of_reach ?? "?");
const radiusOfInter = computed(() => props.problem.radius_of_inter ?? "?");
const region = computed(() => (props.problem.region as [number, number, number, number]) ?? [-100, -100, 100, 100]);
const sink = computed(() => (props.problem.sink as [number, number]) ?? [0, 0]);
const candidates = computed(() => (props.problem.candidates as [number, number][] | undefined) ?? []);
const mobileNodes = computed(() => (props.problem.mobile_nodes as unknown as MobileNode[] | undefined) ?? []);
const candidateCount = computed(() => candidates.value.length);
const mobileNodeCount = computed(() => mobileNodes.value.length);

// -------------------------------------------------------
// Path evaluation (parametric expressions with numpy-style math)
// -------------------------------------------------------

function evalExpr(expr: string, t: number): number {
  const js = expr
    .replace(/np\.cos/g, "Math.cos")
    .replace(/np\.sin/g, "Math.sin")
    .replace(/np\.tan/g, "Math.tan")
    .replace(/np\.pi/g, "Math.PI")
    .replace(/np\.sqrt/g, "Math.sqrt")
    .replace(/np\.exp/g, "Math.exp")
    .replace(/np\.log/g, "Math.log")
    .replace(/np\.abs/g, "Math.abs");
  // eslint-disable-next-line no-new-func
  return new Function("t", `return ${js}`)(t) as number;
}

function sampleSegment(xExpr: string, yExpr: string, steps = 80): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    try {
      pts.push([evalExpr(xExpr, t), evalExpr(yExpr, t)]);
    } catch {
      // skip invalid expression values
    }
  }
  return pts;
}

function buildNodePath(node: MobileNode): [number, number][] {
  let pts: [number, number][] = [];
  for (const [xExpr, yExpr] of node.path_segments) {
    const seg = sampleSegment(xExpr, yExpr);
    if (pts.length > 0) seg.shift(); // remove duplicate junction point
    pts = pts.concat(seg);
  }
  if (node.is_closed) {
    if (pts.length > 0) pts.push([pts[0][0], pts[0][1]]);
  }
  return pts;
}

// -------------------------------------------------------
// Chart
// -------------------------------------------------------

const MOBILE_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];

// Grid margins (px) — must match the setOption grid below
const GRID = { left: 56, right: 24, top: 20, bottom: 56 };

function equalScaleRange(): { xMin: number; xMax: number; yMin: number; yMax: number } {
  const [xmin, ymin, xmax, ymax] = region.value;
  const xCenter = (xmin + xmax) / 2;
  const yCenter = (ymin + ymax) / 2;

  const el = chartEl.value;
  const plotW = el ? el.clientWidth  - GRID.left - GRID.right  : 800;
  const plotH = el ? el.clientHeight - GRID.top  - GRID.bottom : 464;

  // units-per-pixel that fits the full region in each axis
  const scaleX = (xmax - xmin) / plotW;
  const scaleY = (ymax - ymin) / plotH;
  // use the larger scale so the region fits completely
  const scale = Math.max(scaleX, scaleY) * 1.08; // 8% padding

  const xHalf = (scale * plotW) / 2;
  const yHalf = (scale * plotH) / 2;

  return {
    xMin: xCenter - xHalf,
    xMax: xCenter + xHalf,
    yMin: yCenter - yHalf,
    yMax: yCenter + yHalf,
  };
}

function buildOption() {
  if (!ready.value) return;

  const [xmin, ymin, xmax, ymax] = region.value;
  const { xMin, xMax, yMin, yMax } = equalScaleRange();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any[] = [];

  // Region boundary (dashed rectangle)
  series.push({
    name: "Region",
    type: "line",
    data: [
      [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin],
    ],
    lineStyle: { color: "#9ca3af", width: 1.5, type: "dashed" },
    itemStyle: { color: "#9ca3af" },
    symbol: "none",
    silent: true,
    z: 0,
    tooltip: { show: false },
    legendHoverLink: false,
  });

  // Candidates
  if (candidates.value.length > 0) {
    const showLabels = candidates.value.length <= 20;
    series.push({
      name: "Candidates",
      type: "scatter",
      data: candidates.value.map((c, i) => ({
        value: c,
        label: showLabels
          ? { show: true, formatter: String(i + 1), position: "top", fontSize: 9, color: "#6b7280" }
          : { show: false },
      })),
      symbolSize: 9,
      itemStyle: { color: "#d1d5db", borderColor: "#9ca3af", borderWidth: 1.5 },
      z: 1,
    });
  }

  // Mobile node paths
  for (let i = 0; i < mobileNodes.value.length; i++) {
    const node = mobileNodes.value[i];
    const color = MOBILE_COLORS[i % MOBILE_COLORS.length];
    const path = buildNodePath(node);
    if (path.length === 0) continue;

    // Dashed path line
    series.push({
      name: node.name,
      type: "line",
      data: path,
      lineStyle: { color, width: 2, type: "dashed" },
      itemStyle: { color },
      symbol: "none",
      z: 2,
    });

    // Start-point marker with label
    series.push({
      name: `${node.name} pos`,
      type: "scatter",
      data: [{ value: path[0], label: { show: true, formatter: node.name, position: "top", fontSize: 10, color, fontWeight: "bold" } }],
      symbolSize: 10,
      itemStyle: { color, borderColor: "#fff", borderWidth: 1.5 },
      z: 3,
      silent: true,
      tooltip: { show: false },
      legendHoverLink: false,
    });

    // Direction arrow: place a small arrowhead at ~25% of the path
    const arrowIdx = Math.floor(path.length * 0.25);
    if (arrowIdx > 0 && arrowIdx < path.length) {
      const [ax, ay] = path[arrowIdx];
      const [px, py] = path[arrowIdx - 1];
      const angle = (Math.atan2(ay - py, ax - px) * 180) / Math.PI;
      series.push({
        name: `${node.name} arrow`,
        type: "scatter",
        data: [{ value: [ax, ay] }],
        symbol: "arrow",
        symbolSize: 12,
        symbolRotate: angle,
        itemStyle: { color },
        z: 4,
        silent: true,
        tooltip: { show: false },
        legendHoverLink: false,
      });
    }
  }

  // Sink (rendered last so it sits on top)
  series.push({
    name: "Sink",
    type: "scatter",
    data: [{ value: sink.value, label: { show: true, formatter: "Sink", position: "top", fontSize: 11, fontWeight: "bold", color: "#dc2626" } }],
    symbol: "diamond",
    symbolSize: 18,
    itemStyle: { color: "#ef4444", borderColor: "#991b1b", borderWidth: 2 },
    z: 5,
  });

  const legendNames = [
    ...(candidates.value.length > 0 ? ["Candidates"] : []),
    ...mobileNodes.value.map((n) => n.name),
    "Sink",
  ];

  setOption({
    tooltip: {
      trigger: "item",
      formatter: (params: { seriesName: string; data: { value?: [number, number] } | [number, number] }) => {
        const raw = (params.data as { value?: [number, number] }).value ?? (params.data as [number, number]);
        if (!Array.isArray(raw)) return "";
        return `${params.seriesName}<br>(${(raw[0] as number).toFixed(2)}, ${(raw[1] as number).toFixed(2)})`;
      },
    },
    legend: {
      bottom: 4,
      type: "scroll",
      textStyle: { fontSize: 11 },
      data: legendNames,
    },
    grid: { left: GRID.left, right: GRID.right, top: GRID.top, bottom: GRID.bottom },
    xAxis: {
      type: "value",
      min: xMin,
      max: xMax,
      splitLine: { lineStyle: { color: "#f0f0f0" } },
      axisLabel: { fontSize: 11 },
    },
    yAxis: {
      type: "value",
      min: yMin,
      max: yMax,
      splitLine: { lineStyle: { color: "#f0f0f0" } },
      axisLabel: { fontSize: 11 },
    },
    series,
  });
}

function onResize() {
  resize();
  buildOption(); // recalculate equal-scale ranges for the new dimensions
}

onMounted(() => {
  buildOption();
  window.addEventListener("resize", onResize);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", onResize);
});
</script>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.35);
  z-index: 200;
}

.modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: min(920px, 96vw);
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.18);
  z-index: 201;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.modal-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--color-text);
}

.header-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.pill {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--color-border);
  background: var(--color-bg);
  color: var(--color-text-muted);
}

.pill--blue {
  background: #dbeafe;
  border-color: #bfdbfe;
  color: #1d4ed8;
}

.pill--green {
  background: #d1fae5;
  border-color: #a7f3d0;
  color: #065f46;
}

.close-btn {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-md);
  font-size: 14px;
  color: var(--color-text-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s;
}

.close-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

.chart {
  width: 100%;
  height: 540px;
}
</style>
