<template>
  <Teleport to="body">
    <div class="backdrop" @click.self="$emit('close')" />
    <div class="modal">
      <div class="modal-header">
        <div class="header-left">
          <span class="modal-title">DODAG tree · {{ moteCount }} motes</span>
          <div class="header-pills">
            <span class="pill">reach {{ radiusOfReach }}m</span>
            <span v-if="hasExactTree" class="pill pill--blue">{{ edges.length }} edges</span>
            <span v-else class="pill pill--amber">approx only (no exact tree)</span>
            <span v-if="maxDepth !== null" class="pill">max depth {{ maxDepth }}</span>
            <span v-if="disconnectedIds.length > 0" class="pill pill--amber">
              {{ disconnectedIds.length }} disconnected
            </span>
          </div>
        </div>
        <div class="header-actions">
          <ChartExportButton @click="handleExportImage" />
          <button class="close-btn" @click="$emit('close')">✕</button>
        </div>
      </div>
      <div ref="chartEl" class="chart" />
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from "vue";
import type { TopLevelFormatterParams } from "echarts/types/dist/shared";
import { useEChart } from "../../composables/useEChart";
import type { SimulationDto } from "../../types/simlab";
import { chartExportFilename } from "../../utils/chartExport";
import ChartExportButton from "../charts/ChartExportButton.vue";

const props = defineProps<{ sim: SimulationDto }>();
defineEmits<{ (e: "close"): void }>();

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready, resize, exportImage } = useEChart(chartEl);

function handleExportImage() {
  exportImage(chartExportFilename(`dodag-${props.sim.id}`));
}

// -------------------------------------------------------
// Parameters & positions
// -------------------------------------------------------
// Mote ids are assigned by the Cooja builder as (fixed motes first, then
// mobile motes), 1-based; mote 1 is the root. See replace_xml.update_simulation_xml.

interface Params {
  region?: [number, number, number, number];
  radiusOfReach?: number;
  simulationElements?: {
    fixedMotes?: { position: [number, number] }[];
    mobileMotes?: { functionPath?: [string, string][] }[];
  };
}

const params = computed(() => (props.sim.parameters ?? {}) as unknown as Params);
const region = computed<[number, number, number, number]>(
  () => params.value.region ?? [-100, -100, 100, 100],
);
const radiusOfReach = computed(() => params.value.radiusOfReach ?? "?");
const dodag = computed(() => props.sim.dodag ?? null);

/** Evaluate a parametric coordinate expression (numpy-style) at time t. */
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
  return new Function("t", `return ${js}`)(t) as number;
}

/** moteId (1-based) -> [x, y]. Mobile motes use their t=0 start position. */
const positions = computed<Record<number, [number, number]>>(() => {
  const out: Record<number, [number, number]> = {};
  const se = params.value.simulationElements ?? {};
  const fixed = se.fixedMotes ?? [];
  const mobile = se.mobileMotes ?? [];
  let id = 1;
  for (const m of fixed) {
    if (Array.isArray(m.position)) out[id] = [m.position[0], m.position[1]];
    id += 1;
  }
  for (const m of mobile) {
    const seg = m.functionPath?.[0];
    if (seg) {
      try {
        out[id] = [evalExpr(seg[0], 0), evalExpr(seg[1], 0)];
      } catch {
        // unknown start position — leave unplaced
      }
    }
    id += 1;
  }
  return out;
});

const moteCount = computed(() => Object.keys(positions.value).length);

/** Parse the mote id from an RPL address: the last hextet, in hex (fd00::20a:a:a:a -> 10). */
function moteIdFromAddr(addr: string): number | null {
  const groups = addr.split(":");
  const last = groups[groups.length - 1];
  if (!last) return null;
  const id = parseInt(last, 16);
  return Number.isNaN(id) ? null : id;
}

const rootId = computed(() => {
  const root = dodag.value?.tree.root;
  return (root ? moteIdFromAddr(root) : null) ?? 1;
});

/** Depth per moteId, preferring the exact tree, falling back to approx. */
const depthById = computed<Record<number, number>>(() => {
  const out: Record<number, number> = {};
  const d = dodag.value;
  if (!d) return out;
  for (const [addr, depth] of Object.entries(d.tree.depth ?? {})) {
    const id = moteIdFromAddr(addr);
    if (id !== null) out[id] = depth;
  }
  if (Object.keys(out).length === 0) {
    for (const [addr, info] of Object.entries(d.approx.per_node ?? {})) {
      const id = moteIdFromAddr(addr);
      const depth = (info as { approx_depth: number | null }).approx_depth;
      if (id !== null && depth !== null) out[id] = depth;
    }
  }
  return out;
});

const hasExactTree = computed(() => !!dodag.value?.tree.root && edges.value.length > 0);

/** Exact tree edges as {child, parent} mote ids. */
const edges = computed<{ child: number; parent: number }[]>(() => {
  const out: { child: number; parent: number }[] = [];
  const e = dodag.value?.tree.edges ?? {};
  for (const [childAddr, parentAddr] of Object.entries(e)) {
    const child = moteIdFromAddr(childAddr);
    const parent = moteIdFromAddr(parentAddr);
    if (child !== null && parent !== null) out.push({ child, parent });
  }
  return out;
});

const maxDepth = computed(() => {
  const vals = Object.values(depthById.value);
  return vals.length > 0 ? Math.max(...vals) : null;
});

/** Deployed motes (non-root) that never joined the tree. */
const disconnectedIds = computed<number[]>(() => {
  const joined = new Set<number>([rootId.value, ...Object.keys(depthById.value).map(Number)]);
  for (const e of edges.value) joined.add(e.child);
  return Object.keys(positions.value)
    .map(Number)
    .filter((id) => !joined.has(id));
});

// -------------------------------------------------------
// Chart
// -------------------------------------------------------

const DEPTH_COLORS = ["#2563eb", "#0891b2", "#059669", "#65a30d", "#d97706", "#dc2626", "#9333ea"];
function depthColor(id: number): string {
  const d = depthById.value[id];
  if (d === undefined) return "#9ca3af";
  return DEPTH_COLORS[Math.min(d, DEPTH_COLORS.length - 1)] ?? "#9ca3af";
}

const GRID = { left: 56, right: 24, top: 20, bottom: 40 };

function equalScaleRange(): { xMin: number; xMax: number; yMin: number; yMax: number } {
  const [xmin, ymin, xmax, ymax] = region.value;
  const xCenter = (xmin + xmax) / 2;
  const yCenter = (ymin + ymax) / 2;
  const el = chartEl.value;
  const plotW = el ? el.clientWidth - GRID.left - GRID.right : 800;
  const plotH = el ? el.clientHeight - GRID.top - GRID.bottom : 500;
  const scale = Math.max((xmax - xmin) / plotW, (ymax - ymin) / plotH) * 1.08;
  const xHalf = (scale * plotW) / 2;
  const yHalf = (scale * plotH) / 2;
  return { xMin: xCenter - xHalf, xMax: xCenter + xHalf, yMin: yCenter - yHalf, yMax: yCenter + yHalf };
}

function formatTooltip(params: TopLevelFormatterParams): string {
  const p = (Array.isArray(params) ? params[0] : params) as {
    seriesName?: string;
    data?: { value?: [number, number]; moteId?: number };
  };
  const moteId = p.data?.moteId;
  const raw = p.data?.value;
  if (moteId === undefined || !raw) return "";
  const depth = depthById.value[moteId];
  const parent = edges.value.find((e) => e.child === moteId)?.parent;
  const lines = [
    `<b>mote ${moteId}</b>${moteId === rootId.value ? " (root)" : ""}`,
    `pos (${raw[0].toFixed(1)}, ${raw[1].toFixed(1)})`,
    depth !== undefined ? `depth ${depth}` : "not joined",
  ];
  if (parent !== undefined) lines.push(`parent → mote ${parent}`);
  return lines.join("<br>");
}

function buildOption() {
  if (!ready.value) return;

  const [xmin, ymin, xmax, ymax] = region.value;
  const { xMin, xMax, yMin, yMax } = equalScaleRange();
  const pos = positions.value;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any[] = [];

  // Region boundary (dashed rectangle)
  series.push({
    name: "Region",
    type: "line",
    data: [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]],
    lineStyle: { color: "#9ca3af", width: 1.5, type: "dashed" },
    symbol: "none",
    silent: true,
    z: 0,
    tooltip: { show: false },
    legendHoverLink: false,
  });

  // DODAG edges (child → parent) as individual 2-point lines sharing one legend
  // entry, plus a rotated arrowhead near the parent end to show direction.
  const arrowPoints: { value: [number, number] }[] = [];
  const arrowRotations: number[] = [];
  for (const { child, parent } of edges.value) {
    const c = pos[child];
    const p = pos[parent];
    if (!c || !p) continue;
    // All edge lines share the name "DODAG links" so the legend shows a single
    // entry that toggles the whole tree at once.
    series.push({
      name: "DODAG links",
      type: "line",
      data: [c, p],
      lineStyle: { color: "#60a5fa", width: 1.5 },
      symbol: "none",
      z: 1,
      silent: true,
      tooltip: { show: false },
      legendHoverLink: false,
    });
    // Arrowhead at 65% from child toward parent.
    const ax = c[0] + (p[0] - c[0]) * 0.65;
    const ay = c[1] + (p[1] - c[1]) * 0.65;
    arrowPoints.push({ value: [ax, ay] });
    arrowRotations.push((Math.atan2(p[1] - c[1], p[0] - c[0]) * 180) / Math.PI);
  }
  // Arrowheads: one scatter point per edge (symbolRotate is per-series, so emit
  // each as its own tiny series to rotate independently — cheap for tens of edges).
  arrowPoints.forEach((pt, i) => {
    series.push({
      name: "DODAG links",
      type: "scatter",
      data: [pt],
      symbol: "arrow",
      symbolSize: 9,
      symbolRotate: arrowRotations[i] ?? 0,
      itemStyle: { color: "#3b82f6" },
      z: 2,
      silent: true,
      tooltip: { show: false },
      legendHoverLink: false,
    });
  });

  // Nodes: joined (colored by depth) and disconnected (gray), labelled by mote id.
  const joinedData: { value: [number, number]; moteId: number; itemStyle: { color: string } }[] = [];
  const discData: { value: [number, number]; moteId: number }[] = [];
  for (const idStr of Object.keys(pos)) {
    const id = Number(idStr);
    if (id === rootId.value) continue;
    const point = pos[id];
    if (!point) continue;
    if (disconnectedIds.value.includes(id)) {
      discData.push({ value: point, moteId: id });
    } else {
      joinedData.push({ value: point, moteId: id, itemStyle: { color: depthColor(id) } });
    }
  }

  const showLabels = moteCount.value <= 40;
  const nodeLabel = (id: number) => ({
    show: showLabels,
    formatter: String(id),
    position: "top" as const,
    fontSize: 9,
    color: "#374151",
  });

  if (joinedData.length > 0) {
    series.push({
      name: "Joined",
      type: "scatter",
      data: joinedData.map((d) => ({ ...d, label: nodeLabel(d.moteId) })),
      symbolSize: 12,
      itemStyle: { borderColor: "#fff", borderWidth: 1.5 },
      z: 3,
    });
  }

  if (discData.length > 0) {
    series.push({
      name: "Disconnected",
      type: "scatter",
      data: discData.map((d) => ({ ...d, label: nodeLabel(d.moteId) })),
      symbolSize: 11,
      itemStyle: { color: "#e5e7eb", borderColor: "#9ca3af", borderWidth: 1.5, borderType: "dashed" },
      z: 3,
    });
  }

  // Root (rendered last, on top)
  const rootPos = pos[rootId.value];
  if (rootPos) {
    series.push({
      name: "Root",
      type: "scatter",
      data: [{
        value: rootPos,
        moteId: rootId.value,
        label: { show: true, formatter: "root", position: "top", fontSize: 11, fontWeight: "bold", color: "#dc2626" },
      }],
      symbol: "diamond",
      symbolSize: 18,
      itemStyle: { color: "#ef4444", borderColor: "#991b1b", borderWidth: 2 },
      z: 5,
    });
  }

  const legendNames = [
    ...(edges.value.length > 0 ? ["DODAG links"] : []),
    ...(joinedData.length > 0 ? ["Joined"] : []),
    ...(discData.length > 0 ? ["Disconnected"] : []),
    "Root",
  ];

  setOption({
    tooltip: { trigger: "item", formatter: formatTooltip },
    legend: { bottom: 2, type: "scroll", textStyle: { fontSize: 11 }, data: legendNames },
    grid: { left: GRID.left, right: GRID.right, top: GRID.top, bottom: GRID.bottom },
    xAxis: { type: "value", min: xMin, max: xMax, splitLine: { lineStyle: { color: "#f0f0f0" } }, axisLabel: { fontSize: 11 } },
    yAxis: { type: "value", min: yMin, max: yMax, splitLine: { lineStyle: { color: "#f0f0f0" } }, axisLabel: { fontSize: 11 } },
    series,
  });
}

function onResize() {
  resize();
  buildOption();
}

watch(ready, (r) => { if (r) buildOption(); }, { immediate: true });

onMounted(() => window.addEventListener("resize", onResize));
onBeforeUnmount(() => window.removeEventListener("resize", onResize));
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

.pill--amber {
  background: #fef3c7;
  border-color: #fde68a;
  color: #b45309;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
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
