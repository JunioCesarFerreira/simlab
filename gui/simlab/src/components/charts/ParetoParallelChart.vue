<template>
  <div class="chart-wrap">
    <div v-if="!hasSufficientData" class="empty">
      <span v-if="!paretoFront?.length">Pareto front not available</span>
      <span v-else>At least 2 objectives are required for visualization</span>
    </div>
    <div v-else ref="chartEl" class="chart" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from "vue";
import { useEChart } from "../../composables/useEChart";
import { useTheme } from "../../composables/useTheme";
import { PENALTY_THRESHOLD } from "../../types/simlab";
import type { ParetoFrontItemDto, GenerationDto, FloatMap } from "../../types/simlab";

const props = defineProps<{
  paretoFront: ParetoFrontItemDto[] | null | undefined;
  generations?: GenerationDto[];
  objectiveNames?: string[];
  objectiveGoals?: string[];
}>();

const emit = defineEmits<{
  (e: "click-individual", individualId: string): void;
}>();

const { isDark } = useTheme();

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready, on } = useEChart(chartEl);

// ── Derived keys & goals ──────────────────────────────────────────────────────

const availableKeys = computed<string[]>(() => {
  if (props.objectiveNames && props.objectiveNames.length >= 2) return props.objectiveNames;
  const first = props.paretoFront?.[0];
  return first ? Object.keys(first.objectives) : [];
});

const hasSufficientData = computed(
  () => (props.paretoFront?.length ?? 0) > 0 && availableKeys.value.length >= 2,
);

function goalFor(key: string): "min" | "max" {
  const idx = props.objectiveNames?.indexOf(key) ?? -1;
  return (idx >= 0 ? props.objectiveGoals?.[idx] : undefined) === "max" ? "max" : "min";
}

// ── Individual ID resolution ──────────────────────────────────────────────────

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

interface ResolvedItem {
  individualId: string;
  objectives: FloatMap;
}

const resolvedItems = computed<ResolvedItem[]>(() =>
  (props.paretoFront ?? []).flatMap((item) => {
    if (Object.values(item.objectives).some((v) => v >= PENALTY_THRESHOLD)) return [];
    const individualId =
      chromosomeMap.value.get(stableStringify(item.chromosome)) ?? "";
    return [{ individualId, objectives: item.objectives }];
  }),
);

// ── Tooltip ───────────────────────────────────────────────────────────────────

function formatTooltip(params: unknown): string {
  const p = params as { dataIndex: number };
  const item = resolvedItems.value[p.dataIndex];
  if (!item) return "";

  const dark = isDark.value;
  const mutedColor = dark ? "#7f849c" : "#6b7280";
  const textColor  = dark ? "#cdd6f4" : "#111827";

  const idHtml = item.individualId
    ? `<div style="font-family:monospace;font-size:11px;color:${mutedColor};margin-bottom:4px">${item.individualId.slice(0, 16)}…</div>`
    : "";

  const rows = availableKeys.value
    .map((k) => {
      const v = item.objectives[k];
      const goal = goalFor(k);
      return (
        `<tr>` +
        `<td style="color:${mutedColor};padding-right:12px">${k} (${goal === "min" ? "↓" : "↑"})</td>` +
        `<td style="font-weight:600;color:${textColor}">${v !== undefined ? v.toFixed(6) : "—"}</td>` +
        `</tr>`
      );
    })
    .join("");

  return `${idHtml}<table style="font-size:12px;border-spacing:0">${rows}</table>`;
}

// ── Option builder ────────────────────────────────────────────────────────────

function buildOption() {
  if (!hasSufficientData.value || !ready.value) return;

  const dark  = isDark.value;
  const items = resolvedItems.value;
  const keys  = availableKeys.value;

  const axisColor  = dark ? "#45475a" : "#c0cad8";
  const labelColor = dark ? "#7f849c" : "#64748b";
  const nameColor  = dark ? "#a6adc8" : "#374151";
  const splitColor = dark ? "#313244" : "#e5e7eb";
  const bgColor    = dark ? "#181825" : "#ffffff";

  // Multi-stop gradient: works on both dark and light backgrounds
  const colorScale = dark
    ? ["#38bdf8", "#4ade80", "#fde047", "#fb923c", "#f87171"]
    : ["#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626"];

  // Per-axis actual min / max
  const mins = keys.map((k) => Math.min(...items.map((it) => it.objectives[k] ?? 0)));
  const maxs = keys.map((k) => Math.max(...items.map((it) => it.objectives[k] ?? 0)));

  // Data: plain value arrays — visualMap colours each line by dim 0
  const data = items.map((item) => keys.map((k) => item.objectives[k] ?? 0));

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const parallelAxis: any[] = keys.map((k, i) => {
    const goal  = goalFor(k);
    const range = (maxs[i] ?? 0) - (mins[i] ?? 0);
    const pad   = range === 0 ? 0.5 : range * 0.05;
    return {
      dim: i,
      name: `${k} (${goal === "min" ? "↓" : "↑"})`,
      nameLocation: "end",
      nameTextStyle: { fontSize: 11, fontWeight: 600, color: nameColor },
      nameGap: 10,
      inverse: goal === "min",   // ↑ = better direction on every axis
      type: "value",
      min: (mins[i] ?? 0) - pad,
      max: (maxs[i] ?? 1) + pad,
      axisLine: { lineStyle: { color: axisColor, width: 1.5 } },
      axisTick: { lineStyle: { color: axisColor } },
      axisLabel: {
        fontSize: 10,
        color: labelColor,
        formatter: (v: number) =>
          Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0)
            ? v.toExponential(2)
            : v.toPrecision(3),
      },
      splitLine: { show: true, lineStyle: { color: splitColor, opacity: 0.5 } },
    };
  });

  const colorKey  = keys[0] ?? "";
  const colorGoal = goalFor(colorKey);

  setOption({
    backgroundColor: bgColor,
    tooltip: {
      trigger: "item",
      formatter: formatTooltip,
      padding: [8, 12],
      backgroundColor: dark ? "#1e1e2e" : "#ffffff",
      borderColor:     dark ? "#313244" : "#e5e7eb",
      textStyle: { color: dark ? "#cdd6f4" : "#111827", fontSize: 12 },
      extraCssText: "box-shadow: 0 4px 12px rgba(0,0,0,0.15);",
    },
    // Colour lines by the value of the first objective
    visualMap: {
      show: true,
      type: "continuous",
      min: mins[0] ?? 0,
      max: maxs[0] ?? 1,
      dimension: 0,
      orient: "horizontal",
      left: "center",
      bottom: 2,
      // text[0] = high-value end, text[1] = low-value end
      text: [
        `${colorGoal === "min" ? "worse ↑" : "better ↑"} ${colorKey}`,
        `${colorGoal === "min" ? "↓ better" : "↓ worse"} ${colorKey}`,
      ],
      textStyle: { fontSize: 10, color: labelColor },
      calculable: false,
      itemWidth: 14,
      itemHeight: 110,
      inRange: { color: colorScale },
    },
    parallel: {
      left: 40,
      right: 40,
      top: 56,
      bottom: 52,   // room for the visualMap legend
      layout: "horizontal",
      axisExpandable: keys.length > 8,
    },
    parallelAxis,
    series: [
      {
        type: "parallel",
        smooth: false,
        lineStyle: { width: 1.5, opacity: 0.65 },
        emphasis: { lineStyle: { width: 3, opacity: 1 } },
        data,
      },
    ],
  });
}

// ── Reactivity & events ───────────────────────────────────────────────────────

watch(
  [ready, resolvedItems, () => props.objectiveGoals, isDark],
  () => buildOption(),
  { deep: true },
);

onMounted(() => {
  buildOption();
  on("click", (params: { dataIndex?: number }) => {
    const idx = params.dataIndex;
    if (idx === undefined || idx < 0) return;
    const item = resolvedItems.value[idx];
    if (item?.individualId) emit("click-individual", item.individualId);
  });
});
</script>

<style scoped>
/* flex: 1 fills the remaining space inside .chart-card after .section-title */
.chart-wrap {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.chart {
  flex: 1;
  min-height: 200px;
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
</style>
