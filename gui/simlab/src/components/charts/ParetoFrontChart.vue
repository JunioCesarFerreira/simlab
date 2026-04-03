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
      <div class="axis-selectors" v-if="availableKeys.length > 2">
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
      <div ref="chartEl" class="chart" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from "vue";
import { useEChart } from "../../composables/useEChart";
import type { ParetoFrontItemDto, GenerationDto } from "../../types/simlab";

const props = defineProps<{
  paretoFront: ParetoFrontItemDto[] | null | undefined;
  generations?: GenerationDto[];
  objectiveNames?: string[];
}>();

const emit = defineEmits<{
  (e: "click-individual", individualId: string): void;
}>();

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready, on } = useEChart(chartEl);

const availableKeys = computed<string[]>(() => {
  if (props.objectiveNames && props.objectiveNames.length >= 2) return props.objectiveNames;
  const first = props.paretoFront?.[0];
  return first ? Object.keys(first.objectives) : [];
});

const hasSufficientData = computed(
  () => (props.paretoFront?.length ?? 0) > 0 && availableKeys.value.length >= 2,
);

const xKey = ref<string>("");
const yKey = ref<string>("");

watch(availableKeys, (keys) => {
  if (keys.length >= 2) { xKey.value = keys[0]; yKey.value = keys[1]; }
}, { immediate: true });

const xIdx = computed(() => availableKeys.value.indexOf(xKey.value));
const yIdx = computed(() => availableKeys.value.indexOf(yKey.value));

interface ChartPoint {
  value: [number, number];
  individualId: string;
  allObjectives: number[] | Record<string, number>;
}

// Stable serialization of objects to use as lookup key
function stableStringify(val: unknown): string {
  if (Array.isArray(val)) return `[${(val as unknown[]).map(stableStringify).join(",")}]`;
  if (val !== null && typeof val === "object") {
    const entries = Object.keys(val as object)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify((val as Record<string, unknown>)[k])}`);
    return `{${entries.join(",")}}`;
  }
  return JSON.stringify(val);
}

// Chromosome → individual_id map (built from all generations)
const chromosomeMap = computed(() => {
  const map = new Map<string, string>();
  for (const gen of props.generations ?? []) {
    for (const ind of gen.population) {
      map.set(stableStringify(ind.chromosome), ind.individual_id);
    }
  }
  return map;
});

const populationData = computed<ChartPoint[]>(() => {
  const seen = new Set<string>();
  const pts: ChartPoint[] = [];
  for (const gen of props.generations ?? []) {
    for (const ind of gen.population) {
      if (seen.has(ind.individual_id)) continue;
      seen.add(ind.individual_id);
      const x = ind.objectives[xIdx.value];
      const y = ind.objectives[yIdx.value];
      if (x === undefined || y === undefined || isNaN(x) || isNaN(y)) continue;
      pts.push({ value: [x, y], individualId: ind.individual_id, allObjectives: ind.objectives });
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
    const individualId = chromosomeMap.value.get(stableStringify(item.chromosome)) ?? "";
    return [{ value: [x, y], individualId, allObjectives: item.objectives }];
  }),
);

function formatTooltip(p: { data: ChartPoint }): string {
  const d = p.data;
  const idHtml = d.individualId
    ? `<div style="font-family:monospace;font-size:11px;color:#6b7280;margin-bottom:4px">${d.individualId.slice(0, 16)}…</div>`
    : "";

  let objRows: string;
  if (Array.isArray(d.allObjectives)) {
    objRows = (d.allObjectives as number[])
      .map((v, i) => {
        const name = availableKeys.value[i] ?? `obj${i}`;
        return `<tr><td style="color:#6b7280;padding-right:12px">${name}</td><td style="font-weight:600">${v.toFixed(6)}</td></tr>`;
      })
      .join("");
  } else {
    objRows = Object.entries(d.allObjectives as Record<string, number>)
      .map(([k, v]) => `<tr><td style="color:#6b7280;padding-right:12px">${k}</td><td style="font-weight:600">${v.toFixed(6)}</td></tr>`)
      .join("");
  }

  return `${idHtml}<table style="font-size:12px;border-spacing:0">${objRows}</table>`;
}

function buildOption() {
  if (!hasSufficientData.value || !ready.value) return;

  const series = [];

  if (populationData.value.length > 0) {
    series.push({
      name: "Population",
      type: "scatter" as const,
      data: populationData.value,
      symbolSize: 7,
      itemStyle: { color: "#94a3b8", borderColor: "rgba(255,255,255,0.6)", borderWidth: 1, opacity: 0.55 },
      emphasis: { itemStyle: { color: "#64748b", opacity: 1 } },
      z: 1,
    });
  }

  series.push({
    name: "Pareto Front",
    type: "scatter" as const,
    data: paretoData.value,
    symbolSize: 11,
    itemStyle: { color: "#3b82f6", borderColor: "#fff", borderWidth: 1.5 },
    emphasis: { itemStyle: { color: "#1d4ed8", borderColor: "#fff", borderWidth: 2 } },
    z: 2,
  });

  setOption({
    tooltip: {
      trigger: "item",
      formatter: formatTooltip,
      padding: [8, 12],
    },
    legend: {
      bottom: 0,
      textStyle: { fontSize: 12 },
      data: populationData.value.length > 0 ? ["Population", "Pareto Front"] : ["Pareto Front"],
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
    series,
  });
}

watch([ready, () => props.paretoFront, () => props.generations, xKey, yKey], buildOption);
onMounted(() => {
  buildOption();
  on("click", (params: { data?: ChartPoint }) => {
    const id = params.data?.individualId;
    if (id) emit("click-individual", id);
  });
});
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
  min-height: 300px;
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
</style>
