<template>
  <div class="chart-wrap">
    <div v-if="!hasSufficientData" class="empty">
      <span v-if="!paretoFront || paretoFront.length === 0">
        Frente de Pareto não disponível
      </span>
      <span v-else>
        São necessários pelo menos 2 objetivos para visualização
      </span>
    </div>
    <template v-else>
      <div class="axis-selectors" v-if="allKeys.length > 2">
        <label>
          Eixo X:
          <select v-model="xKey">
            <option v-for="k in allKeys" :key="k" :value="k">{{ k }}</option>
          </select>
        </label>
        <label>
          Eixo Y:
          <select v-model="yKey">
            <option v-for="k in allKeys" :key="k" :value="k">{{ k }}</option>
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
import type { ParetoFrontItemDto } from "../../types/simlab";

const props = defineProps<{
  paretoFront: ParetoFrontItemDto[] | null | undefined;
}>();

const chartEl = ref<HTMLElement | null>(null);
const { setOption, ready } = useEChart(chartEl);

const allKeys = computed(() => {
  const first = props.paretoFront?.[0];
  if (!first) return [];
  return Object.keys(first.objectives);
});

const hasSufficientData = computed(
  () => (props.paretoFront?.length ?? 0) > 0 && allKeys.value.length >= 2,
);

const xKey = ref<string>("");
const yKey = ref<string>("");

watch(allKeys, (keys) => {
  if (keys.length >= 2) {
    xKey.value = keys[0];
    yKey.value = keys[1];
  }
}, { immediate: true });

function buildOption() {
  if (!hasSufficientData.value) return;
  const data = (props.paretoFront ?? []).map((item) => ({
    value: [item.objectives[xKey.value], item.objectives[yKey.value]],
    itemStyle: { opacity: 0.85 },
  }));

  setOption({
    tooltip: {
      trigger: "item",
      formatter: (p: { value: number[] }) =>
        `${xKey.value}: <b>${p.value[0].toFixed(4)}</b><br>${yKey.value}: <b>${p.value[1].toFixed(4)}</b>`,
    },
    grid: { left: 60, right: 24, top: 24, bottom: 48 },
    xAxis: {
      name: xKey.value,
      nameLocation: "middle",
      nameGap: 30,
      type: "value",
      splitLine: { lineStyle: { color: "#f0f0f0" } },
    },
    yAxis: {
      name: yKey.value,
      nameLocation: "middle",
      nameGap: 40,
      type: "value",
      splitLine: { lineStyle: { color: "#f0f0f0" } },
    },
    series: [
      {
        type: "scatter",
        data,
        symbolSize: 9,
        itemStyle: { color: "#3b82f6", borderColor: "#fff", borderWidth: 1 },
        emphasis: { itemStyle: { color: "#1d4ed8", borderColor: "#fff", borderWidth: 2 } },
      },
    ],
  });
}

watch([ready, () => props.paretoFront, xKey, yKey], buildOption);
onMounted(buildOption);
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
  min-height: 280px;
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
