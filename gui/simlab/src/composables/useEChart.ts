import { ref, onMounted, onBeforeUnmount, type Ref } from "vue";
import * as echarts from "echarts";

export function useEChart(containerRef: Ref<HTMLElement | null>) {
  let chart: echarts.ECharts | null = null;
  const ready = ref(false);

  onMounted(() => {
    if (containerRef.value) {
      chart = echarts.init(containerRef.value, null, { renderer: "svg" });
      ready.value = true;
    }
  });

  onBeforeUnmount(() => {
    chart?.dispose();
    chart = null;
  });

  function setOption(option: echarts.EChartsOption) {
    chart?.setOption(option, true);
  }

  function resize() {
    chart?.resize();
  }

  return { ready, setOption, resize };
}
