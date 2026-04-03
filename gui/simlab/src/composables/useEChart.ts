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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function on(event: string, handler: (params: any) => void) {
    chart?.on(event, handler);
  }

  return { ready, setOption, resize, on };
}
