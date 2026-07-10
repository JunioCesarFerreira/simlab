import { ref, onMounted, onBeforeUnmount, type Ref } from "vue";
import * as echarts from "echarts";

export function useEChart(containerRef: Ref<HTMLElement | null>) {
  let chart: echarts.ECharts | null = null;
  let ro: ResizeObserver | null = null;
  const ready = ref(false);

  onMounted(() => {
    if (!containerRef.value) return;
    // canvas (not svg): with hundreds/thousands of Pareto/population points,
    // svg pays one DOM node per point — canvas draws them as pixels instead,
    // which is what keeps large scatter series responsive.
    chart = echarts.init(containerRef.value, null, { renderer: "canvas" });
    ready.value = true;
    ro = new ResizeObserver(() => chart?.resize());
    ro.observe(containerRef.value);
  });

  onBeforeUnmount(() => {
    ro?.disconnect();
    ro = null;
    chart?.dispose();
    chart = null;
  });

  function setOption(
    option: echarts.EChartsOption,
    opts: boolean | echarts.SetOptionOpts = true,
  ) {
    // Narrowed to a single (non-union) type per branch so it matches one of
    // ECharts' two setOption overloads — calling with the raw union type
    // doesn't type-check even though either branch alone is valid.
    if (typeof opts === "boolean") {
      chart?.setOption(option, opts);
    } else {
      chart?.setOption(option, opts);
    }
  }

  function resize() {
    chart?.resize();
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function on(event: string, handler: (params: any) => void) {
    chart?.on(event, handler);
  }

  function dispatch(action: echarts.Payload) {
    chart?.dispatchAction(action);
  }

  return { ready, setOption, resize, on, dispatch };
}
