import { ref, watch, onBeforeUnmount, type Ref } from "vue";
import * as echarts from "../lib/echarts";
// Types come from the full package (echarts/core doesn't re-export them);
// type-only imports cost nothing at runtime.
import type { EChartsOption, SetOptionOpts } from "echarts";

export function useEChart(containerRef: Ref<HTMLElement | null>) {
  let chart: echarts.EChartsType | null = null;
  let ro: ResizeObserver | null = null;
  const ready = ref(false);

  // Replayed onto a fresh instance whenever the container element is
  // recreated by v-if (empty state ↔ data state): without this the chart
  // stays bound to the detached old node and renders blank until refresh.
  let lastOption: EChartsOption | null = null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handlers: Array<[string, (params: any) => void]> = [];

  function teardown() {
    ro?.disconnect();
    ro = null;
    chart?.dispose();
    chart = null;
    ready.value = false;
  }

  function init(el: HTMLElement) {
    // canvas (not svg): with hundreds/thousands of Pareto/population points,
    // svg pays one DOM node per point — canvas draws them as pixels instead,
    // which is what keeps large scatter series responsive.
    chart = echarts.init(el, null, { renderer: "canvas" });
    for (const [event, handler] of handlers) chart.on(event, handler);
    if (lastOption) chart.setOption(lastOption, true);
    ro = new ResizeObserver(() => {
      // Skip collapsed/hidden passes — resizing the canvas to 0×0 blanks it
      // and there is nothing to lay out anyway; the observer fires again as
      // soon as the element regains a size.
      if (el.clientWidth > 0 && el.clientHeight > 0) chart?.resize();
    });
    ro.observe(el);
    ready.value = true;
  }

  // flush: "post" so the watcher sees the DOM element after Vue patches it;
  // also covers the element being created later or swapped by v-if.
  watch(
    containerRef,
    (el) => {
      teardown();
      if (el) init(el);
    },
    { immediate: true, flush: "post" },
  );

  onBeforeUnmount(teardown);

  function setOption(
    option: EChartsOption,
    opts: boolean | SetOptionOpts = true,
  ) {
    lastOption = option;
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
    handlers.push([event, handler]);
    chart?.on(event, handler);
  }

  function dispatch(action: echarts.Payload) {
    chart?.dispatchAction(action);
  }

  return { ready, setOption, resize, on, dispatch };
}
