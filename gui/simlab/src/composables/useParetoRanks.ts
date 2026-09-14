import { onScopeDispose, shallowRef, watch, type Ref } from "vue";
import type { SortablePoint } from "../utils/nonDominatedSort";

export function useParetoRanks(input: Ref<{ points: SortablePoint[]; minimize: boolean[] }>) {
  const ranks = shallowRef(new Map<string, number>());
  let worker: Worker | null = null;
  watch(input, (value) => {
    worker?.terminate();
    worker = null;
    ranks.value = new Map();
    if (!value.points.length) return;
    const next = new Worker(new URL("../workers/paretoRanks.ts", import.meta.url), { type: "module" });
    worker = next;
    next.onmessage = (event: MessageEvent<Map<string, number>>) => {
      if (worker !== next) return;
      ranks.value = event.data;
      next.terminate();
      worker = null;
    };
    next.onerror = () => {
      next.terminate();
      if (worker === next) worker = null;
    };
    next.postMessage(value);
  }, { immediate: true, flush: "post" });
  onScopeDispose(() => {
    worker?.terminate();
    worker = null;
  });
  return ranks;
}
