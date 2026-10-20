import { onScopeDispose, ref, shallowRef, watch } from "vue";
import { getHvGd, type HvGdData, type HvGdQuery } from "../api/metrics";

export function useHvGdData(query: () => HvGdQuery) {
  const data = shallowRef<HvGdData | null>(null);
  const state = ref<"loading" | "ready" | "empty" | "error">("loading");
  const errorMsg = ref("");
  const refreshing = ref(false);
  let active: { key: string; controller: AbortController } | null = null;
  let pending = false;
  let lastExperiment = "";

  function key() {
    const q = query();
    return JSON.stringify([q.experimentId, q.objectiveNames, q.objectiveGoals, q.population]);
  }

  async function retry() {
    const q = query();
    const requestKey = key();
    if (active?.key === requestKey) {
      // Coalesce arriving generations into one follow-up, keeping this useful
      // calculation alive instead of restarting it on every polling tick.
      pending = true;
      return;
    }
    active?.controller.abort();
    active = null;
    pending = false;
    if (lastExperiment !== q.experimentId) data.value = null;
    lastExperiment = q.experimentId;
    errorMsg.value = "";
    if (!q.experimentId || q.objectiveNames.length < 2) {
      data.value = null;
      state.value = "empty";
      refreshing.value = false;
      return;
    }
    const request = { key: requestKey, controller: new AbortController() };
    active = request;
    refreshing.value = data.value !== null;
    if (!data.value) state.value = "loading";
    try {
      const result = await getHvGd(q, request.controller.signal);
      if (active !== request) return;
      data.value = result.generations.length ? result : null;
      state.value = data.value ? "ready" : "empty";
    } catch (e) {
      if (active !== request) return;
      errorMsg.value = e instanceof Error ? e.message : String(e);
      state.value = data.value ? "ready" : "error";
    } finally {
      if (active === request) {
        active = null;
        refreshing.value = false;
        if (pending) void retry();
      }
    }
  }

  watch([key, () => query().revision], () => void retry(), { immediate: true });
  onScopeDispose(() => {
    active?.controller.abort();
    active = null;
    pending = false;
  });
  return { data, state, errorMsg, refreshing, retry };
}
