import { defineStore } from "pinia";
import { ref, shallowRef, computed } from "vue";
import type { ExperimentFullDto } from "../../types/simlab";
import { getExperimentFull } from "../../api/experiments";
import { stableStringify } from "../../utils/stableStringify";

export const useExperimentDetailStore = defineStore("experimentDetail", () => {
  // API snapshots are replaced as a whole; proxying every chromosome and point
  // adds substantial traversal overhead to large chart datasets.
  const experiment = shallowRef<ExperimentFullDto | null>(null);
  const revision = ref(0);
  const loading = ref(false);
  const error = ref<string | null>(null);
  let _pollInterval: ReturnType<typeof setTimeout> | null = null;
  let _pollVersion = 0;
  let _request: { id: string; controller: AbortController; promise: Promise<void> } | null = null;
  // Serialized form of the last payload assigned to `experiment`. Polling
  // replaces the whole object every tick; when the backend returns identical
  // data that would still invalidate every computed downstream (including the
  // O(n²) non-dominated sort) and re-render all charts. Comparing the raw
  // JSON first makes an idle poll tick cost one stringify instead.
  let _lastPayload = "";

  function setExperiment(data: ExperimentFullDto): void {
    const payload = JSON.stringify(data);
    if (payload === _lastPayload) return;
    _lastPayload = payload;
    experiment.value = data;
    revision.value++;
  }

  const isRunning = computed(
    () =>
      experiment.value?.status === "Running" ||
      experiment.value?.status === "Waiting",
  );

  const objectiveNames = computed(
    () => experiment.value?.parameters?.objectives?.map((o) => o.metric_name) ?? [],
  );

  const objectiveGoals = computed(
    () => experiment.value?.parameters?.objectives?.map((o) => o.goal) ?? [],
  );

  // chromosome (stable-stringified) -> individual_id. `pareto_front` items only
  // carry {chromosome, objectives} (no individual_id), so every chart needs this
  // reverse lookup to resolve clicks/pins/tooltips. Computed once here instead
  // of independently inside each of the 2D/3D/parallel-coordinates charts.
  const chromosomeToIndividualId = computed<Map<string, string>>(() => {
    const map = new Map<string, string>();
    for (const gen of experiment.value?.generations ?? []) {
      for (const ind of gen.population) {
        map.set(stableStringify(ind.chromosome), ind.individual_id);
      }
    }
    return map;
  });

  function load(id: string, foreground: boolean): Promise<void> {
    if (_request?.id === id) return _request.promise;
    _request?.controller.abort();
    if (experiment.value?.id !== id) {
      experiment.value = null;
      _lastPayload = "";
    }
    if (foreground) {
      loading.value = true;
      error.value = null;
    }
    const controller = new AbortController();
    const promise = getExperimentFull(id, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setExperiment(data);
      })
      .catch((e: unknown) => {
        if (!controller.signal.aborted && foreground) {
          error.value = e instanceof Error ? e.message : String(e);
        }
      })
      .finally(() => {
        if (_request?.controller === controller) {
          _request = null;
          loading.value = false;
        }
      });
    _request = { id, controller, promise };
    return promise;
  }

  function fetch(id: string) {
    return load(id, true);
  }

  function refresh(id: string) {
    return load(id, false);
  }

  function startPolling(id: string, intervalMs = 3000) {
    stopPolling();
    const version = _pollVersion;
    const schedule = () => {
      if (version !== _pollVersion || !isRunning.value) return;
      _pollInterval = setTimeout(async () => {
        // A slow response postpones the next poll, instead of adding requests.
        if (typeof document === "undefined" || !document.hidden) await refresh(id);
        schedule();
      }, intervalMs);
    };
    schedule();
  }

  function stopPolling() {
    _pollVersion++;
    if (_pollInterval !== null) {
      clearTimeout(_pollInterval);
      _pollInterval = null;
    }
  }

  function clear() {
    stopPolling();
    _request?.controller.abort();
    _request = null;
    experiment.value = null;
    error.value = null;
    loading.value = false;
    _lastPayload = "";
  }

  return {
    experiment,
    revision,
    loading,
    error,
    isRunning,
    objectiveNames,
    objectiveGoals,
    chromosomeToIndividualId,
    fetch,
    refresh,
    startPolling,
    stopPolling,
    clear,
  };
});
