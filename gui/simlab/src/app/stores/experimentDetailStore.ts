import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { ExperimentFullDto } from "../../types/simlab";
import { getExperimentFull } from "../../api/experiments";

export const useExperimentDetailStore = defineStore("experimentDetail", () => {
  const experiment = ref<ExperimentFullDto | null>(null);
  const loading = ref(false);
  const error = ref<string | null>(null);
  let _pollInterval: ReturnType<typeof setInterval> | null = null;

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

  async function fetch(id: string) {
    loading.value = true;
    error.value = null;
    try {
      experiment.value = await getExperimentFull(id);
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  async function refresh(id: string) {
    try {
      experiment.value = await getExperimentFull(id);
    } catch {
      // silently ignore polling errors
    }
  }

  function startPolling(id: string, intervalMs = 3000) {
    stopPolling();
    _pollInterval = setInterval(async () => {
      if (isRunning.value) {
        await refresh(id);
      } else {
        stopPolling();
      }
    }, intervalMs);
  }

  function stopPolling() {
    if (_pollInterval !== null) {
      clearInterval(_pollInterval);
      _pollInterval = null;
    }
  }

  function clear() {
    stopPolling();
    experiment.value = null;
    error.value = null;
    loading.value = false;
  }

  return {
    experiment,
    loading,
    error,
    isRunning,
    objectiveNames,
    objectiveGoals,
    fetch,
    refresh,
    startPolling,
    stopPolling,
    clear,
  };
});
