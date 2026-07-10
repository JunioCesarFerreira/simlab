import { defineStore, acceptHMRUpdate } from "pinia";
import { ref, computed } from "vue";
import type { ExperimentInfoDto, ExperimentStatus } from "../../types/simlab";
import {
  getAllExperiments,
  getExperimentsByStatus,
  deleteExperiment,
} from "../../api/experiments";

export const useExperimentsStore = defineStore("experiments", () => {
  const experiments = ref<ExperimentInfoDto[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const selectedStatus = ref<ExperimentStatus | null>(null);

  const filtered = computed(() => {
    if (!selectedStatus.value) return experiments.value;
    return experiments.value.filter((e) => e.status === selectedStatus.value);
  });

  const countByStatus = computed(() => {
    const counts: Record<string, number> = {};
    for (const e of experiments.value) {
      counts[e.status] = (counts[e.status] ?? 0) + 1;
    }
    return counts;
  });

  async function fetchAll() {
    loading.value = true;
    error.value = null;
    try {
      experiments.value = await getAllExperiments();
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  async function fetchByStatus(status: ExperimentStatus) {
    loading.value = true;
    error.value = null;
    try {
      experiments.value = await getExperimentsByStatus(status);
      selectedStatus.value = status;
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  function setFilter(status: ExperimentStatus | null) {
    selectedStatus.value = status;
  }

  /**
   * Delete an experiment (and all its owned artifacts, server-side) and drop
   * it from the local list on success.
   */
  async function remove(id: string): Promise<boolean> {
    const ok = await deleteExperiment(id);
    if (ok) {
      experiments.value = experiments.value.filter((e) => e.id !== id);
    }
    return ok;
  }

  return {
    experiments,
    loading,
    error,
    selectedStatus,
    filtered,
    countByStatus,
    fetchAll,
    fetchByStatus,
    setFilter,
    remove,
  };
});

// Keep the store in sync when it is hot-reloaded during development. Without
// this, Vite HMR retains the previously instantiated store and newly added
// actions (e.g. `remove`) appear missing until a full page reload.
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useExperimentsStore, import.meta.hot));
}
