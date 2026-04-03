import { defineStore } from "pinia";
import { ref, computed } from "vue";
import type { ExperimentInfoDto, ExperimentStatus } from "../../types/simlab";
import {
  getAllExperiments,
  getExperimentsByStatus,
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
  };
});
