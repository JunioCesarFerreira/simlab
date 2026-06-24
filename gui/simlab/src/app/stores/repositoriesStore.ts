import { defineStore } from "pinia";
import { ref } from "vue";
import type { SourceRepositoryDto } from "../../types/simlab";
import { getAllRepositories, deleteRepository } from "../../api/repositories";

export const useRepositoriesStore = defineStore("repositories", () => {
  const repositories = ref<SourceRepositoryDto[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

  async function fetchAll() {
    loading.value = true;
    error.value = null;
    try {
      repositories.value = await getAllRepositories();
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  async function remove(id: string): Promise<boolean> {
    try {
      await deleteRepository(id);
      repositories.value = repositories.value.filter((r) => r.id !== id);
      return true;
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e);
      return false;
    }
  }

  return { repositories, loading, error, fetchAll, remove };
});
