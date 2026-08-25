import { defineStore, acceptHMRUpdate } from "pinia";
import { ref, computed } from "vue";
import type { CampaignInfoDto } from "../../types/simlab";
import { getAllCampaigns } from "../../api/campaigns";
import { matchesQuery } from "../../utils/textSearch";

export const useCampaignsStore = defineStore("campaigns", () => {
  const campaigns = ref<CampaignInfoDto[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const searchQuery = ref("");

  const filtered = computed(() =>
    campaigns.value.filter((c) => matchesQuery(c.name, searchQuery.value)),
  );

  async function fetchAll() {
    loading.value = true;
    error.value = null;
    try {
      campaigns.value = await getAllCampaigns();
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      loading.value = false;
    }
  }

  function remove(id: string) {
    campaigns.value = campaigns.value.filter((c) => c.id !== id);
  }

  return { campaigns, loading, error, searchQuery, filtered, fetchAll, remove };
});

// See experimentsStore: keeps newly added actions available under Vite HMR.
if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useCampaignsStore, import.meta.hot));
}
