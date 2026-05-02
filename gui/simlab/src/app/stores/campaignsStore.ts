import { defineStore } from "pinia";
import { ref } from "vue";
import type { CampaignInfoDto } from "../../types/simlab";
import { getAllCampaigns } from "../../api/campaigns";

export const useCampaignsStore = defineStore("campaigns", () => {
  const campaigns = ref<CampaignInfoDto[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

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

  return { campaigns, loading, error, fetchAll, remove };
});
