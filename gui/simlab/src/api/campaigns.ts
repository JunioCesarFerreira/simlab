import client from "./client";
import type { CampaignDto, CampaignFullDto, CampaignInfoDto } from "../types/simlab";

export async function getAllCampaigns(): Promise<CampaignInfoDto[]> {
  const { data } = await client.get<CampaignInfoDto[]>("/campaigns/");
  return data;
}

export async function getCampaign(id: string): Promise<CampaignDto> {
  const { data } = await client.get<CampaignDto>(`/campaigns/${id}`);
  return data;
}

export async function getCampaignFull(id: string): Promise<CampaignFullDto> {
  const { data } = await client.get<CampaignFullDto>(`/campaigns/${id}/full`);
  return data;
}

export async function createCampaign(
  payload: Omit<CampaignDto, "id">,
): Promise<string> {
  const { data } = await client.post<string>("/campaigns/", payload);
  return data;
}

export async function updateCampaign(
  id: string,
  updates: { name?: string; description?: string; created_time?: string | null },
): Promise<boolean> {
  const { data } = await client.put<boolean>(`/campaigns/${id}`, updates);
  return data;
}

export async function deleteCampaign(id: string): Promise<boolean> {
  const { data } = await client.delete<boolean>(`/campaigns/${id}`);
  return data;
}

export async function addExperimentToCampaign(
  campaignId: string,
  experimentId: string,
): Promise<boolean> {
  const { data } = await client.patch<boolean>(
    `/campaigns/${campaignId}/experiments/${experimentId}`,
  );
  return data;
}

export async function removeExperimentFromCampaign(
  campaignId: string,
  experimentId: string,
): Promise<boolean> {
  const { data } = await client.delete<boolean>(
    `/campaigns/${campaignId}/experiments/${experimentId}`,
  );
  return data;
}
