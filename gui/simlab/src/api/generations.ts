import client from "./client";
import type { GenerationDto } from "../types/simlab";

export async function getGenerationsByExperiment(
  experimentId: string,
): Promise<GenerationDto[]> {
  const { data } = await client.get<GenerationDto[]>(
    `/generations/by-experiment/${experimentId}`,
  );
  return data;
}

export async function getGeneration(id: string): Promise<GenerationDto> {
  const { data } = await client.get<GenerationDto>(`/generations/${id}`);
  return data;
}
