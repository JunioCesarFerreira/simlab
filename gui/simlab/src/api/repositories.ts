import client from "./client";
import type { SourceRepositoryDto } from "../types/simlab";

export async function getAllRepositories(): Promise<SourceRepositoryDto[]> {
  const { data } = await client.get<SourceRepositoryDto[]>("/sources/");
  return data;
}

export async function getRepository(id: string): Promise<SourceRepositoryDto> {
  const { data } = await client.get<SourceRepositoryDto>(`/sources/${id}`);
  return data;
}
