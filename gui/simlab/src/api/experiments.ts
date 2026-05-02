import client from "./client";
import type {
  ExperimentDto,
  ExperimentFullDto,
  ExperimentInfoDto,
  ExperimentStatus,
} from "../types/simlab";

const ALL_STATUSES: ExperimentStatus[] = ["Waiting", "Running", "Done", "Error"];

type BackendExperimentInfo = Omit<ExperimentInfoDto, "status">;

export async function getExperimentsByStatus(
  status: ExperimentStatus,
): Promise<ExperimentInfoDto[]> {
  const { data } = await client.get<BackendExperimentInfo[]>(
    `/experiments/by-status/${status}`,
  );
  return data.map((e) => ({ ...e, status }));
}

export async function getAllExperiments(): Promise<ExperimentInfoDto[]> {
  const results = await Promise.all(
    ALL_STATUSES.map((s) => getExperimentsByStatus(s)),
  );
  return results.flat();
}

export async function getExperiment(id: string): Promise<ExperimentDto> {
  const { data } = await client.get<ExperimentDto>(`/experiments/${id}`);
  return data;
}

export async function getExperimentFull(
  id: string,
): Promise<ExperimentFullDto> {
  const { data } = await client.get<ExperimentFullDto>(
    `/experiments/${id}/full`,
  );
  return data;
}

export async function updateExperiment(
  id: string,
  updates: { name?: string },
): Promise<boolean> {
  const { data } = await client.put<boolean>(`/experiments/${id}`, updates);
  return data;
}

export async function deleteExperiment(id: string): Promise<boolean> {
  const { data } = await client.delete<boolean>(`/experiments/${id}`);
  return data;
}
