import client from "./client";
import type {
  ExperimentCreateDto,
  ExperimentDto,
  ExperimentFullDto,
  ExperimentInfoDto,
  ExperimentStatus,
  RuntimeMetricsSeriesResponseDto,
} from "../types/simlab";

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
  // Single listing endpoint: the status comes from the backend, so experiments
  // never silently vanish if a new status value is introduced server-side.
  const { data } = await client.get<ExperimentInfoDto[]>("/experiments/");
  return data;
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

export async function createExperiment(
  payload: ExperimentCreateDto,
): Promise<string> {
  const { data } = await client.post<string>("/experiments/", payload);
  return data;
}

export async function getRuntimeMetricsSeries(
  id: string,
  maxPoints = 1000,
): Promise<RuntimeMetricsSeriesResponseDto> {
  const { data } = await client.get<RuntimeMetricsSeriesResponseDto>(
    `/experiments/${id}/runtime-metrics`,
    { params: { max_points: maxPoints } },
  );
  return data;
}

export async function plotParetoResults(
  id: string,
  objectives: string[],
  minimize: boolean[],
): Promise<{ status: string; output: string }> {
  const { data } = await client.post<{ status: string; output: string }>(
    `/experiments/${id}/plot-pareto`,
    { objectives, minimize },
  );
  return data;
}
