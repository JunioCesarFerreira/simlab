import client from "./client";
import type {
  MilpEngineStatusDto,
  MilpModelDto,
  MilpModelInfoDto,
  MilpSweepCreatePayload,
  MilpSweepDto,
  MilpSweepInfoDto,
} from "../types/milp";

export async function getMilpModels(): Promise<MilpModelInfoDto[]> {
  const { data } = await client.get<MilpModelInfoDto[]>("/milp/models");
  return data;
}

export async function getMilpModel(key: string): Promise<MilpModelDto> {
  const { data } = await client.get<MilpModelDto>(`/milp/models/${key}`);
  return data;
}

export async function getMilpEngineStatus(): Promise<MilpEngineStatusDto> {
  const { data } = await client.get<MilpEngineStatusDto>("/milp/status");
  return data;
}

export async function createMilpSweep(payload: MilpSweepCreatePayload): Promise<string> {
  const { data } = await client.post<string>("/milp/sweeps", payload);
  return data;
}

export async function listMilpSweeps(): Promise<MilpSweepInfoDto[]> {
  const { data } = await client.get<MilpSweepInfoDto[]>("/milp/sweeps");
  return data;
}

export async function getMilpSweep(id: string): Promise<MilpSweepDto> {
  const { data } = await client.get<MilpSweepDto>(`/milp/sweeps/${id}`);
  return data;
}

export async function cancelMilpSweep(id: string): Promise<boolean> {
  const { data } = await client.patch<boolean>(`/milp/sweeps/${id}/cancel`);
  return data;
}

export async function deleteMilpSweep(id: string): Promise<boolean> {
  const { data } = await client.delete<boolean>(`/milp/sweeps/${id}`);
  return data;
}
