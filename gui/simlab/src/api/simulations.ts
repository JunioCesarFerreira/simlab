import client from "./client";
import type { SimulationDto } from "../types/simlab";

export async function getSimulation(id: string): Promise<SimulationDto> {
  const { data } = await client.get<SimulationDto>(`/simulations/${id}`);
  return data;
}

export async function getSimulationsByIndividual(
  individualId: string,
): Promise<SimulationDto[]> {
  const { data } = await client.get<SimulationDto[]>(
    `/simulations/by-individual/${individualId}`,
  );
  return data;
}
