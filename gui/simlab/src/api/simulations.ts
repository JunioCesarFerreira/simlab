import client from "./client";
import type { SimulationDto } from "../types/simlab";

export async function getSimulation(id: string): Promise<SimulationDto> {
  const { data } = await client.get<SimulationDto>(`/simulations/${id}`);
  return data;
}

/**
 * Simulations of one individual, within one experiment.
 *
 * `experimentId` is required on purpose. The individual id is a chromosome
 * hash and the same chromosome recurs across experiments, so omitting the
 * scope lists every run of that chromosome ever made — which showed an
 * individual's panel nine simulations when two belonged to the experiment on
 * screen.
 */
export async function getSimulationsByIndividual(
  individualId: string,
  experimentId: string,
): Promise<SimulationDto[]> {
  const { data } = await client.get<SimulationDto[]>(
    `/simulations/by-individual/${individualId}`,
    { params: { experiment_id: experimentId } },
  );
  return data;
}
