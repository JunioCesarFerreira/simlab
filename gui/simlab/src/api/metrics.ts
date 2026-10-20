import client from "./client";

export type Population = "survivors" | "offspring" | "archive";
export type PopulationSource = Population | "mixed";

export interface HvGdData {
  generations: number[];
  hv: number[];
  hv_cumulative: number[];
  gd: (number | null)[];
  igd: (number | null)[];
  igd_plus: (number | null)[];
  reference: "true_front" | "final_front" | null;
  reference_size: number;
  normalized: boolean;
  worst_point: Record<string, number>;
  population: Population | null;
  population_source: PopulationSource | null;
  // Aligned with generations; optional when connected to an older API.
  population_sources?: Population[];
  gd_method: "analytical" | "reference_front" | null;
  gd_formula: string | null;
  normalization: string | null;
}

export interface HvGdQuery {
  experimentId: string;
  objectiveNames: string[];
  objectiveGoals: string[];
  population: Population;
  revision?: number;
}

export async function getHvGd(query: HvGdQuery, signal: AbortSignal): Promise<HvGdData> {
  const params = new URLSearchParams();
  query.objectiveNames.forEach((o) => params.append("objectives", o));
  query.objectiveGoals.forEach((g) => params.append("minimize", String(g === "min")));
  params.append("population", query.population);
  // This chart renders the selected set only. The extra accumulated HV can
  // dwarf its cost in many objectives and is not displayed here.
  params.append("include_cumulative", "false");
  const { data } = await client.get<HvGdData>(
    `/experiments/${query.experimentId}/hv-gd`, { params, signal },
  );
  return data;
}
