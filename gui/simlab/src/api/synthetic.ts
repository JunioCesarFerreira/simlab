import client from "./client";

export interface BenchmarkInfo {
  id: string;
  label: string;
  min_objectives: number;
  max_objectives: number | null;
  description: string;
  n_min_formula: string;
}

export async function getBenchmarks(): Promise<BenchmarkInfo[]> {
  const { data } = await client.get<BenchmarkInfo[]>("/synthetic/benchmarks");
  return data;
}
