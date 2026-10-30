/**
 * Scalability benchmarks: create one, list them, read one's analysis.
 *
 * Created through its own route rather than `POST /experiments`, because the
 * benchmark block is a contract of its own and the generic experiment body is
 * typed — a key it does not declare could be dropped on the way in.
 */
import client from "./client";
import type {
  BenchmarkCreateInput,
  BenchmarkCreated,
  BenchmarkDetail,
  BenchmarkSummary,
} from "../types/scalability";

export async function listBenchmarks(signal?: AbortSignal): Promise<BenchmarkSummary[]> {
  const { data } = await client.get<BenchmarkSummary[]>("/scalability/", { signal });
  return data;
}

export async function getBenchmark(id: string, signal?: AbortSignal): Promise<BenchmarkDetail> {
  const { data } = await client.get<BenchmarkDetail>(`/scalability/${id}`, { signal });
  return data;
}

export async function createBenchmark(input: BenchmarkCreateInput): Promise<BenchmarkCreated> {
  const { data } = await client.post<BenchmarkCreated>("/scalability/", input);
  return data;
}
