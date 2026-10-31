/**
 * Seed studies: create one, list them, read one's analysis.
 *
 * Created through its own route rather than `POST /experiments`, for the same
 * reason as the scalability benchmark: the generic experiment body is typed,
 * and a key it does not declare could be dropped on the way in.
 */
import client from "./client";
import type {
  MetricDetail,
  StudyCreateInput,
  StudyCreated,
  StudyDetail,
  StudyEstimate,
  StudySummary,
  StudyTargets,
} from "../types/seedStudy";

export async function listStudies(signal?: AbortSignal): Promise<StudySummary[]> {
  const { data } = await client.get<StudySummary[]>("/seed-studies/", { signal });
  return data;
}

export async function getStudy(
  id: string,
  targets?: StudyTargets | null,
  signal?: AbortSignal,
): Promise<StudyDetail> {
  const { data } = await client.get<StudyDetail>(`/seed-studies/${id}`, {
    params: targets ?? undefined,
    signal,
  });
  return data;
}

export async function getStudyMetric(
  id: string,
  metric: string,
  chromosome: string,
  targets?: StudyTargets | null,
  signal?: AbortSignal,
): Promise<MetricDetail> {
  const { data } = await client.get<MetricDetail>(
    `/seed-studies/${id}/metrics/${encodeURIComponent(metric)}`,
    {
      params: {
        chromosome,
        confidence: targets?.confidence,
        relative_error: targets?.relative_error,
      },
      signal,
    },
  );
  return data;
}

export async function createStudy(input: StudyCreateInput): Promise<StudyCreated> {
  const { data } = await client.post<StudyCreated>("/seed-studies/", input);
  return data;
}

export async function getStudyEstimate(duration: number): Promise<StudyEstimate> {
  const { data } = await client.get<StudyEstimate>("/seed-studies/estimate", {
    params: { duration },
  });
  return data;
}

/** The values as text: the route needs the API key, so a plain link cannot reach it. */
export async function getStudyCsv(id: string): Promise<string> {
  const { data } = await client.get<string>(`/seed-studies/${id}/values.csv`, {
    responseType: "text",
    transformResponse: [(value) => value],
  });
  return data;
}
