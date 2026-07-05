import client from "./client";
import type { ProblemDraft } from "../types/problem";

export interface ProblemInfoDto {
  id: string;
  name: string;
  created_time: string;
  updated_time: string;
  has_background: boolean;
}

export interface ProblemDto {
  id: string;
  name: string;
  created_time: string;
  updated_time: string;
  draft: ProblemDraft;
  background_image_id: string | null;
  image_world_bounds: [number, number, number, number] | null;
}

export async function listProblems(): Promise<ProblemInfoDto[]> {
  const { data } = await client.get<ProblemInfoDto[]>("/problems/");
  return data;
}

export async function createProblem(
  name: string,
  draft: ProblemDraft,
  imageWorldBounds?: [number, number, number, number] | null,
): Promise<string> {
  const { data } = await client.post<string>("/problems/", {
    name,
    draft,
    image_world_bounds: imageWorldBounds ?? null,
  });
  return data;
}

export async function getProblem(id: string): Promise<ProblemDto> {
  const { data } = await client.get<ProblemDto>(`/problems/${id}`);
  return data;
}

export async function updateProblem(
  id: string,
  updates: { name?: string; draft?: ProblemDraft; image_world_bounds?: [number, number, number, number] | null },
): Promise<boolean> {
  const { data } = await client.put<boolean>(`/problems/${id}`, updates);
  return data;
}

export async function deleteProblem(id: string): Promise<boolean> {
  const { data } = await client.delete<boolean>(`/problems/${id}`);
  return data;
}

/** Upload a background image (data URL) for a saved problem. Returns the GridFS file ID. */
export async function uploadBackground(
  problemId: string,
  dataUrl: string,
): Promise<string> {
  const commaIdx = dataUrl.indexOf(",");
  const header = dataUrl.substring(0, commaIdx);
  const base64Data = dataUrl.substring(commaIdx + 1);
  const mimeType = header.match(/data:([^;]+)/)?.[1] ?? "image/png";
  const ext = mimeType.split("/")[1] ?? "png";

  const binary = atob(base64Data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  const formData = new FormData();
  formData.append("file", new Blob([bytes], { type: mimeType }), `background.${ext}`);

  const { data } = await client.put<string>(`/problems/${problemId}/background`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/** Download a problem's background image and return it as a data URL. */
export async function downloadBackground(problemId: string): Promise<string> {
  const response = await client.get<ArrayBuffer>(`/problems/${problemId}/background`, {
    responseType: "arraybuffer",
  });
  const contentType = (response.headers as Record<string, string>)["content-type"] || "image/png";
  const blob = new Blob([response.data], { type: contentType });
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}
