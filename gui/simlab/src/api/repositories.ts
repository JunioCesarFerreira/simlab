import client from "./client";
import type { SourceRepositoryDto } from "../types/simlab";

const baseUrl = import.meta.env.VITE_API_BASE_URL as string;
const apiKey = import.meta.env.VITE_API_KEY as string;

function authHeaders(): HeadersInit {
  return { "X-API-Key": apiKey };
}

async function fetchChecked(input: RequestInfo, init?: RequestInit): Promise<Response> {
  const res = await fetch(input, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res;
}

// ── Read ─────────────────────────────────────────────────────────────────────

export async function getAllRepositories(): Promise<SourceRepositoryDto[]> {
  const { data } = await client.get<SourceRepositoryDto[]>("/sources/");
  return data;
}

export async function getRepository(id: string): Promise<SourceRepositoryDto> {
  const { data } = await client.get<SourceRepositoryDto>(`/sources/${id}`);
  return data;
}

// ── Create ───────────────────────────────────────────────────────────────────

export async function createRepository(
  name: string,
  description: string,
  files: File[],
): Promise<string> {
  const form = new FormData();
  form.append("name", name);
  form.append("description", description);
  for (const f of files) form.append("files", f);

  const res = await fetchChecked(`${baseUrl}/sources/`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  // API returns a quoted JSON string e.g. "\"abc123\""
  const text = await res.text();
  return text.replace(/^"|"$/g, "");
}

// ── Update ───────────────────────────────────────────────────────────────────

export async function updateRepository(
  id: string,
  updates: { name?: string; description?: string },
): Promise<boolean> {
  const { data } = await client.patch<boolean>(`/sources/${id}`, updates);
  return data;
}

// ── File management ──────────────────────────────────────────────────────────

export async function addFilesToRepository(
  id: string,
  files: File[],
): Promise<boolean> {
  const form = new FormData();
  for (const f of files) form.append("files", f);

  await fetchChecked(`${baseUrl}/sources/${id}/files`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  return true;
}

export async function getFileContent(
  repositoryId: string,
  fileId: string,
): Promise<string> {
  const res = await fetchChecked(
    `${baseUrl}/sources/${repositoryId}/files/${fileId}/content`,
    { headers: authHeaders() },
  );
  return res.text();
}

export async function removeFileFromRepository(
  repositoryId: string,
  fileId: string,
): Promise<boolean> {
  const { data } = await client.delete<boolean>(
    `/sources/${repositoryId}/files/${fileId}`,
  );
  return data;
}

// ── Delete ───────────────────────────────────────────────────────────────────

export async function deleteRepository(id: string): Promise<boolean> {
  const { data } = await client.delete<boolean>(`/sources/${id}`);
  return data;
}

// ── Download ─────────────────────────────────────────────────────────────────

export async function downloadRepository(
  id: string,
  repoName: string,
): Promise<void> {
  const res = await fetchChecked(`${baseUrl}/sources/${id}/download`, {
    headers: authHeaders(),
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${repoName || id}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}
