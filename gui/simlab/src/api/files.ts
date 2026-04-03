import type { ID } from "../types/simlab";

const baseUrl = import.meta.env.VITE_API_BASE_URL;
const apiKey = import.meta.env.VITE_API_KEY;

function authHeaders(): HeadersInit {
  return { "X-API-Key": apiKey };
}

async function fetchBlob(url: string): Promise<Blob> {
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.blob();
}

function triggerDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function openBlob(blob: Blob): Promise<void> {
  const url = URL.createObjectURL(blob);
  window.open(url, "_blank");
  // Revoga após delay para dar tempo ao browser de abrir
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

// -------------------------------------------------------
// API pública
// -------------------------------------------------------

export async function fetchBlobUrl(path: string): Promise<string> {
  const blob = await fetchBlob(`${baseUrl}${path}`);
  return URL.createObjectURL(blob);
}

export async function openTopology(individualTopologyPictureId: ID): Promise<void> {
  const url = `${baseUrl}/files/${individualTopologyPictureId}/as/png`;
  const blob = await fetchBlob(url);
  await openBlob(blob);
}

export async function downloadFile(fileId: ID, extension: string): Promise<void> {
  const url = `${baseUrl}/files/${fileId}/as/${extension}`;
  const blob = await fetchBlob(url);
  triggerDownload(blob, `${fileId}.${extension}`);
}

export async function downloadAnalysisZip(experimentId: ID): Promise<void> {
  const url = `${baseUrl}/files/experiments/${experimentId}/analysis/zip`;
  const blob = await fetchBlob(url);
  triggerDownload(blob, `${experimentId}_analysis.zip`);
}

export async function downloadTopologiesZip(experimentId: ID): Promise<void> {
  const url = `${baseUrl}/files/experiments/${experimentId}/topologies/zip`;
  const blob = await fetchBlob(url);
  triggerDownload(blob, `${experimentId}_topologies.zip`);
}
