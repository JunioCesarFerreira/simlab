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

/**
 * Open a tab NOW, while still inside the user-gesture call stack, and point it
 * at the blob once the download finishes. Calling window.open after an await
 * loses the gesture and gets blocked by popup blockers.
 */
async function openAsWindow(fetching: Promise<Blob>): Promise<void> {
  const win = window.open("about:blank", "_blank");
  try {
    const blob = await fetching;
    const url = URL.createObjectURL(blob);
    if (win) {
      win.location = url;
    } else {
      // Opening was blocked anyway (e.g. triggered outside a user gesture).
      window.open(url, "_blank");
    }
    // Revoke after a delay to give the browser time to load the URL
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  } catch (e) {
    win?.close();
    throw e;
  }
}

// -------------------------------------------------------
// Public API
// -------------------------------------------------------

export async function fetchBlobUrl(path: string): Promise<string> {
  const blob = await fetchBlob(`${baseUrl}${path}`);
  return URL.createObjectURL(blob);
}

export async function openTopology(individualTopologyPictureId: ID): Promise<void> {
  const url = `${baseUrl}/files/${individualTopologyPictureId}/as/png`;
  await openAsWindow(fetchBlob(url));
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
