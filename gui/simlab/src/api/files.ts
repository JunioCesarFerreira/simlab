import type { ID } from "../types/simlab";

const baseUrl = import.meta.env.VITE_API_BASE_URL;
const apiKey = import.meta.env.VITE_API_KEY;

export function topologyUrl(simulationId: ID): string {
  return `${baseUrl}/files/simulations/${simulationId}/topology?X-API-Key=${apiKey}`;
}

export function fileDownloadUrl(fileId: ID, extension: string): string {
  return `${baseUrl}/files/${fileId}/as/${extension}?X-API-Key=${apiKey}`;
}

export function experimentAnalysisZipUrl(experimentId: ID): string {
  return `${baseUrl}/files/experiments/${experimentId}/analysis/zip?X-API-Key=${apiKey}`;
}

export function experimentTopologiesZipUrl(experimentId: ID): string {
  return `${baseUrl}/files/experiments/${experimentId}/topologies/zip?X-API-Key=${apiKey}`;
}
