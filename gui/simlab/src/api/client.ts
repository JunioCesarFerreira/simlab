import axios, { AxiosError } from "axios";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  // A stuck backend must not leave pages in `loading` forever.
  timeout: 30_000,
  headers: {
    "X-API-Key": import.meta.env.VITE_API_KEY,
    "Content-Type": "application/json",
  },
});

/**
 * Normalize errors so callers can show `error.message` directly:
 * - FastAPI puts the useful text in `response.data.detail` — surface it
 *   instead of axios' generic "Request failed with status code 500".
 * - Timeouts and network failures get human-readable messages.
 */
client.interceptors.response.use(undefined, (error: unknown) => {
  if (error instanceof AxiosError) {
    const detail = (error.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === "string" && detail.trim()) {
      error.message = `${error.response?.status ?? ""} ${detail}`.trim();
    } else if (error.code === "ECONNABORTED") {
      error.message = "The server took too long to respond (timeout).";
    } else if (error.code === "ERR_NETWORK") {
      error.message = "Could not reach the server — is the API running?";
    }
  }
  return Promise.reject(error);
});

export default client;
