/**
 * Infrastructure client: hosts, Cooja containers and pool occupancy.
 *
 * Two endpoints behave differently from the rest and the difference matters to
 * the caller:
 *
 *  - `requestCheck` and `requestDiscovery` answer **202**. The API cannot
 *    reach the containers and holds no private key, so it records the request
 *    and the master-node carries it out. The caller polls for the result
 *    instead of reading it from the response.
 *  - `deleteHost` answers **409** while anything is running on the host.
 *    That is a normal outcome to show, not an error to swallow.
 */
import client from "./client";
import type {
  AcceptedDto,
  HostCreateInput,
  HostDetailDto,
  HostDto,
  HostUpdateInput,
  OverviewDto,
  PublicKeyDto,
  ResourceCreateInput,
  ResourceDto,
  ResourceSimulationDto,
  ResourceUpdateInput,
  TopologyDto,
} from "../types/infra";

// ── Crypto ───────────────────────────────────────────────────────────────────

/** The key to seal a credential against. Served by the API, owned by the master-node. */
export async function getPublicKey(): Promise<PublicKeyDto> {
  const { data } = await client.get<PublicKeyDto>("/infra/crypto/public-key");
  return data;
}

// ── Overview ─────────────────────────────────────────────────────────────────

/** Pool occupancy, from the same snapshot the scheduler reads. */
export async function getOverview(signal?: AbortSignal): Promise<OverviewDto> {
  const { data } = await client.get<OverviewDto>("/infra/overview", { signal });
  return data;
}

// ── Hosts ────────────────────────────────────────────────────────────────────

export async function listHosts(signal?: AbortSignal): Promise<HostDto[]> {
  const { data } = await client.get<HostDto[]>("/infra/hosts", { signal });
  return data;
}

export async function getHost(id: string, signal?: AbortSignal): Promise<HostDetailDto> {
  const { data } = await client.get<HostDetailDto>(`/infra/hosts/${id}`, { signal });
  return data;
}

/** Register a remote host. The credential must already be sealed. */
export async function createHost(host: HostCreateInput): Promise<string> {
  const { data } = await client.post<string>("/infra/hosts", host);
  return data;
}

/** Change a host. Only the fields present are touched. */
export async function updateHost(id: string, updates: HostUpdateInput): Promise<HostDto> {
  const { data } = await client.put<HostDto>(`/infra/hosts/${id}`, updates);
  return data;
}

/**
 * Take a host in or out of the pool.
 *
 * Disabling drains rather than interrupts: simulations already running are
 * left to finish, and the host simply stops receiving new ones.
 */
export async function setHostEnabled(id: string, enabled: boolean): Promise<HostDto> {
  const action = enabled ? "enable" : "disable";
  const { data } = await client.post<HostDto>(`/infra/hosts/${id}/${action}`);
  return data;
}

/** Remove a host. Rejected with 409 while anything is running on it. */
export async function deleteHost(id: string): Promise<void> {
  await client.delete(`/infra/hosts/${id}`);
}

// ── Requests the master-node carries out ─────────────────────────────────────

/** Ask the master-node to probe a host. Answers 202; poll `last_check`. */
export async function requestCheck(id: string): Promise<AcceptedDto> {
  const { data } = await client.post<AcceptedDto>(`/infra/hosts/${id}/check`);
  return data;
}

/** Ask the master-node to enumerate a host's containers. Answers 202. */
export async function requestDiscovery(id: string): Promise<AcceptedDto> {
  const { data } = await client.post<AcceptedDto>(`/infra/hosts/${id}/discover`);
  return data;
}

// ── Resources ────────────────────────────────────────────────────────────────

export async function listResources(signal?: AbortSignal): Promise<ResourceDto[]> {
  const { data } = await client.get<ResourceDto[]>("/infra/resources", { signal });
  return data;
}

/**
 * Take one container in or out of the pool.
 *
 * `enabled` is the only field a client may write: everything else about a
 * container is discovered or measured.
 */
export async function setResourceEnabled(id: string, enabled: boolean): Promise<ResourceDto> {
  const { data } = await client.put<ResourceDto>(`/infra/resources/${id}`, {
    enabled,
  });
  return data;
}

/**
 * Register one worker on a host, with its own SSH parameters.
 *
 * The counterpart to Docker discovery, and the only way that works on a host
 * whose SSH user cannot reach its Docker daemon.
 */
export async function createResource(
  hostId: string,
  resource: ResourceCreateInput,
): Promise<ResourceDto> {
  const { data } = await client.post<ResourceDto>(`/infra/hosts/${hostId}/resources`, resource);
  return data;
}

/** Change a worker's address or SSH parameters. Only what is sent is touched. */
export async function updateResourceConnection(
  id: string,
  updates: ResourceUpdateInput,
): Promise<ResourceDto> {
  const { data } = await client.put<ResourceDto>(`/infra/resources/${id}/connection`, updates);
  return data;
}

/** Remove one worker. Rejected with 409 while a simulation is on it. */
export async function deleteResource(id: string): Promise<void> {
  await client.delete(`/infra/resources/${id}`);
}

/** What is running on a container, if anything. */
export async function getResourceSimulation(id: string): Promise<ResourceSimulationDto> {
  const { data } = await client.get<ResourceSimulationDto>(`/infra/resources/${id}/simulation`);
  return data;
}

/**
 * The pool as a graph — master-node, hosts, workers — and what runs on each
 * worker. One request for the whole picture; the page polls it while
 * experiments run.
 */
export async function getTopology(signal?: AbortSignal): Promise<TopologyDto> {
  const { data } = await client.get<TopologyDto>("/infra/topology", { signal });
  return data;
}
