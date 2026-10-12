/**
 * Infrastructure: hosts, Cooja containers and pool occupancy.
 *
 * Mirrors `rest-api/api/domain/infra.py`. The asymmetry between what is sent
 * and what is received is deliberate on the server and reproduced here, so the
 * type system refuses to express a plaintext credential in a request body:
 *
 *  - outgoing, a credential is only ever a `SealedSecret`;
 *  - incoming, it is only ever a `SecretInfo` — that one exists, and against
 *    which key, never its contents.
 */

/** A credential sealed in the browser. Opaque by construction. */
export interface SealedSecret {
  key_id: string;
  ciphertext: string;
}

/** What the API says about a stored credential. Never its contents. */
export interface SecretInfo {
  has_secret: boolean;
  alg?: string;
  key_id?: string;
  updated_at?: string | null;
}

/** The master-node's public key — the only half that ever leaves it. */
export interface PublicKeyDto {
  key_id: string;
  alg: string;
  public_key: string;
  fingerprint: string;
  created_at: string | null;
  retired_key_ids: string[];
}

export type HostStatus =
  | "online"
  | "degraded"
  | "offline"
  | "misconfigured"
  | "disabled";

export type ResourceStatus =
  | "available"
  | "reserved"
  | "busy"
  | "offline"
  | "error"
  | "disabled";

export type AddressMode =
  | "container_ip"
  | "published_port"
  | "container_name"
  /** Stated per worker, not derived from Docker. */
  | "explicit";

/** Where a host's list of workers comes from. */
export type DiscoveryMode = "docker" | "manual";

export type AuthMethod = "key" | "password";

export interface HostAccessDto {
  address: string;
  port: number;
  username: string;
  auth_method: AuthMethod;
  secret: SecretInfo;
  key_fingerprint: string | null;
  /** Pinned on first contact; a change afterwards refuses the connection. */
  host_key_fingerprint: string | null;
}

export interface ContainerAccessDto {
  username: string;
  port: number;
  address_mode: AddressMode;
  auth_method: AuthMethod;
  secret: SecretInfo;
  remote_dir?: string;
}

export interface DiscoveryDto {
  mode: DiscoveryMode;
  label_filter: string;
  name_prefix: string | null;
  endpoint: string;
  last_discovery: string | null;
  discovered_count?: number;
  last_discovery_message?: string;
}

/** What Docker reported — the first of the two availability levels. */
export interface ContainerDto {
  id: string;
  name: string;
  image: string;
  state: string;
  checked_at: string | null;
}

/**
 * What the SSH probe found — the second level.
 *
 * `java_version` and `cooja_jar_sha256` are what make a mixed inventory
 * visible: a generation split across different Cooja builds can show up as a
 * host effect in the results.
 */
export interface ProbeDto {
  ssh_ok: boolean;
  java_version: string | null;
  cooja_jar_sha256: string | null;
  checked_at: string | null;
  message?: string;
}

/**
 * A worker's own SSH parameters, where they differ from its host's.
 *
 * Anything absent falls back to the host's `container_access` at connect
 * time, so one configuration per host covers the common case while a single
 * worker can still use a different user, port or credential.
 */
export interface ResourceConnectionDto {
  username?: string;
  port?: number;
  auth_method?: AuthMethod;
  secret?: SecretInfo;
}

export interface ResourceConnectionInput {
  username?: string;
  port?: number;
  auth_method?: AuthMethod;
  secret?: SealedSecret | null;
}

/** One worker, registered by hand rather than discovered. */
export interface ResourceCreateInput {
  name: string;
  /** Where this worker's sshd answers. Required: it is never guessed. */
  endpoint: { address: string; port?: number };
  connection?: ResourceConnectionInput;
  remote_dir?: string;
  enabled?: boolean;
}

export interface ResourceUpdateInput {
  name?: string;
  endpoint?: { address: string; port?: number };
  connection?: ResourceConnectionInput;
  remote_dir?: string;
  enabled?: boolean;
}

export interface ResourceDto {
  id: string;
  host_id: string;
  host_name?: string;
  name: string;
  status: ResourceStatus;
  enabled: boolean;
  container: ContainerDto;
  endpoint: Record<string, unknown>;
  connection: ResourceConnectionDto;
  remote_dir: string;
  probe: ProbeDto;
  current_simulation_id: string | null;
  remote_pid: number | null;
  reserved_at: string | null;
  started_at: string | null;
  origin: string;
  last_check: string | null;
  last_check_message: string;
}

export interface HostDto {
  id: string;
  name: string;
  type: "local" | "remote";
  enabled: boolean;
  status: HostStatus;
  access: HostAccessDto | null;
  container_access: ContainerAccessDto;
  discovery: DiscoveryDto;
  origin: string;
  last_check: string | null;
  last_check_message: string;
  consecutive_failures: number;
  /** The master-node still owes an answer to a requested check. */
  check_pending: boolean;
  discovery_pending: boolean;
  created_time: string | null;
  updated_time: string | null;
}

/**
 * One host's capacity, from the same snapshot the scheduler reads.
 *
 * `free`/`active` count containers; `schedulable` describes the host. They
 * are separate because an unreachable host still *has* its capacity.
 */
export interface HostOccupancyDto {
  id: string;
  name: string;
  type: string;
  status: HostStatus;
  enabled: boolean;
  schedulable: boolean;
  resource_total: number;
  free: number;
  active: number;
  unavailable: number;
  disabled: number;
  last_check: string | null;
  last_check_message: string;
}

export interface OverviewTotalsDto {
  hosts: number;
  hosts_operational: number;
  resources: number;
  free: number;
  active: number;
  unavailable: number;
  disabled: number;
}

export interface OverviewDto {
  totals: OverviewTotalsDto;
  hosts: HostOccupancyDto[];
  /** More than one Cooja build in the pool. */
  heterogeneous: boolean;
  cooja_builds: string[];
}

export interface HostDetailDto {
  host: HostDto;
  resources: ResourceDto[];
  occupancy: HostOccupancyDto | null;
}

export interface ResourceSimulationDto {
  simulation_id: string | null;
  status: string | null;
  experiment_id: string | null;
  generation_id?: string | null;
  started_at?: string | null;
  attempt_id?: string | null;
}

/** A request the master-node will carry out; the API only records it. */
export interface AcceptedDto {
  accepted: boolean;
  requested_at: string | null;
  detail: string;
}

// ── Writes ───────────────────────────────────────────────────────────────────

export interface HostAccessInput {
  address: string;
  port?: number;
  username: string;
  auth_method?: AuthMethod;
  /** Sealed in the browser. There is no way to express a plaintext here. */
  secret?: SealedSecret | null;
}

export interface ContainerAccessInput {
  username?: string;
  port?: number;
  address_mode?: AddressMode;
  auth_method?: AuthMethod;
  secret?: SealedSecret | null;
  remote_dir?: string;
}

export interface DiscoveryInput {
  mode?: DiscoveryMode;
  label_filter?: string;
  name_prefix?: string | null;
}

export interface HostCreateInput {
  name: string;
  enabled?: boolean;
  access: HostAccessInput;
  container_access?: ContainerAccessInput;
  discovery?: DiscoveryInput;
}

export interface HostUpdateInput {
  name?: string;
  enabled?: boolean;
  access?: Partial<HostAccessInput>;
  container_access?: ContainerAccessInput;
  discovery?: DiscoveryInput;
}
