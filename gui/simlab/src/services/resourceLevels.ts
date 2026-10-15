/**
 * Reading a worker's two availability levels.
 *
 * Availability is decided twice: Docker says whether the container exists and
 * runs, and the SSH probe says whether a simulation can actually be launched
 * in it. Keeping the two apart is what lets the page say *which* one failed —
 * "the container is stopped" and "it is up but has no Cooja" need different
 * fixes.
 *
 * The subtlety this module exists for: **where the operator stated the
 * address, Docker's view is not used at all.** On a shared host the SSH user
 * often cannot reach the daemon, so discovery reports nothing and the stored
 * container state says `missing` — while the worker answers perfectly well and
 * is running simulations. Presenting that as a failure reports, in red, a fact
 * the system deliberately ignores.
 */
import type { ResourceDto } from "../types/infra";

export type LevelTone = "ok" | "bad" | "unknown";

export interface LevelView {
  label: string;
  tone: LevelTone;
  title: string;
}

/**
 * Whether a person, rather than Docker, decided how this worker is reached.
 *
 * The answer comes from the API, which computes it from the worker *and* its
 * host with the same function the master-node acts on
 * (`operator_configured`, in `pylib/db/models/infra.py`). It used to be
 * re-derived here from `origin` and `connection` alone, which left out the
 * host's discovery and address modes: a worker under a host converted to
 * manual was shown as a Docker failure although the backend had already
 * decided to ignore that observation.
 *
 * The local reading survives only as a fallback for an older API that does
 * not send the field, and is deliberately the conservative half of the rule.
 */
export function operatorConfigured(resource: ResourceDto): boolean {
  if (resource.docker_applies !== undefined) return !resource.docker_applies;
  if (resource.origin === "manual") return true;
  return Object.keys(resource.connection ?? {}).length > 0;
}

/** How to present level 1 — what Docker reported. */
export function dockerLevel(resource: ResourceDto): LevelView {
  const state = resource.container?.state ?? "";
  const checked = Boolean(resource.container?.checked_at);

  if (state === "running") {
    return { label: state, tone: "ok", title: "Docker reports it running" };
  }
  if (operatorConfigured(resource)) {
    return {
      label: "not used",
      tone: "unknown",
      title:
        "This worker's address was configured by hand, so Docker's view of " +
        "it is not used. Whether it can run a simulation is decided by the " +
        "SSH probe alone.",
    };
  }
  if (!checked) {
    return {
      label: state || "unknown",
      tone: "unknown",
      title: "Docker has not been asked about this container yet",
    };
  }
  return {
    label: state || "unknown",
    tone: "bad",
    title: `Docker reports the container ${state || "in an unknown state"}`,
  };
}

/** How to present level 2 — what the SSH probe found. */
export function probeLevel(resource: ResourceDto): LevelView {
  const probe = resource.probe ?? {};
  if (!probe.checked_at) {
    return {
      label: "not probed",
      tone: "unknown",
      title: "No connection has been attempted yet",
    };
  }
  if (!probe.ssh_ok) {
    return {
      label: "unreachable",
      tone: "bad",
      title: probe.message || "The worker did not answer",
    };
  }
  if (!probe.cooja_jar_sha256) {
    return {
      label: "no Cooja build",
      tone: "bad",
      title:
        probe.message ||
        "It answers, but no cooja.jar was found where it is expected",
    };
  }
  if (probe.java_ok === false) {
    // A jar is a file; running it needs a runtime. Separate from the missing
    // build because the fix is different, and the operator needs to know
    // which one to go and look at.
    return {
      label: "no Java",
      tone: "bad",
      title:
        probe.message ||
        "The Cooja build is there, but the JVM that launches it did not run",
    };
  }
  return {
    label: "ready",
    tone: "ok",
    title: [
      probe.java_version,
      `cooja.jar ${probe.cooja_jar_sha256}`,
      `measured ${evidenceAge(probe.checked_at)}`,
    ]
      .filter(Boolean)
      .join("\n"),
  };
}

/**
 * How old the measurement is, in words.
 *
 * "ready" is a statement about the past, and the page did not say how far
 * past. A healthy worker is only re-measured every few minutes, so a worker
 * that stopped answering goes on reading `ready` until the next probe is due
 * — and an operator looking at a stale result has no way to tell.
 */
export function evidenceAge(checkedAt?: string | null): string {
  if (!checkedAt) return "";
  const seconds = Math.round((Date.now() - new Date(checkedAt).getTime()) / 1000);
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}min ago`;
  const hours = Math.round(minutes / 60);
  return hours < 48 ? `${hours}h ago` : `${Math.round(hours / 24)}d ago`;
}

/** A worker's address, as `host:port`. */
export function endpointLabel(resource: ResourceDto): string {
  const address = resource.endpoint?.address;
  const port = resource.endpoint?.port;
  if (!address) return "—";
  return port ? `${String(address)}:${String(port)}` : String(address);
}

/**
 * What this worker overrides for itself.
 *
 * Only its own settings: repeating the values it inherits from its host would
 * bury the difference, which is the one thing worth seeing per row.
 */
export function overrideSummary(resource: ResourceDto): string {
  const connection = resource.connection ?? {};
  const parts: string[] = [];
  if (connection.username) parts.push(`user ${connection.username}`);
  if (connection.secret?.has_secret) parts.push("own credential");
  if (resource.origin === "manual") parts.push("configured");
  return parts.join(" · ");
}
