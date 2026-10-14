/**
 * How the two availability levels are read.
 *
 * Regression, reported from the running system: a worker whose address had
 * been configured by hand showed Docker as "missing" in red while its SSH
 * probe said "ready" and it was placing simulations. The column reported, as
 * a failure, a fact the system deliberately ignores for that worker.
 *
 * The decision itself now arrives from the API as `docker_applies`, computed
 * with the same `operator_configured` the master-node acts on
 * (`pylib/db/models/infra.py`). Re-deriving it here left out the host-level
 * signals, so these assertions cover both the field and the fallback used
 * against an older API.
 */
import { describe, expect, it } from "vitest";
import {
  dockerLevel,
  endpointLabel,
  operatorConfigured,
  overrideSummary,
  probeLevel,
} from "../resourceLevels";
import type { ResourceDto } from "../../types/infra";

function worker(overrides: Partial<ResourceDto> = {}): ResourceDto {
  return {
    id: "r1",
    host_id: "h1",
    name: "vm7-cooja1",
    status: "available",
    enabled: true,
    container: { id: "", name: "", image: "", state: "running",
                 checked_at: "2026-10-12T20:00:00" },
    endpoint: { address: "localhost", port: 2222 },
    connection: {},
    remote_dir: "/opt/contiki-ng/tools/cooja",
    probe: { ssh_ok: true, java_version: "openjdk 21", cooja_jar_sha256: "a".repeat(64),
             checked_at: "2026-10-12T20:00:00", message: "ready" },
    current_simulation_id: null,
    remote_pid: null,
    reserved_at: null,
    started_at: null,
    origin: "discovered",
    last_check: null,
    last_check_message: "",
    ...overrides,
  } as ResourceDto;
}

describe("who decided how a worker is reached", () => {
  it("a worker with its own connection settings", () => {
    expect(operatorConfigured(worker({ connection: { port: 2222 } }))).toBe(true);
  });

  it("a worker registered by hand", () => {
    expect(operatorConfigured(worker({ origin: "manual" }))).toBe(true);
  });

  it("a purely discovered worker", () => {
    expect(operatorConfigured(worker())).toBe(false);
  });
});

describe("the Docker level", () => {
  it("is not a failure when the address was configured", () => {
    // What the operator saw: "missing" in red on a worker that was working.
    const view = dockerLevel(worker({
      container: { id: "", name: "", image: "", state: "missing",
                   checked_at: "2026-10-12T20:00:00" },
      connection: { username: "root", port: 2222 },
    }));
    expect(view.label).toBe("not used");
    expect(view.tone).toBe("unknown");
    expect(view.title).toContain("configured by hand");
  });

  it("says the same for a manually registered worker", () => {
    const view = dockerLevel(worker({
      origin: "manual",
      container: { id: "", name: "", image: "", state: "missing",
                   checked_at: "2026-10-12T20:00:00" },
    }));
    expect(view.label).toBe("not used");
  });

  it("still flags a stopped container that Docker does manage", () => {
    // Kept where it means something: nobody configured this worker, so
    // Docker's verdict is the only word on whether it exists.
    const view = dockerLevel(worker({
      container: { id: "", name: "", image: "", state: "exited",
                   checked_at: "2026-10-12T20:00:00" },
    }));
    expect(view.label).toBe("exited");
    expect(view.tone).toBe("bad");
  });

  it("reports a running container as running either way", () => {
    expect(dockerLevel(worker({ origin: "manual" })).tone).toBe("ok");
    expect(dockerLevel(worker()).tone).toBe("ok");
  });

  it("distinguishes not-yet-asked from a negative answer", () => {
    const view = dockerLevel(worker({
      container: { id: "", name: "", image: "", state: "", checked_at: null },
    }));
    expect(view.tone).toBe("unknown");
    expect(view.title).toContain("not been asked");
  });
});

describe("the SSH probe level", () => {
  it("is what says a configured worker is usable", () => {
    const view = probeLevel(worker({ connection: { port: 2222 } }));
    expect(view.label).toBe("ready");
    expect(view.tone).toBe("ok");
  });

  it("distinguishes unreachable from present-but-without-Cooja", () => {
    // Different fixes, so one combined badge would hide which.
    expect(probeLevel(worker({
      probe: { ssh_ok: false, java_version: null, cooja_jar_sha256: null,
               checked_at: "2026-10-12T20:00:00", message: "timed out" },
    })).label).toBe("unreachable");

    expect(probeLevel(worker({
      probe: { ssh_ok: true, java_version: "21", cooja_jar_sha256: null,
               checked_at: "2026-10-12T20:00:00", message: "cooja.jar not found" },
    })).label).toBe("no Cooja build");
  });

  it("says nothing has been attempted yet", () => {
    expect(probeLevel(worker({
      probe: { ssh_ok: false, java_version: null, cooja_jar_sha256: null,
               checked_at: null },
    })).label).toBe("not probed");
  });

  it("carries the build details for a mixed-pool check", () => {
    const view = probeLevel(worker());
    expect(view.title).toContain("openjdk 21");
    expect(view.title).toContain("a".repeat(64));
  });
});

describe("what a row shows about a worker", () => {
  it("its address", () => {
    expect(endpointLabel(worker())).toBe("localhost:2222");
    expect(endpointLabel(worker({ endpoint: {} }))).toBe("—");
  });

  it("only what it overrides, never what it inherits", () => {
    const summary = overrideSummary(worker({
      origin: "manual",
      connection: { username: "cooja", secret: { has_secret: true } },
    }));
    expect(summary).toBe("user cooja · own credential · configured");
    // A worker that differs in nothing says nothing.
    expect(overrideSummary(worker())).toBe("");
  });
});

describe("the decision comes from the API", () => {
  it("trusts docker_applies over the local reading", () => {
    // A discovered worker with no overrides, under a host the operator
    // converted to manual: the backend already ignores Docker's view, and the
    // page must say the same thing instead of reporting a failure in red.
    const resource = {
      origin: "discovered",
      connection: {},
      docker_applies: false,
      container: { state: "missing", checked_at: "2026-10-13T00:00:00" },
    } as unknown as ResourceDto;

    expect(operatorConfigured(resource)).toBe(true);
    expect(dockerLevel(resource).label).toBe("not used");
    expect(dockerLevel(resource).tone).toBe("unknown");
  });

  it("keeps Docker's verdict when the API says it applies", () => {
    const resource = {
      origin: "manual",
      connection: { port: 2223 },
      docker_applies: true,
      container: { state: "exited", checked_at: "2026-10-13T00:00:00" },
    } as unknown as ResourceDto;

    expect(operatorConfigured(resource)).toBe(false);
    expect(dockerLevel(resource).tone).toBe("bad");
  });

  it("falls back to the local reading when the field is absent", () => {
    const resource = {
      origin: "manual",
      connection: {},
      container: { state: "missing", checked_at: "2026-10-13T00:00:00" },
    } as unknown as ResourceDto;

    expect(operatorConfigured(resource)).toBe(true);
  });
});
