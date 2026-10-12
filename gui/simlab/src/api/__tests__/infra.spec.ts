/**
 * The infrastructure client, pinned where its behaviour is not obvious.
 *
 * Mostly these assert the URL and payload shape, which matter because the
 * server rejects anything it does not recognise — a wrong key name surfaces
 * as a 422 at runtime rather than a type error at build time.
 *
 * The two that carry real meaning: a check and a discovery are *requests* the
 * master-node fulfils, so the caller must not expect a result in the
 * response; and a credential leaves this client only as a sealed envelope.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import client from "../client";
import {
  createHost,
  deleteHost,
  getOverview,
  getPublicKey,
  getResourceSimulation,
  listHosts,
  listResources,
  requestCheck,
  requestDiscovery,
  setHostEnabled,
  setResourceEnabled,
  updateHost,
} from "../infra";

vi.mock("../client", () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}));

const SEALED = { key_id: "b114dd6d8410a62a", ciphertext: "x".repeat(100) };

beforeEach(() => {
  // Calls accumulate across tests otherwise, so `mock.calls[0]` would keep
  // returning the first test's request.
  vi.clearAllMocks();
  vi.mocked(client.get).mockResolvedValue({ data: {} });
  vi.mocked(client.post).mockResolvedValue({ data: {} });
  vi.mocked(client.put).mockResolvedValue({ data: {} });
  vi.mocked(client.delete).mockResolvedValue({ data: null });
});

describe("reads", () => {
  it("fetches the overview and forwards cancellation", async () => {
    const signal = new AbortController().signal;
    await getOverview(signal);
    const [url, config] = vi.mocked(client.get).mock.calls[0]!;
    expect(url).toBe("/infra/overview");
    expect(config?.signal).toBe(signal);
  });

  it.each([
    ["hosts", listHosts, "/infra/hosts"],
    ["resources", listResources, "/infra/resources"],
  ])("lists %s from the right path", async (_label, fn, path) => {
    await (fn as (s?: AbortSignal) => Promise<unknown>)();
    expect(vi.mocked(client.get).mock.calls[0]![0]).toBe(path);
  });

  it("fetches the public key", async () => {
    await getPublicKey();
    expect(vi.mocked(client.get).mock.calls[0]![0]).toBe(
      "/infra/crypto/public-key",
    );
  });

  it("asks what a container is running", async () => {
    await getResourceSimulation("r1");
    expect(vi.mocked(client.get).mock.calls[0]![0]).toBe(
      "/infra/resources/r1/simulation",
    );
  });
});

describe("registering a host", () => {
  it("sends the credential only as a sealed envelope", async () => {
    vi.mocked(client.post).mockResolvedValue({ data: "new-id" });

    const id = await createHost({
      name: "lab-01",
      access: {
        address: "192.168.0.42",
        username: "simlab",
        auth_method: "key",
        secret: SEALED,
      },
    });

    expect(id).toBe("new-id");
    const [url, body] = vi.mocked(client.post).mock.calls[0]!;
    expect(url).toBe("/infra/hosts");
    const sent = body as { access: { secret: unknown } };
    expect(sent.access.secret).toEqual(SEALED);
    // Nothing resembling a plaintext field goes out of here.
    expect(JSON.stringify(body)).not.toMatch(/password|private_key/);
  });

  it("omits the credential on an edit that keeps the stored one", async () => {
    await updateHost("h1", {
      name: "lab-01",
      access: { address: "10.0.0.5", username: "simlab" },
    });
    const [url, body] = vi.mocked(client.put).mock.calls[0]!;
    expect(url).toBe("/infra/hosts/h1");
    // Absent, not null: the server leaves what it has when the key is missing.
    expect("secret" in (body as { access: object }).access).toBe(false);
  });
});

describe("pool membership", () => {
  it("enables through its own endpoint", async () => {
    await setHostEnabled("h1", true);
    expect(vi.mocked(client.post).mock.calls[0]![0]).toBe("/infra/hosts/h1/enable");
  });

  it("disables through its own endpoint", async () => {
    await setHostEnabled("h1", false);
    expect(vi.mocked(client.post).mock.calls[0]![0]).toBe("/infra/hosts/h1/disable");
  });

  it("writes only `enabled` on a container", async () => {
    await setResourceEnabled("r1", false);
    const [url, body] = vi.mocked(client.put).mock.calls[0]!;
    expect(url).toBe("/infra/resources/r1");
    // The one field a client may write; everything else is measured.
    expect(body).toEqual({ enabled: false });
  });

  it("deletes a host", async () => {
    await deleteHost("h1");
    expect(vi.mocked(client.delete).mock.calls[0]![0]).toBe("/infra/hosts/h1");
  });

  it("lets a 409 reach the caller", async () => {
    // A host that still has work running is a normal outcome to explain, not
    // an error for the client to swallow.
    vi.mocked(client.delete).mockRejectedValue(new Error("409 still running"));
    await expect(deleteHost("h1")).rejects.toThrow(/409/);
  });
});

describe("requests the master-node fulfils", () => {
  it("records a check and returns what was accepted", async () => {
    vi.mocked(client.post).mockResolvedValue({
      data: { accepted: true, requested_at: "2026-01-01T00:00:00", detail: "…" },
    });

    const accepted = await requestCheck("h1");

    expect(vi.mocked(client.post).mock.calls[0]![0]).toBe("/infra/hosts/h1/check");
    // The response says the request was taken, not what the probe found: the
    // caller polls the host for that.
    expect(accepted.accepted).toBe(true);
    expect(accepted).not.toHaveProperty("status");
  });

  it("records a discovery", async () => {
    await requestDiscovery("h1");
    expect(vi.mocked(client.post).mock.calls[0]![0]).toBe("/infra/hosts/h1/discover");
  });
});
