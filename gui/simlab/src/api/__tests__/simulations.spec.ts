/**
 * The individual id is a chromosome hash, stable across experiments, so the
 * request has to carry the experiment too. Without it the panel listed every
 * run of that chromosome ever made — nine simulations where two belonged to
 * the experiment on screen.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import client from "../client";
import { getSimulationsByIndividual } from "../simulations";

vi.mock("../client", () => ({ default: { get: vi.fn() } }));

describe("getSimulationsByIndividual", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("scopes the request to the experiment", async () => {
    vi.mocked(client.get).mockResolvedValue({ data: [] });

    await getSimulationsByIndividual("abc123hash", "6acd5f87062f49f04be6d5dd");

    const [url, config] = vi.mocked(client.get).mock.calls[0]!;
    expect(url).toBe("/simulations/by-individual/abc123hash");
    expect(config?.params).toEqual({ experiment_id: "6acd5f87062f49f04be6d5dd" });
  });

  it("returns the simulations as received", async () => {
    const data = [
      { id: "s1", random_seed: 123456 },
      { id: "s2", random_seed: 336157 },
    ];
    vi.mocked(client.get).mockResolvedValue({ data });

    const found = await getSimulationsByIndividual("abc123hash", "exp1");

    expect(found).toBe(data);
  });
});
