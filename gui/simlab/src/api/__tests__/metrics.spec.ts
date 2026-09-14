import { expect, it, vi } from "vitest";
import client from "../client";
import { getHvGd } from "../metrics";

vi.mock("../client", () => ({ default: { get: vi.fn() } }));

it("requests only the displayed series and forwards cancellation", async () => {
  vi.mocked(client.get).mockResolvedValue({ data: { generations: [] } });
  const signal = new AbortController().signal;
  await getHvGd({ experimentId: "a", objectiveNames: ["latency", "throughput"],
    objectiveGoals: ["min", "max"], population: "survivors" }, signal);
  const [url, config] = vi.mocked(client.get).mock.calls[0]!;
  expect(url).toBe("/experiments/a/hv-gd");
  expect(config?.signal).toBe(signal);
  const params = config?.params as URLSearchParams;
  expect(params.getAll("objectives")).toEqual(["latency", "throughput"]);
  expect(params.getAll("minimize")).toEqual(["true", "false"]);
  expect(params.get("include_cumulative")).toBe("false");
});
