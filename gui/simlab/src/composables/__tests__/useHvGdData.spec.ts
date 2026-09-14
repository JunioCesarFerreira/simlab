import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { effectScope, nextTick, ref, type EffectScope } from "vue";
import { getHvGd, type HvGdData, type HvGdQuery } from "../../api/metrics";
import { useHvGdData } from "../useHvGdData";

vi.mock("../../api/metrics", () => ({ getHvGd: vi.fn() }));
const api = vi.mocked(getHvGd);
const result = (hv = 1) => ({ generations: [0], hv: [hv] }) as HvGdData;
const flush = async () => { await nextTick(); await Promise.resolve(); await nextTick(); };
let scope: EffectScope;

function mount() {
  const query = ref<HvGdQuery & { revision: number }>({ experimentId: "a", objectiveNames: ["x", "y"],
    objectiveGoals: ["min", "min"], population: "survivors", revision: 1 });
  scope = effectScope();
  const state = scope.run(() => useHvGdData(() => query.value))!;
  return { query, ...state };
}
beforeEach(() => api.mockReset());
afterEach(() => scope?.stop());

describe("live quality metrics", () => {
  it("coalesces arriving revisions without cancelling an active calculation", async () => {
    let resolve!: (data: HvGdData) => void;
    api.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    const { query, data } = mount();
    const signal = api.mock.calls[0]![1];
    query.value.revision++;
    await flush();
    query.value.revision++;
    await flush();
    expect(api).toHaveBeenCalledTimes(1);
    expect(signal.aborted).toBe(false);
    api.mockResolvedValueOnce(result(2));
    resolve(result());
    await flush();
    expect(api).toHaveBeenCalledTimes(2);
    expect(data.value?.hv).toEqual([2]);
  });

  it("cancels changed selections and discards their late responses", async () => {
    let resolve!: (data: HvGdData) => void;
    api.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    const { query, data } = mount();
    const signal = api.mock.calls[0]![1];
    api.mockResolvedValueOnce(result(2));
    query.value.population = "archive";
    await flush();
    expect(signal.aborted).toBe(true);
    resolve(result(1));
    await flush();
    expect(data.value?.hv).toEqual([2]);
  });

  it("keeps the chart visible on refresh failure and allows retry", async () => {
    api.mockResolvedValueOnce(result());
    const { query, state, data, errorMsg, retry } = mount();
    await flush();
    api.mockRejectedValueOnce(new Error("timeout"));
    query.value.revision++;
    await flush();
    expect(state.value).toBe("ready");
    expect(data.value?.hv).toEqual([1]);
    expect(errorMsg.value).toBe("timeout");
    api.mockResolvedValueOnce(result(2));
    await retry();
    expect(data.value?.hv).toEqual([2]);
    expect(errorMsg.value).toBe("");
  });

  it("retries an empty reference when a new generation arrives", async () => {
    api.mockResolvedValueOnce({ ...result(), generations: [] });
    const { query, state } = mount();
    await flush();
    expect(state.value).toBe("empty");
    api.mockResolvedValueOnce(result());
    query.value.revision++;
    await flush();
    expect(state.value).toBe("ready");
  });

  it("aborts on disposal without starting a queued refresh", async () => {
    let resolve!: (data: HvGdData) => void;
    api.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    const { query, data } = mount();
    query.value.revision++;
    await flush();
    scope.stop();
    expect(api.mock.calls[0]![1].aborted).toBe(true);
    resolve(result());
    await flush();
    expect(data.value).toBeNull();
    expect(api).toHaveBeenCalledTimes(1);
  });
});
