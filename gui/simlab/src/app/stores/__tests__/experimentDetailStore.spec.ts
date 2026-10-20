import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { getExperimentFull } from "../../../api/experiments";
import type { ExperimentFullDto } from "../../../types/simlab";
import { useExperimentDetailStore } from "../experimentDetailStore";

vi.mock("../../../api/experiments", () => ({ getExperimentFull: vi.fn() }));
const api = vi.mocked(getExperimentFull);
const snapshot = (id = "a", status = "Running") => ({
  id, status, generations: [], parameters: { objectives: [] },
}) as unknown as ExperimentFullDto;

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
  api.mockReset();
});
afterEach(() => {
  useExperimentDetailStore().clear();
  vi.useRealTimers();
});

describe("experiment chart polling", () => {
  it("does not overlap slow polls and waits after the response", async () => {
    const store = useExperimentDetailStore();
    api.mockResolvedValueOnce(snapshot());
    await store.fetch("a");
    let resolve!: (data: ExperimentFullDto) => void;
    api.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    store.startPolling("a");
    await vi.advanceTimersByTimeAsync(30_000);
    expect(api).toHaveBeenCalledTimes(2);
    const concurrent = store.refresh("a");
    expect(api).toHaveBeenCalledTimes(2);
    resolve(snapshot());
    await concurrent;
    await vi.advanceTimersByTimeAsync(2999);
    expect(api).toHaveBeenCalledTimes(2);
    api.mockResolvedValueOnce(snapshot("a", "Done"));
    await vi.advanceTimersByTimeAsync(1);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(api).toHaveBeenCalledTimes(3);
  });

  it("ignores late responses from an old experiment", async () => {
    const store = useExperimentDetailStore();
    let resolve!: (data: ExperimentFullDto) => void;
    api.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    const old = store.fetch("a");
    const signal = api.mock.calls[0]![1]!;
    api.mockResolvedValueOnce(snapshot("b"));
    await store.fetch("b");
    expect(signal.aborted).toBe(true);
    resolve(snapshot("a"));
    await old;
    expect(store.experiment?.id).toBe("b");
  });

  it("does not invalidate charts for an identical payload", async () => {
    const store = useExperimentDetailStore();
    api.mockImplementation(async () => snapshot());
    await store.fetch("a");
    const previous = store.experiment;
    const revision = store.revision;
    await store.refresh("a");
    expect(store.experiment).toBe(previous);
    expect(store.revision).toBe(revision);
    api.mockResolvedValueOnce(snapshot("a", "Done"));
    await store.refresh("a");
    expect(store.revision).toBe(revision + 1);
  });

  it("clear prevents pending requests and polls from restoring the page", async () => {
    const store = useExperimentDetailStore();
    api.mockResolvedValueOnce(snapshot());
    await store.fetch("a");
    store.startPolling("a");
    let resolve!: (data: ExperimentFullDto) => void;
    api.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    await vi.advanceTimersByTimeAsync(3000);
    store.clear();
    resolve(snapshot());
    await vi.advanceTimersByTimeAsync(30_000);
    expect(store.experiment).toBeNull();
    expect(api).toHaveBeenCalledTimes(2);
  });
});
