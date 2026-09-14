import { afterEach, expect, it, vi } from "vitest";
import { effectScope, nextTick, shallowRef } from "vue";
import { useParetoRanks } from "../useParetoRanks";

afterEach(() => vi.unstubAllGlobals());

it("terminates obsolete work and ignores results from an old scope", async () => {
  const workers: FakeWorker[] = [];
  class FakeWorker {
    onmessage?: (event: { data: Map<string, number> }) => void;
    terminate = vi.fn();
    postMessage = vi.fn((value: unknown) => structuredClone(value));
    constructor() { workers.push(this); }
  }
  vi.stubGlobal("Worker", FakeWorker);
  const input = shallowRef({ points: [{ id: "a", objectives: [1, 2] }], minimize: [true, true] });
  const scope = effectScope();
  const ranks = scope.run(() => useParetoRanks(input))!;
  expect(workers[0]!.postMessage).toHaveBeenCalledOnce();
  input.value = { ...input.value, points: [{ id: "b", objectives: [2, 1] }] };
  await nextTick();
  expect(workers[0]!.terminate).toHaveBeenCalledOnce();
  workers[0]!.onmessage!({ data: new Map([["a", 0]]) });
  expect(ranks.value.size).toBe(0);
  workers[1]!.onmessage!({ data: new Map([["b", 0]]) });
  expect(ranks.value).toEqual(new Map([["b", 0]]));
  expect(workers[1]!.terminate).toHaveBeenCalledOnce();
  input.value = { ...input.value, points: [{ id: "c", objectives: [2, 2] }] };
  await nextTick();
  scope.stop();
  expect(workers[2]!.terminate).toHaveBeenCalledOnce();
});
