/**
 * A chart height remembered for the session — and never a reason for the
 * chart not to render when the storage is missing or broken.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { usePersistedHeight } from "../usePersistedHeight";

function fakeStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    data,
  };
}

describe("usePersistedHeight", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("starts from the height chosen earlier in the session", () => {
    vi.stubGlobal("sessionStorage", fakeStorage({ "simlab:chart-h:usl:pool": "520" }));

    expect(usePersistedHeight("usl:pool", 380).value).toBe(520);
  });

  it("keeps each chart's height apart", () => {
    vi.stubGlobal("sessionStorage", fakeStorage({ "simlab:chart-h:usl:pool": "520" }));

    expect(usePersistedHeight("usl:machine", 380).value).toBe(380);
  });

  it("ignores a stored value that is not a height", () => {
    vi.stubGlobal("sessionStorage", fakeStorage({ "simlab:chart-h:usl:pool": "tall" }));

    expect(usePersistedHeight("usl:pool", 380).value).toBe(380);
  });

  it("falls back to the default when the storage is unavailable", () => {
    vi.stubGlobal("sessionStorage", {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    });

    const height = usePersistedHeight("usl:pool", 380);
    height.value = 600;

    expect(height.value).toBe(600);
  });

  it("writes once a drag settles, not on every pixel", async () => {
    const storage = fakeStorage();
    vi.stubGlobal("sessionStorage", storage);
    const height = usePersistedHeight("usl:pool", 380);

    for (const h of [400, 450, 500.4]) {
      height.value = h;
      await nextTick();
    }
    expect(storage.data.size).toBe(0);

    vi.advanceTimersByTime(300);
    expect(storage.data.get("simlab:chart-h:usl:pool")).toBe("500");
  });
});
