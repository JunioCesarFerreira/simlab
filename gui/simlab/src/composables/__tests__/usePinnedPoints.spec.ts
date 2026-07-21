import { describe, it, expect } from "vitest";
import { usePinnedPoints, pinColorAt, PIN_COLORS } from "../usePinnedPoints";

describe("usePinnedPoints", () => {
  it("starts with no pins", () => {
    const { markedIds } = usePinnedPoints();
    expect(markedIds.value).toEqual([]);
  });

  it("togglePin pins an unpinned id", () => {
    const { markedIds, togglePin, isPinned } = usePinnedPoints();
    togglePin("a");
    expect(markedIds.value).toEqual(["a"]);
    expect(isPinned("a")).toBe(true);
  });

  it("togglePin unpins an already-pinned id", () => {
    const { markedIds, togglePin } = usePinnedPoints();
    togglePin("a");
    togglePin("a");
    expect(markedIds.value).toEqual([]);
  });

  it("supports pinning more than one point at the same time", () => {
    const { markedIds, togglePin, isPinned } = usePinnedPoints();
    togglePin("a");
    togglePin("b");
    togglePin("c");
    expect(markedIds.value).toEqual(["a", "b", "c"]);
    expect(isPinned("a")).toBe(true);
    expect(isPinned("b")).toBe(true);
    expect(isPinned("c")).toBe(true);
  });

  it("unpinning one id does not affect the others (pin order preserved)", () => {
    const { markedIds, togglePin, unpin } = usePinnedPoints();
    togglePin("a");
    togglePin("b");
    togglePin("c");
    unpin("b");
    expect(markedIds.value).toEqual(["a", "c"]);
  });

  it("unpin is a no-op for an id that isn't pinned", () => {
    const { markedIds, togglePin, unpin } = usePinnedPoints();
    togglePin("a");
    unpin("nonexistent");
    expect(markedIds.value).toEqual(["a"]);
  });

  it("clearPins removes every pin", () => {
    const { markedIds, togglePin, clearPins } = usePinnedPoints();
    togglePin("a");
    togglePin("b");
    clearPins();
    expect(markedIds.value).toEqual([]);
  });

  it("markMode defaults to off", () => {
    const { markMode } = usePinnedPoints();
    expect(markMode.value).toBe(false);
  });

  it("each call returns independent state", () => {
    const first = usePinnedPoints();
    const second = usePinnedPoints();
    first.togglePin("a");
    expect(second.markedIds.value).toEqual([]);
  });
});

describe("pinColorAt", () => {
  it("returns colors from the palette in order", () => {
    expect(pinColorAt(0)).toBe(PIN_COLORS[0]);
    expect(pinColorAt(1)).toBe(PIN_COLORS[1]);
  });

  it("cycles once the pin count exceeds the palette size", () => {
    expect(pinColorAt(PIN_COLORS.length)).toBe(PIN_COLORS[0]);
    expect(pinColorAt(PIN_COLORS.length + 2)).toBe(PIN_COLORS[2]);
  });
});
