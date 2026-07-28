import { describe, it, expect } from "vitest";
import { ref } from "vue";
import { usePinnedPoints, pinColorAt, PIN_COLORS } from "../usePinnedPoints";

/** Fresh backing refs + composable, mirroring how each chart component
 *  wires its `defineModel`s to usePinnedPoints. */
function create() {
  return usePinnedPoints(ref(false), ref<string[]>([]));
}

describe("usePinnedPoints", () => {
  it("starts with no pins", () => {
    const { markedIds } = create();
    expect(markedIds.value).toEqual([]);
  });

  it("togglePin pins an unpinned id", () => {
    const { markedIds, togglePin, isPinned } = create();
    togglePin("a");
    expect(markedIds.value).toEqual(["a"]);
    expect(isPinned("a")).toBe(true);
  });

  it("togglePin unpins an already-pinned id", () => {
    const { markedIds, togglePin } = create();
    togglePin("a");
    togglePin("a");
    expect(markedIds.value).toEqual([]);
  });

  it("supports pinning more than one point at the same time", () => {
    const { markedIds, togglePin, isPinned } = create();
    togglePin("a");
    togglePin("b");
    togglePin("c");
    expect(markedIds.value).toEqual(["a", "b", "c"]);
    expect(isPinned("a")).toBe(true);
    expect(isPinned("b")).toBe(true);
    expect(isPinned("c")).toBe(true);
  });

  it("unpinning one id does not affect the others (pin order preserved)", () => {
    const { markedIds, togglePin, unpin } = create();
    togglePin("a");
    togglePin("b");
    togglePin("c");
    unpin("b");
    expect(markedIds.value).toEqual(["a", "c"]);
  });

  it("unpin is a no-op for an id that isn't pinned", () => {
    const { markedIds, togglePin, unpin } = create();
    togglePin("a");
    unpin("nonexistent");
    expect(markedIds.value).toEqual(["a"]);
  });

  it("clearPins removes every pin", () => {
    const { markedIds, togglePin, clearPins } = create();
    togglePin("a");
    togglePin("b");
    clearPins();
    expect(markedIds.value).toEqual([]);
  });

  it("markMode defaults to off", () => {
    const { markMode } = create();
    expect(markMode.value).toBe(false);
  });

  it("instances backed by different refs are independent", () => {
    const first = create();
    const second = create();
    first.togglePin("a");
    expect(second.markedIds.value).toEqual([]);
  });

  // Regression: switching the Pareto chart between 2D and 3D unmounts one
  // component and mounts the other. Each chart's `defineModel`s are bound to
  // the SAME parent refs, so two usePinnedPoints instances sharing those refs
  // (standing in for "the 2D chart" and "the 3D chart") must see one
  // another's pins and mode toggles instead of resetting on the swap.
  it("instances sharing the same refs (2D chart / 3D chart) stay in sync", () => {
    const markMode = ref(false);
    const markedIds = ref<string[]>([]);
    const chart2d = usePinnedPoints(markMode, markedIds);
    const chart3d = usePinnedPoints(markMode, markedIds);

    chart2d.markMode.value = true;
    chart2d.togglePin("a");
    chart2d.togglePin("b");

    expect(chart3d.markMode.value).toBe(true);
    expect(chart3d.markedIds.value).toEqual(["a", "b"]);
    expect(chart3d.isPinned("a")).toBe(true);

    chart3d.unpin("a");
    expect(chart2d.markedIds.value).toEqual(["b"]);
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
