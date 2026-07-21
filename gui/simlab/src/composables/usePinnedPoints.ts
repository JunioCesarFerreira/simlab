import type { Ref } from "vue";

/**
 * Multi-point pin logic shared by ParetoFrontChart and ParetoFront3DChart.
 *
 * `markMode`/`markedIds` are owned by the caller (a `defineModel` in each
 * chart, normally bound to the same two refs in the parent page) rather than
 * created here — that is what keeps the same pins showing when the page
 * swaps the 2D chart for the 3D one: both charts read/write the identical
 * shared state instead of each holding its own local copy that resets on
 * mount.
 *
 * Pinned ids are tracked in an array (pin order = badge/color order) and
 * mutated immutably so Vue's reactivity always picks up the change. Callers
 * must NOT filter pinned points out of their source series — the point stays
 * in its original series and the chart draws an additional highlight overlay
 * for each pinned id on top of it. That is what keeps a pinned point from
 * ever disappearing: pinning only adds a decoration, it never removes the
 * original marker.
 */
export function usePinnedPoints(markMode: Ref<boolean>, markedIds: Ref<string[]>) {
  function isPinned(id: string): boolean {
    return markedIds.value.includes(id);
  }

  /** Pin `id` if not already pinned, otherwise unpin it. */
  function togglePin(id: string): void {
    markedIds.value = isPinned(id)
      ? markedIds.value.filter((x) => x !== id)
      : [...markedIds.value, id];
  }

  function unpin(id: string): void {
    markedIds.value = markedIds.value.filter((x) => x !== id);
  }

  function clearPins(): void {
    markedIds.value = [];
  }

  return { markMode, markedIds, isPinned, togglePin, unpin, clearPins };
}

/** Warm, mutually-distinct palette for pinned-point highlights, cycled by
 *  pin order — kept separate from RANK_PALETTE (front colors) so a pin never
 *  visually collides with the front/rank color of the point underneath it. */
export const PIN_COLORS = [
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // violet
  "#0ea5e9", // sky
  "#22c55e", // green
  "#ec4899", // pink
] as const;

export function pinColorAt(index: number): string {
  return PIN_COLORS[index % PIN_COLORS.length]!;
}
