/**
 * Reading a host's containers as one bar.
 *
 * The four counts come from the scheduler's own inventory snapshot, so what
 * is drawn is what the placement decision saw. They are a **partition** of
 * the inventory, and that is the property the bar depends on: the segments
 * are drawn as fractions of `resource_total`, so a container counted twice
 * would push the bar past 100% and one counted in no category would leave a
 * gap that looks like missing hardware.
 *
 * The partition is produced by `inventory_snapshot` in
 * `pylib/db/repositories/infra.py`. Two rules there are worth knowing when
 * reading a bar:
 *
 * * **`disabled` follows the toggle, not the status.** A worker switched
 *   back on moves to `unavailable` until the next health cycle measures it —
 *   it is in the pool and not yet usable, which is a different fact from
 *   being switched off. Keying the bucket on the status made the bar say
 *   "disabled" next to a toggle that said otherwise.
 * * **`unavailable` is the complement**, not a list of statuses, so a state
 *   nobody enumerated still lands in a segment instead of vanishing.
 *
 * Pure, and kept out of the component so that correspondence can be tested.
 */

export type SegmentKind = "active" | "free" | "unavailable" | "disabled";

export interface OccupancyCounts {
  free: number;
  active: number;
  unavailable: number;
  disabled: number;
  /** The inventory. The segments partition it, so it is the denominator. */
  resourceTotal?: number;
  /** In use *and* administratively out — an attribute of `active`. */
  draining?: number;
}

export interface Segment {
  kind: SegmentKind;
  label: string;
  count: number;
  percent: number;
}

const LABELS: Array<[SegmentKind, string]> = [
  // Order is meaningful: in use first, then idle capacity, then what cannot
  // be used — so the eye reads occupancy left to right.
  ["active", "in use"],
  ["free", "free"],
  ["unavailable", "unavailable"],
  ["disabled", "disabled"],
];

/**
 * The denominator.
 *
 * `resource_total` when the API sends it, because the segments are a
 * partition of exactly that. Summing them is the fallback for an older API —
 * correct only while the partition holds, which is why
 * {@link occupancySegments} caps the total rather than trusting it blindly.
 */
export function occupancyTotal(counts: OccupancyCounts): number {
  if (counts.resourceTotal !== undefined) return counts.resourceTotal;
  return counts.active + counts.free + counts.unavailable + counts.disabled;
}

/** The segments to draw, widest meaning first, zero counts dropped. */
export function occupancySegments(counts: OccupancyCounts): Segment[] {
  const total = occupancyTotal(counts);
  const counted = counts.active + counts.free + counts.unavailable + counts.disabled;
  // A bar can only ever be the whole bar. If the counts and the inventory
  // ever disagree — a backend that double-counts, or one that drops a state —
  // the segments are drawn against whichever is larger, so the result is a
  // short bar rather than one that overflows its own container.
  const denominator = Math.max(total, counted);
  return LABELS.map(([kind, label]) => ({ kind, label, count: counts[kind] }))
    .filter((row) => row.count > 0)
    .map((row) => ({
      ...row,
      percent: denominator === 0 ? 0 : (row.count / denominator) * 100,
    }));
}

/**
 * Whether the counts and the inventory disagree.
 *
 * Not a condition the UI can fix, and not one it should hide: it means the
 * bar is describing something other than the host's containers.
 */
export function occupancyIsConsistent(counts: OccupancyCounts): boolean {
  if (counts.resourceTotal === undefined) return true;
  const counted = counts.active + counts.free + counts.unavailable + counts.disabled;
  return counted === counts.resourceTotal;
}

/** The sentence under the bar, and its `aria-label`. */
export function occupancySummary(counts: OccupancyCounts): string {
  if (occupancyTotal(counts) === 0) return "No containers on this host";
  const parts = occupancySegments(counts).map((s) => `${s.count} ${s.label}`);
  // Draining is not a category of its own: it says something extra about the
  // workers already counted as in use.
  if (counts.draining) parts.push(`${counts.draining} of them draining`);
  return parts.join(", ");
}
