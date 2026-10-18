/**
 * Reading where a simulation ran.
 *
 * Two simulations of the same individual differ by seed *and* by machine. The
 * panel showed only the seed, so a difference between replicas could not be
 * told apart from a difference between workers — and the distribution of work
 * across hosts could not be audited at all, even though the master-node had
 * been recording it since the reservation.
 *
 * Extracted as a module so the rules are testable without mounting a
 * component, the same way `resourceLevels` is.
 */
import type {
  PlacementDistributionDto,
  PlacementDto,
  PlacementWorkerDto,
  SimulationDto,
} from "../types/simlab";

/** Host and worker, as `host/worker` — what the card shows. */
export function placementLabel(placement?: PlacementDto | null): string {
  if (!placement) return "";
  const host = placement.host_name || "?";
  const worker = placement.resource_name || "?";
  return `${host}/${worker}`;
}

/**
 * Whether two runs executed in the same place.
 *
 * The container id decides when both have one: a worker's *name* can be
 * reused by a rebuilt container, and then two runs that look alike came from
 * different machines. Falls back to host and worker, which is all an older
 * record carries.
 */
export function ranInTheSamePlace(a?: PlacementDto | null, b?: PlacementDto | null): boolean {
  if (!a || !b) return false;
  if (a.container_id && b.container_id) return a.container_id === b.container_id;
  return a.host_name === b.host_name && a.resource_name === b.resource_name;
}

/** How many runs each host took, for reading a generation at a glance. */
export function countByHost(simulations: SimulationDto[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const sim of simulations) {
    const host = sim.placement?.host_name;
    if (!host) continue;
    counts.set(host, (counts.get(host) ?? 0) + 1);
  }
  return counts;
}

/* ── An experiment's distribution over the machines that ran it ───────────
 *
 * The functions above read *one* run's placement. These read the aggregate
 * the API publishes for a whole experiment, and they exist because three of
 * its decisions are easy to get wrong in a template:
 *
 * 1. **What a percentage is a percentage of.** The share is over the
 *    *placed* simulations. A run still queued belongs to no container, so
 *    counting it would make every container's share shrink as the queue
 *    grows.
 * 2. **What a bar is scaled against.** The busiest container, not the total:
 *    with twelve workers every bar would otherwise be a sliver, and the
 *    question is which of them carried more.
 * 3. **When a rounded number lies.** `0.0%` reads as "none", which is wrong
 *    for a container that did run something.
 */

/** How many containers appear in the distribution, across every host. */
export function workerCount(data?: PlacementDistributionDto | null): number {
  return data?.hosts.reduce((n, host) => n + host.workers.length, 0) ?? 0;
}

/**
 * The scale for the bars: the busiest container's simulation count.
 *
 * Never zero, so a caller can divide by it. An empty distribution gives 1,
 * which renders nothing because every count is already 0.
 */
export function busiestWorker(data?: PlacementDistributionDto | null): number {
  const totals = data?.hosts.flatMap((host) => host.workers.map((w) => w.total)) ?? [];
  return Math.max(1, ...totals);
}

/** One segment's width, as a percentage of the busiest container's bar. */
export function segmentWidth(count: number, busiest: number): number {
  if (busiest <= 0) return 0;
  return (100 * count) / busiest;
}

/**
 * A share, with the one rounding that would otherwise mislead called out.
 *
 * A container holding 1 of 4000 simulations is 0.025%, and `0.0%` is read as
 * "this one did nothing" — the opposite of what it did.
 */
export function formatShare(share: number): string {
  if (share > 0 && share < 0.1) return "<0.1%";
  return `${share.toFixed(1)}%`;
}

/**
 * What a container produced, not just how much it was given.
 *
 * `done` is always shown — including as `0 done`, which is the case worth
 * seeing: a container that took work and returned nothing. The other two
 * appear only when non-zero, so the common line stays short.
 */
export function describeOutcome(worker: PlacementWorkerDto): string {
  const parts = [`${worker.done} done`];
  if (worker.failed) parts.push(`${worker.failed} failed`);
  if (worker.active) parts.push(`${worker.active} running`);
  return parts.join(", ");
}

/**
 * The sentence shown instead of the bars, or an empty string when there are
 * bars to show.
 *
 * Three different "nothing here" states, and they are not interchangeable:
 * work is queued and none placed yet, nothing ever ran, and no record at all.
 */
export function emptyMessage(data?: PlacementDistributionDto | null): string {
  if (!data) return "No record of where this experiment ran.";
  if (data.placed > 0) return "";
  if (data.unplaced > 0) {
    return `No simulation has been placed on a container yet (${data.unplaced} waiting).`;
  }
  return "No simulation of this experiment ran on a container.";
}
