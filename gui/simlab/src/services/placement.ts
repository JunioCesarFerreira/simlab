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
import type { PlacementDto, SimulationDto } from "../types/simlab";

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
export function ranInTheSamePlace(
  a?: PlacementDto | null,
  b?: PlacementDto | null,
): boolean {
  if (!a || !b) return false;
  if (a.container_id && b.container_id) return a.container_id === b.container_id;
  return (
    a.host_name === b.host_name && a.resource_name === b.resource_name
  );
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
