/**
 * Fast non-dominated sort (NSGA-II style).
 * Returns a Map from point ID to 0-indexed rank (0 = Pareto front, 1 = second front, …).
 *
 * Points with equal objectives are treated as non-dominating each other,
 * so duplicates end up on the same front as each other.
 */

export interface SortablePoint {
  id: string;
  objectives: number[]; // original-space values
}

/** b dominates a → true when b is ≥ as good on all, strictly better on at least one. */
function dominates(b: number[], a: number[], minimize: boolean[]): boolean {
  let betterOnOne = false;
  for (let i = 0; i < a.length; i++) {
    const bi = b[i]!;
    const ai = a[i]!;
    const bBetter = minimize[i] ? bi < ai : bi > ai;
    const bWorse  = minimize[i] ? bi > ai : bi < ai;
    if (bWorse) return false;
    if (bBetter) betterOnOne = true;
  }
  return betterOnOne;
}

export function computeRanks(
  points: SortablePoint[],
  minimize: boolean[],
): Map<string, number> {
  // Lexicographic order places every possible dominator before its target.
  // Insert into the first front that does not dominate the point. A binary
  // search over fronts avoids storing the quadratic graph of dominance edges.
  const ordered = points.map((p) => ({
    id: p.id,
    objectives: p.objectives.map((v, i) => minimize[i] ? v : -v),
  })).sort((a, b) => {
    for (let i = 0; i < a.objectives.length; i++) {
      const difference = a.objectives[i]! - b.objectives[i]!;
      if (difference !== 0) return difference;
    }
    return 0;
  });
  const fronts: number[][][] = [];
  const allMin = minimize.map(() => true);
  const result = new Map<string, number>();
  for (const point of ordered) {
    let low = 0;
    let high = fronts.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (fronts[mid]!.some((other) => dominates(other, point.objectives, allMin))) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    if (low === fronts.length) fronts.push([]);
    fronts[low]!.push(point.objectives);
    result.set(point.id, low);
  }
  return result;
}

/** Deduplicates by objective tuple, computes ranks, then maps back to all IDs. */
export function computeRanksWithDuplicates(
  points: SortablePoint[],
  minimize: boolean[],
): Map<string, number> {
  // Canonical representative per unique objective vector
  const objKeyToRep = new Map<string, SortablePoint>();
  const idToKey = new Map<string, string>();

  for (const p of points) {
    const key = p.objectives.join(",");
    idToKey.set(p.id, key);
    if (!objKeyToRep.has(key)) objKeyToRep.set(key, p);
  }

  const unique = [...objKeyToRep.values()];
  const rankMap = computeRanks(unique, minimize);

  const result = new Map<string, number>();
  for (const p of points) {
    const key = idToKey.get(p.id)!;
    const rep = objKeyToRep.get(key)!;
    result.set(p.id, rankMap.get(rep.id) ?? 0);
  }
  return result;
}
