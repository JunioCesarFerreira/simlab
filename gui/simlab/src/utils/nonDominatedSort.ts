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
    const bBetter = minimize[i] ? b[i] < a[i] : b[i] > a[i];
    const bWorse  = minimize[i] ? b[i] > a[i] : b[i] < a[i];
    if (bWorse) return false;
    if (bBetter) betterOnOne = true;
  }
  return betterOnOne;
}

export function computeRanks(
  points: SortablePoint[],
  minimize: boolean[],
): Map<string, number> {
  const n = points.length;
  const domCount  = new Int32Array(n);       // how many points dominate point i
  const dominated = Array.from({ length: n }, () => [] as number[]); // which points i dominates

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (dominates(points[j].objectives, points[i].objectives, minimize)) {
        domCount[i]++;
        dominated[j].push(i);
      } else if (dominates(points[i].objectives, points[j].objectives, minimize)) {
        domCount[j]++;
        dominated[i].push(j);
      }
    }
  }

  const rank = new Int32Array(n);
  let current: number[] = [];

  for (let i = 0; i < n; i++) {
    if (domCount[i] === 0) current.push(i);
  }

  let r = 0;
  while (current.length > 0) {
    const next: number[] = [];
    for (const i of current) {
      rank[i] = r;
      for (const j of dominated[i]) {
        if (--domCount[j] === 0) next.push(j);
      }
    }
    r++;
    current = next;
  }

  const result = new Map<string, number>();
  for (let i = 0; i < n; i++) result.set(points[i].id, rank[i]);
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
