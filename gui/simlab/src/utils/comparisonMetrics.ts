import type { ParetoFrontItemDto, ObjectiveItem } from '../types/simlab';
import { isPenalized } from '../types/simlab';

export function extractFront(items: ParetoFrontItemDto[], objectives: ObjectiveItem[]): number[][] {
  return items
    .map(item => objectives.map(o => item.objectives[o.metric_name] ?? 0))
    .filter(vals => !isPenalized(vals));
}

// Returns true if a Pareto-dominates b
export function dominates(a: number[], b: number[], isMin: boolean[]): boolean {
  let strict = false;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!, bi = b[i]!;
    if (isMin[i]) {
      if (ai > bi) return false;
      if (ai < bi) strict = true;
    } else {
      if (ai < bi) return false;
      if (ai > bi) strict = true;
    }
  }
  return strict;
}

// C(A, B): fraction of B dominated by at least one solution in A
export function coverage(A: number[][], B: number[][], isMin: boolean[]): number {
  if (!B.length) return 0;
  let count = 0;
  for (const b of B) if (A.some(a => dominates(a, b, isMin))) count++;
  return count / B.length;
}

// Additive epsilon-indicator I_eps(A, B):
// smallest eps s.t. every b in B is eps-dominated by some a in A.
// For min-obj i: needs a_i <= b_i + eps  →  eps >= a_i - b_i
// For max-obj i: needs a_i >= b_i - eps  →  eps >= b_i - a_i
export function epsilonIndicator(A: number[][], B: number[][], isMin: boolean[]): number {
  if (!A.length || !B.length) return Infinity;
  let worst = -Infinity;
  for (const b of B) {
    let best = Infinity;
    for (const a of A) {
      let eps = -Infinity;
      for (let i = 0; i < a.length; i++) {
        const e = isMin[i] ? a[i]! - b[i]! : b[i]! - a[i]!;
        if (e > eps) eps = e;
      }
      if (eps < best) best = eps;
    }
    if (best > worst) worst = best;
  }
  return worst;
}

// IGD+(A, B): averaged modified Hausdorff distance from B to A.
// For each b in B finds the nearest a in A using the non-dominated direction:
//   d+(a,b) = sqrt( sum_i max(a_i - b_i, 0)^2 )  for min objectives
//             sqrt( sum_i max(b_i - a_i, 0)^2 )  for max objectives
export function igdPlus(A: number[][], B: number[][], isMin: boolean[]): number {
  if (!A.length || !B.length) return Infinity;
  let total = 0;
  for (const b of B) {
    let minD = Infinity;
    for (const a of A) {
      let d2 = 0;
      for (let i = 0; i < a.length; i++) {
        const diff = isMin[i] ? Math.max(a[i]! - b[i]!, 0) : Math.max(b[i]! - a[i]!, 0);
        d2 += diff * diff;
      }
      const d = Math.sqrt(d2);
      if (d < minD) minD = d;
    }
    total += minD;
  }
  return total / B.length;
}

// Schott's spacing: measures distribution uniformity (lower = more uniform)
export function spacing(front: number[][]): number {
  const n = front.length;
  if (n < 2) return 0;
  const dists = front.map((a, i) => {
    let minD = Infinity;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const b = front[j]!;
      let d2 = 0;
      for (let k = 0; k < a.length; k++) d2 += (a[k]! - b[k]!) ** 2;
      const d = Math.sqrt(d2);
      if (d < minD) minD = d;
    }
    return minD;
  });
  const mean = dists.reduce((s, d) => s + d, 0) / n;
  return Math.sqrt(dists.reduce((s, d) => s + (d - mean) ** 2, 0) / (n - 1));
}
