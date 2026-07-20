/**
 * NSGA-III reference-point helpers (Das-Dennis construction).
 *
 * With M objectives and p divisions per axis, the simplex lattice has
 * H = C(M + p - 1, p) reference points. NSGA-III niching works best when the
 * population size is at least H (conventionally the smallest multiple of 4).
 */

/** H = C(M + p - 1, p). Returns 0 for out-of-domain inputs. */
export function referencePointCount(M: number, p: number): number {
  if (!Number.isInteger(M) || !Number.isInteger(p) || M < 2 || p < 1) return 0
  let h = 1
  for (let i = 1; i <= p; i++) h = (h * (M - 1 + i)) / i
  return Math.round(h)
}

/** Smallest multiple of 4 that is ≥ H. */
export function suggestedPopulationSize(h: number): number {
  if (h <= 0) return 0
  return Math.ceil(h / 4) * 4
}
