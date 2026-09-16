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

/**
 * Finest Das-Dennis lattice that still fits in a population of *populationSize*:
 * the largest p with H(M, p) <= populationSize.
 *
 * A fixed p cannot serve every M — the lattice grows combinatorially. p = 10
 * gives 11 directions in M = 2, which leaves a population of 50 competing for a
 * tenth of the niches it could use, and 3003 in M = 6, which is the opposite
 * failure. Deriving p from M and the population keeps roughly one direction per
 * individual, which is the regime NSGA-III niching is designed for.
 *
 * Returns 0 for out-of-domain inputs, and never exceeds *maxDivisions* (the
 * lattice for M = 2 would otherwise run to p = populationSize - 1, which is
 * correct but makes the number look alarming next to the other cases).
 */
export function suggestedDivisions(
  M: number,
  populationSize: number,
  maxDivisions = 200,
): number {
  if (!Number.isInteger(M) || M < 2 || !Number.isInteger(populationSize) || populationSize < 2) {
    return 0
  }
  let best = 0
  for (let p = 1; p <= maxDivisions; p++) {
    const h = referencePointCount(M, p)
    if (h === 0 || h > populationSize) break
    best = p
  }
  return best
}

/**
 * Expected number of variables mutated per child.
 *
 * The two probabilities compose: *probMt* decides whether a child is mutated at
 * all, then *perGeneProb* is tested per variable. The product is easy to lose
 * sight of — 0.1 x 0.05 over 10 variables is one variable touched every twenty
 * children — so the wizard shows this number rather than the two factors alone.
 */
export function expectedMutatedVariables(
  probMt: number,
  perGeneProb: number,
  nVars: number,
): number {
  if (![probMt, perGeneProb, nVars].every((v) => Number.isFinite(v) && v >= 0)) return 0
  return probMt * perGeneProb * nVars
}
