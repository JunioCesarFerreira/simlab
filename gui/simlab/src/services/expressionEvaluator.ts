/**
 * Evaluates a parametric expression string at a given t ∈ [0, 1].
 * Supports Python/NumPy syntax: np.cos, np.sin, np.pi, np.sqrt, np.abs, np.tan.
 */

const NP_REPLACEMENTS: [RegExp, string][] = [
  [/np\.cos/g,  'Math.cos'],
  [/np\.sin/g,  'Math.sin'],
  [/np\.pi/g,   'Math.PI'],
  [/np\.sqrt/g, 'Math.sqrt'],
  [/np\.abs/g,  'Math.abs'],
  [/np\.tan/g,  'Math.tan'],
  [/np\.exp/g,  'Math.exp'],
  [/np\.log/g,  'Math.log'],
]

function toJs(expr: string): string {
  let s = expr
  for (const [from, to] of NP_REPLACEMENTS) s = s.replace(from, to)
  return s
}

const fnCache = new Map<string, (t: number) => number>()

function compile(expr: string): (t: number) => number {
  if (fnCache.has(expr)) return fnCache.get(expr)!
  const js = toJs(expr)
   
  const fn = new Function('t', `"use strict"; return (${js})`) as (t: number) => number
  fnCache.set(expr, fn)
  return fn
}

export type EvalResult =
  | { ok: true; value: number }
  | { ok: false; error: string }

export function evalExpr(expr: string, t: number): EvalResult {
  try {
    const fn = compile(expr)
    const value = fn(t)
    if (!isFinite(value)) return { ok: false, error: 'Expression returned non-finite value' }
    return { ok: true, value }
  } catch (e) {
    return { ok: false, error: String(e) }
  }
}

/**
 * Sample a custom segment into N world-coordinate points.
 * Returns null if any evaluation fails.
 */
export function sampleCustomSegment(
  exprX: string,
  exprY: string,
  n = 60
): [number, number][] | null {
  const pts: [number, number][] = []
  for (let i = 0; i <= n; i++) {
    const t = i / n
    const rx = evalExpr(exprX, t)
    const ry = evalExpr(exprY, t)
    if (!rx.ok || !ry.ok) return null
    pts.push([rx.value, ry.value])
  }
  return pts
}

/**
 * Validate a custom expression: tries to compile and evaluate at t=0 and t=1.
 * Returns null if valid, error string otherwise.
 */
export function validateExpr(expr: string): string | null {
  if (!expr.trim()) return 'Expression is empty'
  for (const t of [0, 0.5, 1]) {
    const r = evalExpr(expr, t)
    if (!r.ok) return r.error
  }
  return null
}
