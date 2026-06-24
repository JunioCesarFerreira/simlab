import type { DraftSegment } from '../types/problem'

function fmt(n: number): string {
  return String(Math.round(n * 1000) / 1000)
}

export function buildSegmentExpressions(seg: DraftSegment): [string, string] {
  if (seg.type === 'line') {
    const { start, end } = seg
    const dx = end[0] - start[0]
    const dy = end[1] - start[1]
    const exprX = Math.abs(dx) < 1e-9 ? fmt(start[0]) : `${fmt(start[0])} + ${fmt(dx)} * t`
    const exprY = Math.abs(dy) < 1e-9 ? fmt(start[1]) : `${fmt(start[1])} + ${fmt(dy)} * t`
    return [exprX, exprY]
  }

  if (seg.type === 'ellipse') {
    const { center, radiusX, radiusY } = seg
    const exprX = `${fmt(center[0])} + ${fmt(radiusX)} * np.cos(2 * np.pi * t)`
    const exprY = `${fmt(center[1])} + ${fmt(radiusY)} * np.sin(2 * np.pi * t)`
    return [exprX, exprY]
  }

  return [seg.exprX, seg.exprY]
}
