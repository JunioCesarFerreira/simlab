import { nanoid } from 'nanoid'
import type {
  ChromosomeDraft,
  ExportedChromosomeFile,
  MacProtocol,
} from '../types/problem'

export type ChromosomeImportResult =
  | { ok: true; chromosome: ChromosomeDraft }
  | { ok: false; error: string }

function macLabel(gene: unknown): MacProtocol {
  return gene === 1 ? 'tsch' : 'csma'
}

export function importChromosomeJson(json: string): ChromosomeImportResult {
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch (e) {
    return { ok: false, error: `Invalid JSON: ${e}` }
  }

  const file = parsed as ExportedChromosomeFile
  if (!file?.problem_name || !file?.chromosome) {
    return { ok: false, error: 'JSON must contain "problem_name" and "chromosome"' }
  }

  const c = file.chromosome as Record<string, unknown>

  if (file.problem_name === 'problem1') {
    const relays = Array.isArray(c.relays) ? c.relays as Array<{ x: number; y: number }> : []
    return {
      ok: true,
      chromosome: {
        kind: 'problem1',
        macProtocol: macLabel(c.mac_protocol),
        relays: relays.map(r => ({ id: nanoid(), x: Number(r.x), y: Number(r.y) })),
      },
    }
  }

  if (file.problem_name === 'problem2' || file.problem_name === 'problem3') {
    const mask = Array.isArray(c.mask) ? (c.mask as unknown[]).map(b => (b ? 1 : 0)) : []
    return {
      ok: true,
      chromosome: {
        kind: file.problem_name,
        macProtocol: macLabel(c.mac_protocol),
        mask,
      },
    }
  }

  if (file.problem_name === 'problem4') {
    // The inverse of exportChromosome: drop the visits to the base (sojourn 0)
    // and number the other stops from the first candidate.
    const stops = Array.isArray(c.route) ? (c.route as unknown[]).map(i => Number(i)) : []
    const times = Array.isArray(c.sojourn_times) ? (c.sojourn_times as unknown[]).map(t => Number(t)) : []
    const kept = stops.map((s, i) => [s, times[i] ?? 0] as const).filter(([s]) => s !== 0)
    const route = kept.map(([s]) => s - 1)
    const sojournTimes = kept.map(([, t]) => t)
    return {
      ok: true,
      chromosome: {
        kind: 'problem4',
        macProtocol: macLabel(c.mac_protocol),
        route,
        sojournTimes,
      },
    }
  }

  return { ok: false, error: `Unknown problem_name: ${file.problem_name}` }
}
