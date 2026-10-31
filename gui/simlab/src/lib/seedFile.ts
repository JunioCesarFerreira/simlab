/**
 * The seeds file: `{ "seeds": [...] }` or a bare list.
 *
 * Shared by the launch wizard and the seed study, so that a list exported from
 * one can be imported in the other.
 */
export type SeedFile = { ok: true; seeds: number[] } | { ok: false; error: string };

export function parseSeedFile(text: string): SeedFile {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { ok: false, error: "Invalid JSON file" };
  }
  const list = Array.isArray(raw) ? raw : (raw as { seeds?: unknown } | null)?.seeds;
  if (!Array.isArray(list) || list.length === 0) {
    return { ok: false, error: 'Invalid file — expected { "seeds": [...] }' };
  }
  const seeds = list.map((value) => parseInt(String(value), 10));
  if (seeds.some((value) => !Number.isInteger(value))) {
    return { ok: false, error: "Invalid file — seeds must be integers" };
  }
  return { ok: true, seeds };
}

export function serializeSeedFile(seeds: number[]): string {
  return JSON.stringify({ seeds }, null, 2);
}
