/**
 * Reading what an operator types as a machine's concurrency limit.
 *
 * Blank means no limit. Anything else must be a whole number from 1 — the
 * API refuses the rest, and saying so before the request is kinder.
 */
export type ParsedLimit = { ok: true; value: number | null } | { ok: false; error: string };

export function parseLimit(text: string): ParsedLimit {
  const trimmed = text.trim();
  if (trimmed === "") return { ok: true, value: null };
  if (!/^\d+$/.test(trimmed)) return { ok: false, error: "a whole number, or blank for none" };
  const value = Number(trimmed);
  if (value < 1) return { ok: false, error: "at least 1 — leave it blank for no limit" };
  return { ok: true, value };
}
