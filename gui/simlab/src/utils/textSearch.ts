/**
 * Case- and accent-insensitive text matching for the list filters. Names are
 * frequently written with diacritics ("Estimativa de população", "Variações de
 * seeds"), so a plain `toLowerCase().includes()` would miss the entry whenever
 * the user types the unaccented form.
 */
export function normalizeText(text: string): string {
  return text
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();
}

/**
 * True when every whitespace-separated token of `query` appears somewhere in
 * `text`, in any order. An empty (or blank) query matches everything, so the
 * caller can bind it straight to an input without special-casing.
 */
export function matchesQuery(text: string, query: string): boolean {
  const tokens = normalizeText(query).split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return true;

  const haystack = normalizeText(text);
  return tokens.every((t) => haystack.includes(t));
}
