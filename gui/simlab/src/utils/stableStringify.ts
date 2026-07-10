/**
 * Deterministic serialization used as a lookup key for structurally-equal
 * objects (e.g. matching a `pareto_front` item's chromosome back to the
 * `individual_id` of the population entry it came from). Object keys are
 * sorted so key order never affects the resulting string.
 */
export function stableStringify(val: unknown): string {
  if (Array.isArray(val)) return `[${val.map(stableStringify).join(",")}]`;

  if (val !== null && typeof val === "object") {
    const entries = Object.keys(val as object)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${stableStringify((val as Record<string, unknown>)[k])}`);
    return `{${entries.join(",")}}`;
  }

  return JSON.stringify(val);
}
