import { ref, watch, type Ref } from "vue";

/**
 * A chart height the viewer chose, remembered for the browser session.
 *
 * Session storage, like the experiment page's chart heights: a preference
 * for this tab, not a setting — and the storage may be unavailable (private
 * windows, blocked site data), in which case the height just starts from
 * `initial` again.
 */
export function usePersistedHeight(key: string, initial: number): Ref<number> {
  const storageKey = `simlab:chart-h:${key}`;
  let stored: number | null = null;
  try {
    const raw = sessionStorage.getItem(storageKey);
    const value = raw === null ? NaN : Number(raw);
    if (Number.isFinite(value) && value > 0) stored = value;
  } catch {
    // unavailable: fall back to the default
  }
  const height = ref(stored ?? initial);

  // Debounced: a resize drag changes the height dozens of times per second.
  let timer: ReturnType<typeof setTimeout> | null = null;
  watch(height, (value) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      try {
        sessionStorage.setItem(storageKey, String(Math.round(value)));
      } catch {
        // ignore quota / security errors
      }
    }, 300);
  });
  return height;
}
