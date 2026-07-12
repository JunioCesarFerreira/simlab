import { ref } from "vue";

/**
 * Last unhandled runtime error, surfaced by the global banner in App.vue.
 * Populated by app.config.errorHandler (see main.ts) so a component that
 * throws doesn't just silently kill its subtree.
 */
const runtimeError = ref<string | null>(null);

export function reportRuntimeError(err: unknown, context?: string) {
  const msg = err instanceof Error ? err.message : String(err);
  runtimeError.value = context ? `${context}: ${msg}` : msg;
}

export function useRuntimeError() {
  function dismiss() {
    runtimeError.value = null;
  }
  return { runtimeError, dismiss };
}
