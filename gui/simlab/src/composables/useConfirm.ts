import { ref } from "vue";

export interface ConfirmOptions {
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  /** Style the confirm button as destructive (red). */
  danger?: boolean;
}

interface ActiveConfirm extends ConfirmOptions {
  resolve: (ok: boolean) => void;
}

const active = ref<ActiveConfirm | null>(null);

/**
 * Promise-based replacement for window.confirm(), rendered by the global
 * ConfirmDialog mounted in App.vue:
 *
 *   if (!(await confirmDialog({ title, message, danger: true }))) return;
 */
export function confirmDialog(options: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    // A second request while one is open cancels the first.
    active.value?.resolve(false);
    active.value = { ...options, resolve };
  });
}

/** Internal — used by the ConfirmDialog component. */
export function useConfirmState() {
  function settle(ok: boolean) {
    active.value?.resolve(ok);
    active.value = null;
  }
  return { active, settle };
}
