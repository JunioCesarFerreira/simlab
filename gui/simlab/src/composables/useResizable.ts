import { ref, type Ref } from "vue";

interface ResizableOptions {
  initial?: number;
  min?: number;
  max?: number;
  /** External ref to drive (e.g. a v-model passthrough) instead of an internal one. */
  height?: Ref<number>;
}

export function useResizable(options: ResizableOptions = {}) {
  const { initial = 420, min = 180, max = 1400 } = options;
  const height = options.height ?? ref(initial);

  function clamp(v: number): number {
    return Math.max(min, Math.min(max, v));
  }

  /** Keyboard-driven resize: adjust by a delta, clamped to [min, max]. */
  function nudge(delta: number) {
    height.value = clamp(height.value + delta);
  }

  // Bind with @pointerdown (not @mousedown). Pointer capture routes every
  // subsequent pointer event to the handle — including a release outside the
  // browser window — so a drag can never leak a document-level listener that
  // keeps resizing after the button is up (symptom: the chart follows the
  // cursor / collapses until refresh).
  function startResize(e: PointerEvent) {
    const handle = e.currentTarget as HTMLElement;
    const startY = e.clientY;
    const startH = height.value;

    function onMove(ev: PointerEvent) {
      height.value = clamp(startH + (ev.clientY - startY));
    }

    function onEnd() {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onEnd);
      handle.removeEventListener("pointercancel", onEnd);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    handle.setPointerCapture(e.pointerId);
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onEnd);
    handle.addEventListener("pointercancel", onEnd);
    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  }

  return { height, startResize, nudge };
}
