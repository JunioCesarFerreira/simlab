import { ref } from "vue";

interface ResizableOptions {
  initial?: number;
  min?: number;
  max?: number;
}

export function useResizable(options: ResizableOptions = {}) {
  const { initial = 420, min = 180, max = 1400 } = options;
  const height = ref(initial);

  function startResize(e: MouseEvent) {
    const startY = e.clientY;
    const startH = height.value;

    function onMove(ev: MouseEvent) {
      height.value = Math.max(min, Math.min(max, startH + (ev.clientY - startY)));
    }

    function onUp() {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }

    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    e.preventDefault();
  }

  return { height, startResize };
}
