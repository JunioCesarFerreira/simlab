import type * as echarts from "../lib/echarts";

function triggerDownload(dataUrl: string, filename: string): void {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

/**
 * zrender's SVG painter ignores `type`/`backgroundColor`/`pixelRatio` and
 * always returns a `data:image/svg+xml` URL from `getDataURL` (see
 * SVGPainter.toDataURL) — used by the HV/GD charts, which render with
 * `renderer: "svg"` for crisp lines on a handful of data points. Rasterizing
 * it onto a canvas keeps every chart's export a real, theme-matching PNG
 * regardless of which renderer it was drawn with.
 */
function rasterizeSvgDataUrl(
  svgDataUrl: string,
  backgroundColor: string,
  pixelRatio: number,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.width * pixelRatio;
      canvas.height = img.height * pixelRatio;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas 2D context unavailable"));
        return;
      }
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = () => reject(new Error("Failed to rasterize chart SVG for export"));
    img.src = svgDataUrl;
  });
}

export interface ExportChartImageOptions {
  /** Defaults to '#ffffff' — ECharts itself falls back to white when neither
   *  this nor the chart option set a backgroundColor, which would leave a
   *  transparent PNG that is unreadable once pasted onto a light page. */
  backgroundColor?: string;
  pixelRatio?: number;
}

/**
 * Exports the current rendered state of an ECharts instance as a PNG and
 * triggers a browser download. Works uniformly across the canvas, svg and
 * echarts-gl (WebGL) renderers used across the app's charts — echarts-gl's
 * WebGL layer is created with `preserveDrawingBuffer: true`, so `getDataURL`
 * can read back the last rendered frame; svg-rendered charts are rasterized
 * (see `rasterizeSvgDataUrl`) so the result is always a genuine PNG.
 */
export async function exportChartImage(
  chart: echarts.EChartsType | null | undefined,
  filename: string,
  opts: ExportChartImageOptions = {},
): Promise<void> {
  if (!chart) return;
  const backgroundColor = opts.backgroundColor ?? "#ffffff";
  const requestedPixelRatio = opts.pixelRatio ?? 2;
  // zrender's canvas compositor only bridges custom layers (echarts-gl's
  // WebGL 3-D scenes) through its normal per-layer draw path when
  // `pixelRatio <= chart.getDevicePixelRatio()`; a higher pixelRatio silently
  // falls back to redrawing the 2-D display list element by element, which
  // has no notion of the WebGL layer and exports a blank 3-D scene. Capping
  // here keeps every chart — 2-D or 3-D — on the safe path.
  const pixelRatio = Math.min(requestedPixelRatio, chart.getDevicePixelRatio());
  const dataUrl = chart.getDataURL({ type: "png", pixelRatio, backgroundColor });
  // The SVG rasterization path below draws onto its own canvas, so it isn't
  // subject to the constraint above — the originally requested pixelRatio is
  // safe to use there for a sharper raster.
  const pngUrl = dataUrl.startsWith("data:image/svg+xml")
    ? await rasterizeSvgDataUrl(dataUrl, backgroundColor, requestedPixelRatio)
    : dataUrl;
  triggerDownload(pngUrl, filename);
}

/** Date-stamped PNG filename, e.g. "pareto-front_2026-07-12.png". */
export function chartExportFilename(base: string): string {
  const stamp = new Date().toISOString().slice(0, 10);
  return `${base}_${stamp}.png`;
}
