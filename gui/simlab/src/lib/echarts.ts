/**
 * Central ECharts entry for the 2-D charts, built on `echarts/core` so the
 * bundle only carries the chart types and components we actually register —
 * the bare `import * as echarts from "echarts"` pulls the entire library
 * (pie, maps, candlestick, …) into every route chunk that draws a chart.
 *
 * The 3-D components (ParetoFront3DChart, Dtlz2Surface3D, …) import this same
 * module: echarts-gl registers itself on `echarts/lib/echarts`, which is the
 * exact instance `echarts/core` re-exports, so gl works against this registry.
 * They are additionally loaded through defineAsyncComponent so the echarts-gl
 * payload stays in their own lazy chunks.
 *
 * Registering a new chart/option feature? Add its module here, or the chart
 * silently renders nothing.
 */
import * as echarts from "echarts/core";
import { LineChart, ParallelChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  LegendScrollComponent,
  ParallelComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer, SVGRenderer } from "echarts/renderers";

echarts.use([
  ScatterChart,
  LineChart,
  ParallelChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  LegendScrollComponent,
  DataZoomComponent,
  VisualMapComponent,
  ParallelComponent,
  CanvasRenderer,
  SVGRenderer,
]);

export * from "echarts/core";
