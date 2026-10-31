/**
 * The charts of a seed study, as plain ECharts options.
 *
 * One function per chart, taking the data and the theme palette. Pure, so a
 * chart can be tested without a canvas, and the component that draws them
 * stays one component.
 */
import type { EChartsOption } from "echarts";
import type { chartPalette } from "./chartTheme";
import { chromosomeLabel, formatValue } from "./seedStudy";
import type { MetricDetail, PanelAnalysis, StudyMetric } from "../types/seedStudy";

type Palette = ReturnType<typeof chartPalette>;

function base(c: Palette): EChartsOption {
  return {
    backgroundColor: c.bg,
    animation: false,
    grid: { left: 72, right: 28, top: 36, bottom: 52 },
    tooltip: {
      trigger: "item",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
    },
    legend: { top: 4, textStyle: { color: c.text, fontSize: 12 } },
  };
}

function axis(c: Palette, name: string, extra: Record<string, unknown> = {}) {
  return {
    type: "value" as const,
    name,
    nameLocation: "middle" as const,
    nameGap: 32,
    nameTextStyle: { color: c.text, fontSize: 12 },
    axisLabel: { color: c.muted, hideOverlap: true, formatter: (v: number) => formatValue(v) },
    axisLine: { lineStyle: { color: c.grid } },
    splitLine: { lineStyle: { color: c.grid } },
    scale: true,
    ...extra,
  };
}

function targetLine(c: Palette, label: string, y: number) {
  return {
    silent: true,
    symbol: "none",
    lineStyle: { type: "dashed" as const, color: c.gd },
    label: { color: c.muted, fontSize: 11, formatter: label, position: "insideEndTop" as const },
    data: [{ yAxis: y }],
  };
}

export function histogramOption(detail: MetricDetail, c: Palette): EChartsOption {
  return {
    ...base(c),
    legend: { show: false },
    xAxis: {
      type: "category",
      name: detail.metric,
      nameLocation: "middle",
      nameGap: 32,
      nameTextStyle: { color: c.text, fontSize: 12 },
      data: detail.histogram.map((bin) => formatValue((bin.from + bin.to) / 2)),
      axisLabel: { color: c.muted, hideOverlap: true },
      axisLine: { lineStyle: { color: c.grid } },
    },
    yAxis: axis(c, "Seeds", { minInterval: 1, scale: false }),
    tooltip: {
      ...(base(c).tooltip as object),
      formatter: (params: unknown) => {
        const { dataIndex } = params as { dataIndex: number };
        const bin = detail.histogram[dataIndex]!;
        return `${formatValue(bin.from)} to ${formatValue(bin.to)}<br/><b>${bin.count}</b> seed(s)`;
      },
    },
    series: [
      {
        type: "bar",
        data: detail.histogram.map((bin) => bin.count),
        barCategoryGap: "4%",
        itemStyle: { color: c.hv },
      },
    ],
  };
}

export function ecdfOption(detail: MetricDetail, c: Palette): EChartsOption {
  const sorted = detail.values.map((v) => v.value).sort((a, b) => a - b);
  const points = sorted.map((value, i) => [value, (i + 1) / sorted.length]);
  return {
    ...base(c),
    legend: { show: false },
    xAxis: axis(c, detail.metric),
    yAxis: axis(c, "Share of seeds at or below", { min: 0, max: 1, scale: false }),
    tooltip: { ...(base(c).tooltip as object), trigger: "axis" },
    series: [
      {
        type: "line",
        step: "end",
        data: points,
        showSymbol: false,
        lineStyle: { width: 2, color: c.hv },
        itemStyle: { color: c.hv },
      },
    ],
  };
}

export function boxOption(detail: MetricDetail, c: Palette): EChartsOption {
  const box = detail.box;
  const stats = box ? [box.low, box.q1, box.median, box.q3, box.high] : [];
  return {
    ...base(c),
    legend: { show: false },
    grid: { left: 72, right: 28, top: 20, bottom: 52 },
    xAxis: axis(c, detail.metric),
    yAxis: { type: "category", data: ["seeds"], axisLabel: { show: false } },
    series: [
      {
        type: "boxplot",
        data: [stats],
        itemStyle: { color: c.hvArea, borderColor: c.hv },
      },
      {
        type: "scatter",
        data: detail.values.map((v) => ({
          value: [v.value, "seeds"],
          itemStyle: { color: v.flag ? c.gd : c.muted, opacity: v.flag ? 1 : 0.6 },
          seed: v.seed,
        })),
        symbolSize: 7,
      },
    ],
  };
}

export function qqOption(detail: MetricDetail, c: Palette): EChartsOption {
  const mean = detail.summary.mean ?? 0;
  const sd = detail.summary.sd ?? 0;
  const zs = detail.qq.map((p) => p[0]);
  const low = Math.min(...zs);
  const high = Math.max(...zs);
  return {
    ...base(c),
    legend: { data: ["Observed", "Normal reference"], top: 4, textStyle: { color: c.text } },
    xAxis: axis(c, "Normal quantile"),
    yAxis: axis(c, detail.metric),
    series: [
      {
        name: "Observed",
        type: "scatter",
        data: detail.qq,
        symbolSize: 7,
        itemStyle: { color: c.hv },
      },
      {
        name: "Normal reference",
        type: "line",
        data: [
          [low, mean + low * sd],
          [high, mean + high * sd],
        ],
        showSymbol: false,
        lineStyle: { type: "dashed", color: c.gd },
        itemStyle: { color: c.gd },
      },
    ],
  };
}

/** By position in the seed list: a drift here would say the runs are not independent. */
export function orderOption(detail: MetricDetail, c: Palette): EChartsOption {
  return {
    ...base(c),
    legend: { show: false },
    xAxis: axis(c, "Position in the seed list", { minInterval: 1 }),
    yAxis: axis(c, detail.metric),
    tooltip: {
      ...(base(c).tooltip as object),
      formatter: (params: unknown) => {
        const { dataIndex } = params as { dataIndex: number };
        const value = detail.values[dataIndex]!;
        return (
          `seed ${value.seed}<br/><b>${formatValue(value.value)}</b>` +
          (value.flag ? `<br/>outlier (${value.flag})` : "")
        );
      },
    },
    series: [
      {
        type: "scatter",
        data: detail.values.map((v) => ({
          value: [v.order + 1, v.value],
          itemStyle: { color: v.flag ? c.gd : c.hv },
        })),
        symbolSize: 8,
        markLine:
          detail.summary.mean === null
            ? undefined
            : {
                silent: true,
                symbol: "none",
                lineStyle: { type: "dashed", color: c.muted },
                label: { color: c.muted, fontSize: 11, formatter: "mean" },
                data: [{ yAxis: detail.summary.mean }],
              },
      },
    ],
  };
}

export function runningOption(detail: MetricDetail, c: Palette): EChartsOption {
  const line = (name: string, key: "mean" | "low" | "high", dashed: boolean) => ({
    name,
    type: "line" as const,
    data: detail.running.map((point) => [point.n, point[key]]),
    showSymbol: false,
    lineStyle: {
      width: dashed ? 1 : 2,
      type: dashed ? ("dashed" as const) : ("solid" as const),
      color: dashed ? c.muted : c.hv,
    },
    itemStyle: { color: dashed ? c.muted : c.hv },
  });
  return {
    ...base(c),
    legend: { data: ["Running mean", "Interval"], top: 4, textStyle: { color: c.text } },
    xAxis: axis(c, "Seeds used", { minInterval: 1 }),
    yAxis: axis(c, detail.metric),
    tooltip: { ...(base(c).tooltip as object), trigger: "axis" },
    series: [
      line("Running mean", "mean", false),
      line("Interval", "low", true),
      { ...line("Interval", "high", true) },
    ],
  };
}

/**
 * P(within the target) against the number of seeds, or the expected value of an
 * extreme, whose target moves with the number of seeds.
 */
export function precisionOption(
  detail: MetricDetail,
  confidence: number,
  noTarget: boolean,
  c: Palette,
): EChartsOption {
  const curve = detail.precision?.curve ?? [];
  if (noTarget) {
    return {
      ...base(c),
      legend: { show: false },
      xAxis: axis(c, "Seeds aggregated (k)", { minInterval: 1 }),
      yAxis: axis(c, `Expected ${detail.metric}`),
      tooltip: { ...(base(c).tooltip as object), trigger: "axis" },
      series: [
        {
          type: "line",
          data: curve.map((point) => [point.k, point.expected]),
          showSymbol: false,
          lineStyle: { width: 2, color: c.hv },
          itemStyle: { color: c.hv },
        },
      ],
    };
  }
  return {
    ...base(c),
    legend: { show: false },
    xAxis: axis(c, "Seeds aggregated (k)", { minInterval: 1 }),
    yAxis: axis(c, "P(within the target)", { min: 0, max: 1, scale: false }),
    tooltip: { ...(base(c).tooltip as object), trigger: "axis" },
    series: [
      {
        type: "line",
        data: curve.map((point) => [point.k, point.p]),
        showSymbol: false,
        lineStyle: { width: 2, color: c.hv },
        itemStyle: { color: c.hv },
        markLine: targetLine(c, `${Math.round(confidence * 100)}%`, confidence),
      },
    ],
  };
}

export function reliabilityOption(panel: PanelAnalysis, target: number, c: Palette): EChartsOption {
  return {
    ...base(c),
    legend: {
      data: ["Eρ² (ranking)", "Φ (against a threshold)", "Spearman, disjoint seeds"],
      top: 4,
      textStyle: { color: c.text, fontSize: 12 },
    },
    xAxis: axis(c, "Seeds per solution (m)", { minInterval: 1, min: 1 }),
    yAxis: axis(c, "Reliability", { min: 0, max: 1, scale: false }),
    tooltip: { ...(base(c).tooltip as object), trigger: "axis" },
    series: [
      {
        name: "Eρ² (ranking)",
        type: "line",
        data: panel.reliability.map((row) => [row.m, row.e_rho2]),
        showSymbol: false,
        lineStyle: { width: 2.5, color: c.hv },
        itemStyle: { color: c.hv },
        markLine: targetLine(c, String(target), target),
      },
      {
        name: "Φ (against a threshold)",
        type: "line",
        data: panel.reliability.map((row) => [row.m, row.phi]),
        showSymbol: false,
        lineStyle: { width: 1.5, type: "dotted", color: c.igd },
        itemStyle: { color: c.igd },
      },
      {
        name: "Spearman, disjoint seeds",
        type: "scatter",
        data: panel.rank_stability.map((row) => [row.k, row.spearman]),
        symbolSize: 8,
        itemStyle: { color: c.colorB },
      },
    ],
  };
}

/** Each solution as a box, sorted by median: the noise against the differences. */
export function bySolutionOption(metric: StudyMetric, c: Palette): EChartsOption {
  const rows = metric.by_chromosome
    .map((row, index) => ({ row, index }))
    .filter(({ row }) => row.n > 0)
    .sort((a, b) => (a.row.median ?? 0) - (b.row.median ?? 0));
  return {
    ...base(c),
    legend: { show: false },
    xAxis: {
      type: "category",
      name: "Solutions, by median",
      nameLocation: "middle",
      nameGap: 32,
      nameTextStyle: { color: c.text, fontSize: 12 },
      data: rows.map(({ row, index }) => chromosomeLabel(row.genome_hash, index)),
      axisLabel: { show: false },
      axisLine: { lineStyle: { color: c.grid } },
    },
    yAxis: axis(c, metric.name),
    series: [
      {
        type: "boxplot",
        data: rows.map(({ row }) => [row.p5!, row.q1!, row.median!, row.q3!, row.p95!]),
        itemStyle: { color: c.hvArea, borderColor: c.hv },
      },
    ],
  };
}
