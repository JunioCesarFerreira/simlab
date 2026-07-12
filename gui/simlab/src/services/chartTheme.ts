/**
 * Shared light/dark theme tokens for the ECharts-based components.
 * Single source of truth — HvGdChart, ExperimentsComparison and the 3-D
 * charts used to each carry their own copy of these values.
 */

/** Palette for the 2-D (canvas/svg) charts. */
export function chartPalette(dark: boolean) {
  return {
    bg: "transparent",
    text: dark ? "#cdd6f4" : "#334155",
    muted: dark ? "#6c7086" : "#94a3b8",
    grid: dark ? "#313244" : "#e2e8f0",
    tooltip: dark ? "#1e1e2e" : "#ffffff",
    tooltipBorder: dark ? "#313244" : "#e2e8f0",
    // Convergence metrics (HV / GD)
    hv: dark ? "#89b4fa" : "#2563eb",
    gd: dark ? "#f38ba8" : "#dc2626",
    hvArea: dark ? "rgba(137,180,250,0.12)" : "rgba(37,99,235,0.08)",
    gdArea: dark ? "rgba(243,139,168,0.12)" : "rgba(220,38,38,0.08)",
    // Two-experiment comparison accents (A = blue, B = orange)
    colorA: dark ? "#89b4fa" : "#3b82f6",
    colorB: dark ? "#fab387" : "#f97316",
    areaA: dark ? "rgba(137,180,250,0.12)" : "rgba(59,130,246,0.10)",
    areaB: dark ? "rgba(250,179,135,0.12)" : "rgba(249,115,22,0.10)",
  };
}

/** Color tokens for the echarts-gl 3-D scenes. */
export function gl3dColors(dark: boolean) {
  return {
    axis: dark ? "#45475a" : "#c0cad8",
    label: dark ? "#7f849c" : "#64748b",
    name: dark ? "#a6adc8" : "#374151",
    split: dark ? "#313244" : "#dde3eb",
    bg: dark ? "#181825" : "#ffffff",
    env: dark ? "#181825" : "#ffffff",
  };
}

/**
 * Standard value-axis config for the 3-D scatter scenes.
 * echarts-gl's axis options are untyped upstream, hence the `any`.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function gl3dAxis(name: string, dark: boolean): any {
  const c = gl3dColors(dark);
  return {
    name,
    nameTextStyle: { color: c.name, fontSize: 12, fontWeight: 600 },
    type: "value",
    axisLine: { lineStyle: { color: c.axis, width: 1.5 } },
    axisTick: { lineStyle: { color: c.axis } },
    axisLabel: { color: c.label, fontSize: 11 },
    splitLine: { lineStyle: { color: c.split, opacity: 0.6 } },
  };
}
