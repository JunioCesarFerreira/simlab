import { describe, it, expect } from "vitest";
import { METRIC_KINDS, isMetricItemComplete, metricKind } from "../metricKinds";

describe("metricKinds", () => {
  it("offers no kind twice", () => {
    const values = METRIC_KINDS.map((k) => k.value);
    expect(new Set(values).size).toBe(values.length);
  });

  it("does not offer the kinds the evaluator never had", () => {
    // max, min, last and percentile were in the wizard and not in pylib/statistics.py:
    // a metric that used one became NaN without failing the run.
    for (const k of ["max", "min", "last", "percentile"]) expect(metricKind(k)).toBeUndefined();
  });

  it("accepts a complete item", () => {
    expect(isMetricItemComplete({ name: "latency", kind: "median", column: "latency" })).toBe(true);
  });

  it("refuses an unknown kind or an empty name or column", () => {
    expect(isMetricItemComplete({ name: "x", kind: "percentile", column: "latency" })).toBe(false);
    expect(isMetricItemComplete({ name: " ", kind: "mean", column: "latency" })).toBe(false);
    expect(isMetricItemComplete({ name: "x", kind: "mean", column: "" })).toBe(false);
  });

  it("requires a denominator for the ratio kinds", () => {
    const item = { name: "p", kind: "ratio_sum", column: "total_energy_mj" };
    expect(isMetricItemComplete(item)).toBe(false);
    expect(isMetricItemComplete({ ...item, denominator: "energy_time_ms" })).toBe(true);
  });

  it("takes q between 0 and 1, or none (the evaluator then uses 0.5)", () => {
    const item = { name: "p95", kind: "quantile", column: "latency" };
    expect(isMetricItemComplete(item)).toBe(true);
    expect(isMetricItemComplete({ ...item, q: 0.95 })).toBe(true);
    expect(isMetricItemComplete({ ...item, q: 95 })).toBe(false);
    expect(isMetricItemComplete({ ...item, q: 0 })).toBe(false);
  });
});
