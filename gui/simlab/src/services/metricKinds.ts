import type { MetricItem } from "../types/simlab";

/**
 * The kinds the master-node can evaluate (pylib/statistics.py, STAT_DISPATCH).
 * The two lists must match: a kind the evaluator does not know turns the metric
 * into NaN without failing the run (pylib/tests/test_statistics.py checks it).
 */
export interface MetricKind {
  value: string;
  /** One line shown under the select. */
  hint: string;
  /** Item fields this kind reads besides name and column. */
  uses: ReadonlyArray<"q" | "scale" | "denominator">;
}

export const METRIC_KINDS: ReadonlyArray<MetricKind> = [
  { value: "mean", hint: "Mean of the column over all rows; empty cells are skipped.", uses: [] },
  { value: "median", hint: "Median of the column over all rows.", uses: [] },
  { value: "quantile", hint: "Quantile q of the column over all rows; q from 0 to 1 (0.95 = 95th percentile).", uses: ["q"] },
  { value: "sum_all", hint: "Sum of the column over all rows.", uses: [] },
  { value: "sum_max", hint: "Per node, its largest value, summed over nodes. The total of a counter that only grows.", uses: [] },
  { value: "sum_last_minus_first", hint: "Per node, last value minus first (time order), summed over nodes. Leaves out each node's first row.", uses: [] },
  { value: "sum_rate", hint: "Sum of the column divided by the time the log spans, per second.", uses: [] },
  { value: "ratio_sum", hint: "scale × sum of the column ÷ sum of the denominator column, over all rows.", uses: ["denominator", "scale"] },
  { value: "ratio_sum_max", hint: "scale × sum_max of the column ÷ sum_max of the denominator column.", uses: ["denominator", "scale"] },
  { value: "node_coverage", hint: "Distinct values of the column (nodes) ÷ sensor nodes in the simulation.", uses: [] },
  { value: "inverse_median", hint: "scale ÷ median of the column.", uses: ["scale"] },
];

export function metricKind(value: string): MetricKind | undefined {
  return METRIC_KINDS.find((k) => k.value === value);
}

/** Whether the master-node can evaluate this item as written. */
export function isMetricItemComplete(m: MetricItem): boolean {
  const kind = metricKind(m.kind);
  if (!kind || !m.name.trim() || !m.column.trim()) return false;
  if (kind.uses.includes("denominator") && !(m.denominator ?? "").trim()) return false;
  if (kind.uses.includes("q") && m.q !== undefined && !(m.q > 0 && m.q <= 1)) return false;
  return true;
}
