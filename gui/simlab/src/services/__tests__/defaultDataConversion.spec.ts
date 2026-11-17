import { describe, it, expect } from "vitest";
import { defaultDataConversion } from "../defaultDataConversion";
import { isMetricItemComplete } from "../metricKinds";

// Metrics whose name is deliberately not the name of the column they read.
const DELIBERATE_ALIASES: Record<string, string> = {
  energy: "total_energy_mj",
  power_mw: "total_energy_mj",
  throughput: "server_received",
  delivery_ratio: "server_received",
  latency_mean: "latency",
  latency_p95: "latency",
  node_coverage: "node",
};

describe("defaultDataConversion", () => {
  const metrics = defaultDataConversion().metrics;

  it("reads, for each metric, the column of the same name unless the alias is deliberate", () => {
    for (const m of metrics) {
      const expected = DELIBERATE_ALIASES[m.name] ?? m.name;
      expect(m.column, `metric '${m.name}'`).toBe(expected);
    }
  });

  it("reads one latency column, the one-way delay of the reports, and optimises its median", () => {
    // v1 had rtt_latency, r2n_latency and n2r_latency: obtained from probes the root sent, and not
    // comparable between CSMA and TSCH. Reintroducing any of them would put them back in every experiment.
    const latencies = metrics.filter((m) => m.name.includes("latency") || m.column.includes("latency"));
    expect(new Set(latencies.map((m) => m.column))).toEqual(new Set(["latency"]));
    expect(latencies.find((m) => m.name === "latency")?.kind).toBe("median");
  });

  it("says how much of the network the other metrics describe", () => {
    const byName = Object.fromEntries(metrics.map((m) => [m.name, m]));
    expect(byName.node_coverage?.kind).toBe("node_coverage");
    expect(byName.delivery_ratio).toMatchObject({ kind: "ratio_sum_max", denominator: "total_sent" });
    expect(byName.power_mw).toMatchObject({ kind: "ratio_sum", denominator: "energy_time_ms", scale: 1000 });
  });

  it("has no metric name twice", () => {
    const names = metrics.map((m) => m.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it("passes the wizard's own validation", () => {
    for (const m of metrics) expect(isMetricItemComplete(m), m.name).toBe(true);
  });
});
