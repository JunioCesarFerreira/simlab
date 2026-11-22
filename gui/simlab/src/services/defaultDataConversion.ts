import type { DataConversionConfigDto } from "../types/simlab";

/**
 * How a Cooja log is turned into metrics, unless an experiment says otherwise.
 * It reads the columns of firmware v2 (docs/markdown/FIRMWARE_METRICS.md).
 *
 * Shared by the experiment wizard and the scalability benchmark. The benchmark
 * does not read these metrics — its analysis uses only timestamps — but the
 * master-node evaluates them after every run. A column the firmware does not
 * print gives NaN without failing the run, so a second copy of this list would
 * drift in silence.
 *
 * Latency, energy and the counters only exist for reports that reached the
 * root, so the list carries how much of the network they describe:
 * node_coverage (nodes heard from) and delivery_ratio (reports that arrived).
 */
export function defaultDataConversion(): DataConversionConfigDto {
  return {
    node_col: "node",
    time_col: "root_time_now",
    metrics: [
      { name: "cpu_energy_mj", kind: "sum_all", column: "cpu_energy_mj" },
      { name: "lpm_energy_mj", kind: "sum_all", column: "lpm_energy_mj" },
      { name: "radio_tx_energy_mj", kind: "sum_all", column: "radio_tx_energy_mj" },
      { name: "radio_rx_energy_mj", kind: "sum_all", column: "radio_rx_energy_mj" },
      // Counters since boot, never reset: each node's largest value is its total.
      { name: "total_sent", kind: "sum_max", column: "total_sent" },
      { name: "bytes_tx", kind: "sum_max", column: "bytes_tx" },
      { name: "server_bytes_rx", kind: "sum_max", column: "server_bytes_rx" },
      { name: "hops", kind: "mean", column: "hops" },
      // One-way delay of the reports, node to root, on the simulation clock.
      // The median is the objective: under TSCH a few reports wait seconds and
      // pull the mean far from the typical report. Mean and 95th percentile are
      // kept to show the tail.
      // Firmware v2 only: v1 (rpl-udp-csma, rpl-udp-tsch) has no such column, its
      // closest is `rtt_latency`; point these metrics at it in step 5 to run v1.
      { name: "latency", kind: "median", column: "latency" },
      { name: "latency_mean", kind: "mean", column: "latency" },
      { name: "latency_p95", kind: "quantile", column: "latency", q: 0.95 },
      { name: "energy", kind: "sum_all", column: "total_energy_mj" },
      // Mean power of a node that reported, in mW: energy over the node-time it
      // was spent in. Unlike `energy` it does not grow with how many nodes were
      // heard from or for how long. v1 has no energy_time_ms: use `energy` there.
      { name: "power_mw", kind: "ratio_sum", column: "total_energy_mj", denominator: "energy_time_ms", scale: 1000 },
      { name: "throughput", kind: "sum_max", column: "server_received" },
      // Reports that reached the root over reports the nodes generated, and the
      // fraction of the sensor nodes whose reports reached it at all.
      { name: "delivery_ratio", kind: "ratio_sum_max", column: "server_received", denominator: "total_sent" },
      { name: "node_coverage", kind: "node_coverage", column: "node" },
    ],
  };
}
