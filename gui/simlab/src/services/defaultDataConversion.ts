import type { DataConversionConfigDto } from "../types/simlab";

/**
 * How a Cooja log is turned into metrics, unless an experiment says otherwise.
 *
 * Shared by the experiment wizard and the scalability benchmark. The benchmark
 * does not read these metrics — its analysis uses only timestamps — but the
 * master-node evaluates them after every run, and a run whose metrics cannot
 * be evaluated ends in error. Two copies of this list would drift, and the one
 * that drifted would make a benchmark of failures.
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
      { name: "total_sent", kind: "sum_last_minus_first", column: "total_sent" },
      { name: "total_received", kind: "sum_last_minus_first", column: "total_received" },
      { name: "server_sent", kind: "sum_last_minus_first", column: "server_received" },
      { name: "bytes_tx", kind: "sum_last_minus_first", column: "bytes_tx" },
      { name: "bytes_rx", kind: "sum_last_minus_first", column: "bytes_rx" },
      { name: "server_bytes_rx", kind: "sum_last_minus_first", column: "server_bytes_rx" },
      { name: "r2n_latency", kind: "mean", column: "r2n_latency" },
      { name: "n2r_latency", kind: "mean", column: "n2r_latency" },
      { name: "hops", kind: "mean", column: "hops" },
      { name: "rtt_latency", kind: "mean", column: "rtt_latency" },
      { name: "latency", kind: "mean", column: "rtt_latency" },
      { name: "energy", kind: "sum_all", column: "total_energy_mj" },
      { name: "throughput", kind: "sum_last_minus_first", column: "server_received" },
    ],
  };
}
