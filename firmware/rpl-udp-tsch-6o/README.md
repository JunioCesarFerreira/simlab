# Multi-Objective RPL Metrics Collection with Contiki-NG

This project provides a simulation framework based on Contiki-NG for collecting execution and communication metrics in RPL-based Wireless Sensor Networks (WSNs). The goal is to support research in multi-objective optimization by enabling realistic measurement of energy consumption, latency, packet flow, and link quality.

## 🧭 Overview

The system comprises two types of motes:

- **UDP Root (Sink):** Acts as the RPL root. It periodically sends PING packets (probes) to all known nodes and receives structured responses (PONG + metrics).
- **UDP Node (Sensor):** Joins the RPL network and replies to root probes with metrics about internal execution, energy use, and network statistics.

All communication is done over UDP using IPv6 and TSCH MAC scheduling. Latency is measured via a ping-pong RTT estimation technique.

## 📂 File Structure

- `root.c` — Logic for the root node. Initializes RPL DAG, sends probes, receives metrics, computes RTT, and logs results.
- `node.c` — Logic for the sensor nodes. Measures energy, connectivity, and communication stats, and replies with a `node_metrics_packet_t`.
- `metrics-packet.h` — Defines the shared packet structures used between root and node (e.g., `ping_packet_t`, `node_metrics_packet_t`).
- `project-conf.h` — Enables energy tracking using `ENERGEST_CONF_ON` and configures MAC/PHY behavior.
- `Makefile` — Defines build rules for simulation in Cooja or native deployment.

## 📊 Metrics Collected

Each node periodically sends a `node_metrics_packet_t` containing:

- `cpu_energy_mJ` — CPU energy consumption in millijoules
- `lpm_energy_mJ` — Low Power Mode energy consumption
- `radio_tx_energy_mJ` — Radio transmission energy
- `radio_rx_energy_mJ` — Radio reception energy
- `total_sent` / `total_received` — Packet counters
- `bytes_tx` / `bytes_rx` — Byte counters
- `from_root_to_node_latency` — Latency estimate (root → node)
- `last_rssi` — Received Signal Strength Indicator (in dBm)
- `last_lqi` — Link Quality Indicator (0–255)

In addition, the root computes:

- RTT (ping-pong round-trip time)
- One-way latency estimate (RTT / 2)
- Hop count based on IPv6 TTL

## ⚙️ How It Works

1. The root starts the simulation as the **RPL DAG root** and opens a UDP socket.
2. Sensor nodes join the RPL network using the TSCH MAC layer.
3. Periodically, the root sends a `ping_packet_t` to each known node.
4. Each node:
   - Measures energy and communication metrics since the last report.
   - Responds with a `node_metrics_packet_t` and echoes the ping.
5. The root:
   - Measures RTT from the echoed ping.
   - Extracts hop count from TTL.
   - Logs all metrics as JSON-formatted or tabular output (configurable).

## 🧪 Simulation in Cooja

This project is fully compatible with [Cooja](https://github.com/contiki-ng/cooja), the Contiki-NG network simulator.

### To build and run:

1. Open **Cooja** and create a new simulation.
2. Import this project and add one **root** mote and multiple **nodes**.
3. Use the provided `Makefile` to compile `root.c` and `node.c` for `cooja`.
4. Observe real-time metric logging via the Cooja serial output.

## ✅ Requirements

- Contiki-NG (latest stable release)
- Cooja Simulator (built from Contiki-NG or Docker image)
- GCC toolchain (for native builds or automated tests)

## 📘 Notes

- All metrics are printed in structured format (JSON) to ease data analysis.
- The project is suitable for use in WSN benchmark design, protocol evaluation, and cross-layer optimization studies.
- Energy values are approximated using Contiki-NG’s `energest` counters and estimated power draw (mW).

---

See also [README DEV](README_dev.md)

---