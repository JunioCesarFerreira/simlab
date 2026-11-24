# RPL metrics over TSCH (legacy firmware)

> **Legacy.** SimLab neither uses nor uploads this firmware, and the default
> data conversion does not read its columns. The current firmware is in
> [firmware/README.md](../README.md), and how each of its metrics is obtained in
> [FIRMWARE_METRICS.md](../../docs/markdown/FIRMWARE_METRICS.md).

A Contiki-NG application that measures energy, latency and traffic in an RPL
network over TSCH.

## Overview

- **Root (sink):** the RPL root. Every 10 s it pings every node it has heard
  from, times the echo, and prints one JSON line per report received.
- **Node (sensor):** every 10 s, plus a jitter below 1 s drawn once at start,
  sends a `node_metrics_packet_t` to the root. It answers a ping by echoing it
  (the echo carries no metrics).

## File structure

- `root.c` — RPL root, ping emitter, receiver and logger.
- `node.c` — periodic report, ping echo, energy accounting.
- `metrics-packet.h` — `ping_packet_t` and `node_metrics_packet_t`.
- `project-conf.h` — only enables Energest (`ENERGEST_CONF_ON`).
- `Makefile` — selects TSCH.

## What the root prints

| Column | Meaning |
|---|---|
| `node` | the node's IPv6 address |
| `cpu_mj` | CPU term of the last interval (see below) |
| `total_mj` | sum of the CPU, LPM, TX and RX terms of the last interval |
| `latency_ms` | root→node latency the node measured on the last ping, in uptime ticks (1 ms in Cooja) |
| `volume_B` | application bytes the node sent and received, cumulative |
| `response_ms` | round trip of the last ping; `1` before the first one |
| `throughput_bps` | new bytes since the previous report × 8000 / `response_ms`. It is bytes over one round trip, not a rate over the interval |
| `root_time_now` | the root's uptime ticks at reception |

## Limitations

- **The `*_mj` values are not millijoules**: seconds × mA (17.4, 19.7, 1.8,
  0.0545) gives mA·s, with no supply voltage, and each term is truncated to an
  integer.
- The packet carries RSSI and LQI, and the root reads the IPv6 TTL as `hops`,
  but none of them is printed. The packet's `total_sent`, `total_received`,
  `bytes_tx` and `bytes_rx` are never filled.
- The root's table holds 100 nodes; a packet from one more node dereferences a
  null pointer.

---

Building, flags and post-processing: [README_dev.md](README_dev.md)
