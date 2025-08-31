# Developer Guide – Multi-Objective RPL Metrics Framework

This guide provides technical instructions and implementation notes for developers working with the RPL-based metrics collection system built on Contiki-NG. It includes build setup, file structure, debugging options, and post-processing instructions.

---

## 🛠 Build Instructions

### Cooja (Simulation)
1. Open **Cooja** (from Contiki-NG).
2. Compile each source file for simulation:
```sh
make node.c TARGET=cooja
make root.c TARGET=cooja
```

3. In Cooja:

   * Create a new simulation.
   * Add one mote compiled with `root.c`.
   * Add one or more motes compiled with `node.c`.
   * Use the serial output window to observe results.

### Native (Command-Line or Test Scripts)

For headless simulation (e.g., `native` platform):

```sh
make node TARGET=native
make root TARGET=native
```

---

## 📁 Codebase Structure

| File               | Purpose                                                                      |
| ------------------ | ---------------------------------------------------------------------------- |
| `root.c`           | Root node behavior: RPL DAG root, ping emitter, receiver, logger             |
| `node.c`           | Sensor node behavior: probe responder, energy and network metrics collection |
| `metrics-packet.h` | Defines `ping_packet_t` and `node_metrics_packet_t` used in UDP payloads     |
| `project-conf.h`   | Contiki system configuration (e.g., MAC, `ENERGEST_CONF_ON`)                 |
| `Makefile`         | Target-specific build configuration                                          |

---

## 🔍 Debugging Options

Enable debug flags by uncommenting them in `node.c` or `root.c`:

```c
//#define DEBUG_ENERGY_TIME_IN_SECONDS
//#define DEBUG_PRINT_METRICS_PACKET
//#define DEBUG_RX_CALLBACK
//#define PRINT_TAB_LOG
#define PRINT_JSON_LOG
```

* `DEBUG_ENERGY_TIME_IN_SECONDS`: prints raw energy times in seconds
* `PRINT_JSON_LOG`: emits metrics as a one-line JSON (recommended for parsing)
* `PRINT_TAB_LOG`: emits human-readable logs for manual inspection

---

## 📊 Post-Processing and Analysis

### Example: Load metrics in Python

If logging is set to `PRINT_JSON_LOG`, you can process the output with:

```python
import pandas as pd

df = pd.read_json("cooja_output.log", lines=True)

# Filter or analyze
print(df.groupby("node")[["cpu_energy_mj", "rtt_latency"]].mean())
```

### Sample metrics fields:

| Field                | Description                                      |
| -------------------- | ------------------------------------------------ |
| `cpu_energy_mj`      | Energy in mJ spent in CPU active mode            |
| `lpm_energy_mj`      | Energy in mJ spent in low-power mode             |
| `radio_tx_energy_mj` | TX radio energy in mJ                            |
| `radio_rx_energy_mj` | RX radio energy in mJ                            |
| `r2n_latency`        | Root-to-node latency (in ticks)                  |
| `rtt_latency`        | Round-trip time divided by 2 (estimated latency) |
| `hops`               | Number of hops from root to node                 |
| `last_rssi`          | Last recorded RSSI in dBm                        |
| `last_lqi`           | Last recorded Link Quality Indicator (0–255)     |

---

## 🧩 Extending This Project

This framework is extensible for:

* MQTT or CoAP integration
* Real node deployment (e.g., Zolertia, Sky)
* Other MAC layers (CSMA, nullmac, LMAC)
* New metrics (battery voltage, memory, congestion)

To add new metrics:

1. Extend `node_metrics_packet_t` in `metrics-packet.h`
2. Update `fill_node_metrics_packet()` in `node.c`
3. Update `metrics_print()` in `root.c` for logging

---

## 📚 References

* Contiki-NG Docs: [https://contiki-ng.readthedocs.io/](https://contiki-ng.readthedocs.io/)
* IEEE 802.15.4 & TSCH specs
* RPL: RFC 6550 (Routing Protocol for LLNs)

---

Feel free to open issues or submit pull requests with new metrics, analyses, or improvements to simulation realism.

