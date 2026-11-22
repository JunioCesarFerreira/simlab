# Firmware

Contiki-NG applications that SimLab runs in Cooja motes. Each directory is one
source repository: a `root.c` (RPL root, Cooja mote 1) and a `node.c` (sensor
node) that share `metrics-packet.h`, `project-conf.h` and a `Makefile`. The root
prints one JSON line per report it receives; `pylib/cooja_builder/csv_converter.py`
turns those lines into the CSV the data conversion reads.

How each metric is obtained, what it measures and its limits:
[docs/markdown/FIRMWARE_METRICS.md](../docs/markdown/FIRMWARE_METRICS.md).

## Directories

| Directory | Version | MAC | Status |
|---|---|---|---|
| `rpl-udp-csma` | v1 | CSMA | Frozen. Used by 28,898 of the 29,630 simulations in the reference database. |
| `rpl-udp-tsch` | v1 | TSCH | Frozen. Used by 92 simulations. |
| `rpl-udp-csma-v2` | v2 | CSMA | Current. Not yet used by an experiment. |
| `rpl-udp-tsch-v2` | v2 | TSCH | Current. Not yet used by an experiment. |
| `rpl-udp-tsch-6o` | legacy | TSCH | Not used and not uploaded. Its `Makefile` builds `udp-client` and `udp-server`, which are not in the directory, so it does not build as it stands. Its JSON columns differ from v1 and v2. |

`repository_example.json` is an example of two repository records as SimLab stores
them: the repository id, name and description, and the id of each source file.

## v1: `rpl-udp-csma`, `rpl-udp-tsch`

Every node sends the root a metrics report every 10 s plus a jitter. The root
probes every node with a ping every 10 s; the node answers with a pong. The root
prints, per report or pong, the node's energy for the last interval, its
counters, three latencies derived from the probes (`rtt_latency`, `r2n_latency`,
`n2r_latency`) and the hop count.

Finished experiments and the published papers ran v1. Do not edit these
directories: the firmware snapshot of every finished experiment holds their digests, and `pylib/tests/test_firmware_v2.py` pins them;
a change goes into a new directory. Several v1 metrics do not measure what their
names say (FIRMWARE_METRICS.md, section 7).

## v2: `rpl-udp-csma-v2`, `rpl-udp-tsch-v2`

The same application as v1 with the measurement fixed. The two directories hold
the same `node.c`, `root.c`, `metrics-packet.h` and `project-conf.h`, byte for
byte; the `Makefile`s differ only in `MAKE_MAC = MAKE_MAC_TSCH`. Each record
carries `"fw_version":2`.

### Changes from v1

**Same application for both MACs**

- The CSMA and TSCH variants share their sources, so the MAC layer is the only
  difference between them. The v1 variants had drifted apart.

**Traffic**

- No probes. The root sends nothing to the nodes; the only application traffic
  is the reports. In v1 the root's burst of pings overflowed the two CSMA
  neighbour queues and was mostly dropped, and under TSCH the probes took the
  shared cell the reports needed (617 reports delivered with probes against
  6,416 without, on the same instance).
- The report jitter is drawn again for every period, in [0, 1 s). v1 drew it
  once, so two nodes with close draws collided in every period.
- Under TSCH a node does not queue a report while it is not associated.

**Latency**

- One latency: the one-way delay of each report, node to root (`latency`, ms).
  The node writes the simulation time into the report (`RTIMER_NOW()`, the same
  on every mote and under both MACs); the root subtracts it from the time the
  report arrives.
- `rtt_latency`, `r2n_latency` and `n2r_latency` are gone. They came from the
  probes, and the one-way pair subtracted `clock_time()` readings of different
  motes, which start at each mote's own boot, up to 1 s apart.
- The firmware refuses to build outside Cooja (`#error`), where the simulation
  clock is not shared between motes.

**Energy**

- The node reports cumulative Energest times since boot and never resets them.
  The root computes each interval's energy from the difference with the newest
  report it has. A lost report no longer loses the energy of its interval; v1
  zeroed the node's counters at each report.
- A report that arrives after a later one (TSCH reorders) adds no energy; the
  root tells them apart by the report's sequence number.
- Energy is in mJ and includes the supply voltage (3 V). v1 multiplied current by
  time and labelled the result mJ.
- Energy is kept with three decimals. v1 truncated each interval to whole units,
  which recorded about a quarter of the TX energy.
- The currents and voltage (`ENERGY_*` in `root.c`) can be overridden in
  `project-conf.h`; the nodes do not use them.
- New column `energy_time_ms`: the length of the interval the energy columns
  cover, so that energy over time gives a mean power (`power_mw`).

**Counters**

- `bytes_tx` and `server_bytes_rx` accumulate. In v1 they held the size of the
  last packet, so their growth was always 0.
- `server_received` counts reports only. In v1 it also counted pongs, about a
  third of the total.
- `hops` counts links (`UIP_TTL − ttl + 1`): 1 for a direct child of the root.
  v1 counted relays, one less.
- New columns: `tx_time_ms` and `rx_time_ms` (cumulative radio time),
  `reports_lost` (reports generated minus received, cumulative).
- Removed with the probes: `node_time`, `total_received`, `bytes_rx`,
  `server_sent`.

**Link quality**

- `rssi` and `lqi` are read when the report is generated. v1 read them when a
  ping arrived.

**Robustness**

- Every packet starts with a type byte, and every field has a fixed width
  (`stdint` types). v1 told its packets apart by length and used `unsigned int`
  and `unsigned long`, whose size depends on the platform.
- The root keeps counters for up to 200 nodes (`ROOT_MAX_MOTES`) and ignores
  further nodes with one warning. v1 kept 100 and dereferenced a null pointer
  for the 101st.
- The node prints its address once, not every period.

**TSCH schedule**

- `rpl-udp-tsch-v2` uses one shared cell every 3 slots
  (`TSCH_SCHEDULE_CONF_DEFAULT_LENGTH 3`, the value of Contiki-NG's 6TiSCH
  example). With Contiki-NG's default of 7, 26 nodes reporting every 10 s
  congest the shared cell: the queues toward the relays fill up, RPL's DAO-ACKs
  time out, and nodes drop their route and stop reporting for minutes. With 3,
  TSCH delivers what CSMA delivers, at a fifth of the power, in all six runs
  measured (two topologies of 26 nodes, three seeds each; FIRMWARE_METRICS.md,
  section 8). With 7, the same runs delivered 46 to 321 reports per node against
  about 890. The setting is in the shared
  `project-conf.h`; CSMA builds do not read it.

**Simulation speed**

- The neighbour table and the root's route table hold 64 entries
  (`NBR_TABLE_CONF_MAX_NEIGHBORS`, `NETSTACK_MAX_ROUTE_ENTRIES` in
  `project-conf.h`) instead of the Cooja platform's 300. The output does not
  change, and simulations run several times faster. Details below.

### Simulation speed

On every tick of a mote, Cooja copies the mote's whole RAM (`.data` and `.bss`)
from Java into the native library and back (`ContikiMote.execute`). It has to,
because all motes of a type share one library. TSCH ticks far more often than
CSMA, since every node wakes up in every active slot, so the copy dominates.
Sampling the simulation thread's stack (120 samples, `jcmd Thread.print`) found
it copying memory 87% of the time and running Contiki at most 5%.

The Cooja platform sizes the neighbour and route tables for 300 entries. With
300, the `.bss` of a TSCH node is 272 KB, and 158 KB of that is the TSCH
neighbour queues (300 neighbours × 64 queue slots). With 64 it is 118 KB. Cooja
still spends 77% of its time copying, but each copy is much smaller.

The output is identical, bit for bit, as long as no table fills. That holds up
to **63 motes, root included**:

- All neighbour tables share one pool of `NBR_TABLE_CONF_MAX_NEIGHBORS`
  addresses: IPv6 neighbours, RPL neighbours, link statistics and TSCH
  neighbours. A node can hold every other mote, because an entry outlives a
  neighbour that moves out of range. TSCH also adds two virtual neighbours (the
  EB and broadcast addresses). That makes motes + 1 entries.
- The root's source-routing table holds every node of the DODAG and the root
  itself, motes entries in all.

Past 63 motes a full table evicts a neighbour, and runs diverge from the
300-entry build. The largest problem in the reference database has 53 nodes.
The identity also relies on RPL Lite. RPL Classic's neighbour policy reads the
table capacity directly, so under RPL Classic the capacity would matter even
when the table is not full.

Measured on 2026-11-21. Each run had 26 nodes plus the root and lasted 180
simulated minutes. Every run printed the same mote lines as the 300-entry build
at the same simulation times, in the same order:

| Instance | Lines | Wall time, 64 entries |
|---|---|---|
| TSCH, topology A, seed 987654 | 27,064 | 1,167 s |
| TSCH, topology B, seed 123456 | 27,062 | 820 s |
| TSCH, topology A, seed 336157 | 27,047 | 816 s |
| CSMA, topology A, seed 987654 | 26,864 | 168 s |
| CSMA, topology B, seed 123456 | 26,861 | 160 s |

The runs shared the machine with eight production simulations. Under that load,
the 300-entry TSCH build advanced 1.8 simulated seconds per wall-clock second,
about 100 minutes per run, and the 64-entry build 9 to 13. The 300-entry build is
slow mostly because of memory-bandwidth contention, so the gain is smaller on an
idle machine: there, the 300-entry build took 1,063 s on 2026-11-15.

### Output

One JSON line per report received, on the root's serial port:

```text
node, fw_version,
cpu_energy_mj, lpm_energy_mj, radio_tx_energy_mj, radio_rx_energy_mj, total_energy_mj, energy_time_ms,
tx_time_ms, rx_time_ms, total_sent, bytes_tx,
latency, hops, lqi, rssi,
server_received, server_bytes_rx, reports_lost, root_time_now
```

The default data conversion (`gui/simlab/src/services/defaultDataConversion.ts`)
reads these columns. Run with v1, its latency metrics and `power_mw` are NaN;
FIRMWARE_METRICS.md, section 9, says what to change.

## Configuration

| Setting | Where | Value |
|---|---|---|
| Report period | `BASE_INTERVAL`, `JITTER` in `node.c` | 10 s + [0, 1 s) |
| Energest | `ENERGEST_CONF_ON` in `project-conf.h` | on |
| TSCH shared cell (v2) | `TSCH_SCHEDULE_CONF_DEFAULT_LENGTH` in `project-conf.h` | every 3 slots |
| Energy model (v2) | `ENERGY_SUPPLY_V`, `ENERGY_CPU_MA`, `ENERGY_LPM_MA`, `ENERGY_TX_MA`, `ENERGY_RX_MA` in `root.c` | 3.0 V; 1.8, 0.0545, 17.4, 19.7 mA |
| Root node table (v2) | `ROOT_MAX_MOTES` in `root.c` | 200 |
| Neighbour and route tables (v2) | `NBR_TABLE_CONF_MAX_NEIGHBORS`, `NETSTACK_MAX_ROUTE_ENTRIES` in `project-conf.h` | 64 (output identical up to 63 motes) |
| UDP ports | `node.c`, `root.c` | node 8765, root 5678 |

The radio values (17.4 and 19.7 mA) were inherited from v1 and have not been
checked against the CC2420 datasheet.

## Building and uploading

SimLab copies a repository's files into Cooja's directory in the
`simulation-cooja` image and builds them from the simulation file
(`make root.cooja TARGET=cooja`, `make node.cooja TARGET=cooja`); the
`Makefile`s expect Contiki-NG two levels up (`CONTIKI=../..`).

To register repositories with SimLab, run from this directory:

```sh
python upload-firmware.py rpl-udp-csma-v2 rpl-udp-tsch-v2   # or no argument for all four
```

`SIMLAB_API_BASE`, `SIMLAB_API_KEY` and `SIMLAB_CA_BUNDLE` / `SIMLAB_TLS_VERIFY`
select the API and its credentials. An experiment uses v2 when it is launched
with the `-v2` repositories; finished experiments are not affected.

## Rules for changes

- Do not edit v1. Put a change in a new directory and give it a version.
- Keep the two v2 variants identical except for `MAKE_MAC` in the `Makefile`.
- Keep the root printing every column the default data conversion reads.

`pylib/tests/test_firmware_v2.py` checks all three, the TSCH schedule setting,
the table sizes, and runs real v2 log lines through the converter and the default
conversion.

If you change `project-conf.h` for speed, check that the output does not change.
Replay a few instances with the old and new builds, and compare the root's
lines and their `t_us` one by one.
