# Firmware metrics — how each metric is obtained

Every number SimLab reports about a simulated network — latency, energy,
throughput and the rest — is measured by the firmware that runs in the Cooja
motes and then reduced by the data conversion. This document follows each
metric from the mote to the objective, says what it does and does not measure,
what changed between the two firmware generations, and what can be compared
between the CSMA and TSCH MACs (section 8).

Read it before quoting a metric in a paper: several of the numbers produced by
firmware v1, which the published results used, do not mean what their
names suggest (section 7).

## 1. Firmware versions

| Directory | Version | MAC | State |
|---|---|---|---|
| `firmware/rpl-udp-csma` | v1 | CSMA | **Frozen.** Run by 28,898 of the 29,630 simulations in the reference database (2026-11-13, before v2 existed). |
| `firmware/rpl-udp-tsch` | v1 | TSCH | **Frozen.** Run by 92 simulations (same count). |
| `firmware/rpl-udp-csma-v2` | v2 | CSMA | Current. |
| `firmware/rpl-udp-tsch-v2` | v2 | TSCH | Current. One shared cell every 3 slots (§8.1). |
| `firmware/rpl-udp-tsch-6o` | legacy | TSCH | No experiment used it; not uploaded by `upload-firmware.py`. Different JSON columns, so the default conversion does not read it. |

[firmware/README.md](../../firmware/README.md) describes each directory and lists every
change v2 makes to v1.

**v1 is not edited.** The snapshot of every finished experiment holds the digest
of these files ([FIRMWARE_TRACEABILITY.md](FIRMWARE_TRACEABILITY.md)), and
`pylib/tests/test_firmware_v2.py` pins them, so a change in place fails the test
instead of silently breaking the link between a result and its code.

The two v2 directories contain the same `node.c`, `root.c`, `metrics-packet.h`
and `project-conf.h`, byte for byte (a test checks it). Only the `Makefile`
differs, by `MAKE_MAC = MAKE_MAC_TSCH`. A CSMA-versus-TSCH comparison with v2
therefore changes the MAC layer and nothing else; with v1 the two variants had
drifted apart in small ways.

Every v2 record carries `"fw_version":2`, so a CSV says which firmware produced it.

## 2. From the mote to the objective

```text
node (every ~10 s)      cumulative counters + send time ──UDP report──▶ root (Cooja mote 1)
root                    arrival time − send time = latency; one JSON line per report, on the serial port
Cooja                   COOJA.testlog: "[t_us=…] [Mote:1] {…}"
csv_converter           one CSV row per JSON line   (pylib/cooja_builder/csv_converter.py)
data conversion         evaluate_config → network_metrics   (pylib/statistics.py)
seed aggregator Ψa      one value per objective, across the seeds of an individual
```

Facts about this chain that bound what any metric can say:

- **A row exists only for a report that reached the root.** A node that never
  joined the network, and a report lost on the way, produce no row. Everything
  downstream is computed from the rows that exist.
- The converter reads lines from **mote 1 only** (`[Mote:1]`), so the root must be
  mote id 1. Lines that fail to parse as JSON are skipped silently.
- Rows are sorted by node and `root_time_now` before being reduced.
- The Cooja template uses the UDGM radio with `success_ratio_tx/rx = 1.0`
  ([simulation_template.xml](../../mo-engine/simulation_template.xml)): a frame is
  never lost to random channel noise, only to collisions, full queues and
  schedule limits. A low variance between seeds means stability **in this radio
  model**, not in a real channel.
- `CLOCK_SECOND` is 1000 on the Cooja platform, so one clock tick is one
  millisecond. Both MACs report times in ms.

### 2.1 What a data conversion does

Each metric is `{name, kind, column}`: reduce the column `column` of the CSV with
the function `kind`. Some kinds take more: `q` (a quantile), `denominator` (a second
column), `scale` (a factor).

| `kind` | Reduction | Detail |
|---|---|---|
| `mean` | Mean of the column over all rows of all nodes | Empty cells (`null` in the JSON) are **skipped**. Rows are not weighted per node, so a node that reports more often counts more. |
| `median` | Median of the column over all rows | Same rows as `mean`. |
| `quantile` | Quantile `q` of the column over all rows, `q` from 0 to 1 | `q = 0.95` is the 95th percentile; 0.5 if `q` is absent. |
| `sum_all` | Sum of the column over all rows | Non-numeric cells count as zero. Used for quantities reported *per interval*. |
| `sum_max` | Per node, the largest value, then summed over nodes | The total of a *cumulative* counter. Counts every report, and a report that arrives after a later one (TSCH reorders) cannot lower it. |
| `sum_last_minus_first` | Per node, last value minus first value (rows in time order), negatives clipped to 0, then summed over nodes | Also for cumulative counters, but the first report of each node is the baseline and is not counted. Kept for v1 conversions. |
| `ratio_sum` | `scale` × sum of `column` ÷ sum of `denominator`, over all rows | A ratio of totals, not a mean of per-row ratios: a long interval weighs as much as it lasted. NaN if the denominator sums to 0. |
| `ratio_sum_max` | `scale` × `sum_max(column)` ÷ `sum_max(denominator)` | The ratio of two counters' totals. |
| `node_coverage` | Distinct values of `column` ÷ sensor nodes in the simulation | The master-node counts the motes that wrote to the testlog, root excluded (`rpl_dodag.count_deployed_nodes`), because the CSV has no row for a node that never reached the root. A run in which no report reached the root gives 0 here and in the counting kinds, NaN in the others ([master-node](../../master-node/README.md#results-that-lack-a-value)). |
| `sum_rate` | Sum of the column ÷ the time the rows span, per second | Not used by the default conversion. |
| `inverse_median` | `scale` ÷ median of the column | Not used by the default conversion. |

The wizard offers exactly these kinds (`gui/simlab/src/services/metricKinds.ts`; a test
compares the two lists). It used to offer `max`, `min`, `last` and `percentile`, which
the evaluator never had; no stored experiment used them.

A metric that cannot be computed (a missing column, an empty column, an unknown kind,
a ratio without its denominator) becomes NaN with a message in the master-node log,
and the simulation is still `Done`. The exception is a column that exists with nothing
numeric in it under `sum_all` or `sum_max`, which gives 0. If the NaN is an objective,
the engine counts that replica as missing for it: by default the individual is then
refused and gets the worst objectives
([the policy](../../mo-engine/lib/strategy/README.md#the-policy)). If no report at all
reached the root, the simulation is still `Done`: counts and `node_coverage` are 0, the
other metrics NaN
([master-node](../../master-node/README.md#results-that-lack-a-value)).

## 3. The columns of a record

The root prints one JSON line per report. v2 keeps the v1 name for every column
that means the same thing and drops the ones that came from the probes (last
row). The default conversion reads v2; to run v1, point `latency` at
`rtt_latency` in step 5 of the launch wizard.

| Column | v1 | v2 |
|---|---|---|
| `node` | Sender's global IPv6 address | same |
| `root_time_now` | Root clock when the line was printed, ms (CSMA: local clock; TSCH: network time) | same |
| `cpu_energy_mj`, `lpm_energy_mj`, `radio_tx_energy_mj`, `radio_rx_energy_mj`, `total_energy_mj` | Energy of **one interval**, computed on the node, **truncated to whole mJ**, in mA·s (§5.2) | Energy of one interval, computed on the root from cumulative times, in mJ with 3 decimals (§5.2) |
| `energy_time_ms` | *does not exist* | Length of that interval, ms: the node's CPU + LPM time between this report and the newest one before it (§5.2) |
| `total_sent` | Reports the node generated (cumulative) | same (the report's `seq`) |
| `bytes_tx` | **Size of the last packet**, not a total | Cumulative UDP payload bytes the node sent (§5.5) |
| `server_received` | Reports **and pongs** received from this node (cumulative) | **Reports only** (cumulative) |
| `server_bytes_rx` | Size of the packet that triggered the line | Cumulative payload bytes of the reports received from this node |
| `latency` | *does not exist* (v1 has `rtt_latency`, see below) | One-way delay of this report, node to root, ms (§5.1) |
| `hops` | Relays on the path (`64 − ttl`) | Links on the path (`UIP_TTL − ttl + 1`) (§5.4) |
| `lqi`, `rssi` | LQI/RSSI of the last frame the radio received when a ping arrived | same, read when the report is generated (§5.6) |
| *new in v2* | | `fw_version`, `tx_time_ms`, `rx_time_ms` (cumulative radio time), `reports_lost` (generated minus received, cumulative) |
| *only in v1* | `node_time`, `total_received`, `bytes_rx`, `server_sent` (pings, cumulative), `rtt_latency`, `r2n_latency`, `n2r_latency` | removed: the root no longer sends probes, and the three latencies came from them (§5.1) |

## 4. The default metrics, one by one

The default data conversion
([defaultDataConversion.ts](../../gui/simlab/src/services/defaultDataConversion.ts),
mirrored by [util/create_exp.py](../../util/create_exp.py); a test checks that the two
lists are equal) defines 16 metrics. "Reads" is the CSV column; the quantity is the
aggregate over the whole simulation. The wizard's default objectives are `latency`
(minimise), `power_mw` (minimise) and `throughput` (maximise).

| Metric | `kind` · reads | What it measures | Caveats |
|---|---|---|---|
| `latency` | `median` · `latency` | Median one-way delay of the reports that reached the root, node to root. **Default objective.** | §5.1. Only delivered reports have a latency; read it with `delivery_ratio` and `node_coverage`. |
| `latency_mean` | `mean` · `latency` | Mean of the same delays. | Follows the tail: under TSCH several times the median (§5.1). |
| `latency_p95` | `quantile` (`q` 0.95) · `latency` | 95th percentile of the same delays. | The tail the median leaves out. |
| `power_mw` | `ratio_sum` · `total_energy_mj` ÷ `energy_time_ms`, × 1000 | Mean power of a node that reported, mW: its energy over the node-time that energy covers. **Default objective.** | §5.2. Does not grow with how many nodes reported or for how long; nodes that never reported are not in it (see `node_coverage`). |
| `energy` | `sum_all` · `total_energy_mj` | Energy of all nodes that reported, over the simulation. | Grows with nodes × time reported (§5.2); a node that never reported contributes nothing. Not comparable across runs with different coverage. |
| `throughput` | `sum_max` · `server_received` | Reports the root received, summed over nodes. The `V` of paper 1. **Default objective.** | §5.3. v1 also counted pongs, 32% of the total on average (§7). |
| `delivery_ratio` | `ratio_sum_max` · `server_received` ÷ `total_sent` | Reports received over reports generated. | §5.3. Counts the reports generated up to each node's last delivered one. |
| `node_coverage` | `node_coverage` · `node` | Fraction of the sensor nodes whose reports reached the root at least once. | §5.3. Needs the master-node's count of deployed nodes; NaN when it is not known. |
| `cpu_energy_mj` | `sum_all` · `cpu_energy_mj` | CPU energy. | In Cooja the CPU is on from boot and never sleeps, so this is a constant times the time the node reported (§5.2). |
| `lpm_energy_mj` | `sum_all` · `lpm_energy_mj` | Low-power-mode energy. | **Always 0**: Cooja never switches the LPM state on (§5.2). |
| `radio_tx_energy_mj` | `sum_all` · `radio_tx_energy_mj` | Energy of the radio transmitting. | v1 recorded about a quarter of it (truncation, §5.2). |
| `radio_rx_energy_mj` | `sum_all` · `radio_rx_energy_mj` | Energy of the radio on and not transmitting. | The bulk of `energy` under CSMA. |
| `total_sent` | `sum_max` · `total_sent` | Reports the nodes generated. | Whether or not they arrived. |
| `bytes_tx` | `sum_max` · `bytes_tx` | Application payload bytes the nodes sent. | v1: always 0 (§5.5). |
| `server_bytes_rx` | `sum_max` · `server_bytes_rx` | Report payload bytes the root received. | v1: always 0. |
| `hops` | `mean` · `hops` | Mean number of links between a node and the root. | Off by one in v1 (§5.4). Weighted by reports, not by node. |

The default conversion used to have six more metrics (`rtt_latency`, `r2n_latency`,
`n2r_latency`, `total_received`, `bytes_rx`, `server_sent`). They read columns that only
the probes produced. Finished experiments keep them in their stored conversion.

The counters use `sum_max`, not v1's `sum_last_minus_first`, which leaves each node's
first report out: under TSCH, where a node may deliver a few dozen reports, that is a
larger share than under CSMA, so it would bias the comparison.

## 5. How each quantity is measured

### 5.1 Latency

`latency` is **the one-way delay of the reports, node to root**, in milliseconds:
the time from the moment the node generates a report to the moment the root's UDP
stack receives it. That is what a data-collection application experiences, and it
includes queueing in the node, MAC access (CSMA backoff, or the wait for the next
TSCH cell), retransmissions and every hop in between.

**How.** The node writes the simulation time, in µs, into the report just before it
hands the packet to the stack (`send_us`). The root reads the same clock when the
report arrives and prints the difference. There is one value per received report, no
probe traffic, and no assumption about the path being symmetric.

**Why the simulation clock.** Both readings are `RTIMER_NOW()`, which Cooja sets to
the global simulation time on every mote before it runs
(`ContikiClock.doActionsBeforeTick`), so it is the same value on every mote and under
both MACs. `clock_time()` is not: the template starts each mote after a random delay of
up to `<motedelay_us>` (1 s) and its clock counts from its own boot. On a CSMA run the
difference between the root's clock and each node's, per report, had a per-node median
between −819 ms and +71 ms, while a delay cannot be negative. That is why the v1
`r2n_latency` and `n2r_latency`, which subtracted two `clock_time()` readings, were not
delays (§7). Using the simulation clock ties the measurement to Cooja:
`metrics-packet.h` refuses to build for another target, because on real hardware two
motes' rtimers cannot be subtracted.

**Checked in Cooja** (two topologies from the database (26 nodes and the root), 180 min each, replayed
under both MACs; the comparison is in §8). In all four runs no value was negative or
missing, and the delay grows with the number of hops. Under CSMA the minimum is 5.8 ms
(one hop) and the median is 27, 55, 82 and 110 ms for 1 to 4 hops, about 28 ms per hop;
the mean of the CSMA run on which the earlier, probing draft of v2 was also tried was
64.4 ms against that draft's RTT/2 of 61.2 ms, so the two estimates agree.

**What it is not.**

- **Only delivered reports have a latency.** A report that never arrives, and a node that
  never joined, are absent, so a MAC that delivers less can look faster. Read it
  together with `delivery_ratio` and `node_coverage` (§5.3, §8).
- **The objective is the median.** Every report is one value, so a node that delivers
  more counts more. Under TSCH the mean follows the tail: on one run the mean is 3.8
  times the median, because a few reports waited seconds, one of them 65 s (§7, §8). The
  default conversion therefore optimises the median (`latency`) and keeps the mean
  (`latency_mean`) and the 95th percentile (`latency_p95`) to show the tail.
- **It is a simulation measurement.** It needs Cooja's global clock.

**v1** had three latencies, all from probes (pings) the root sent to the nodes:
`rtt_latency` (RTT/2 of the last pong, 0 until the first), `r2n_latency` and
`n2r_latency`. The root probed every node in the same instant, so with the default CSMA
limit of two neighbour queues most of each burst was dropped in the MAC layer without a
log line (78% in the instance replayed in §7), and a node that never got a ping
reported 0 in every row and pulled the mean down. Under TSCH the probes needed the
single shared cell of the minimal schedule and mostly did not come back (§7). The
one-way columns were the subtraction of two unsynchronised clocks on CSMA, or, on
TSCH, delays on the shared network time. None of the three could be compared across the
two MACs, and the probes were load that the MACs handled very differently, so the
latency they produced said more about the probe than about the application.

### 5.2 Energy

Each node has Contiki-NG's **Energest** counters: the time spent in four states.
Energy is time times a constant draw.

| State | Counter | What switches it on |
|---|---|---|
| CPU | `ENERGEST_TYPE_CPU` | On from boot. Nothing in the Cooja platform turns it off. |
| Low-power mode | `ENERGEST_TYPE_LPM` | **Never**, on the Cooja platform. Hence `lpm_energy_mj = 0`. |
| Radio transmit | `ENERGEST_TYPE_TRANSMIT` | `cooja-radio.c`, while a frame is being sent. |
| Radio listen | `ENERGEST_TYPE_LISTEN` | `cooja-radio.c`, while the radio is on and not transmitting. Under CSMA that is the whole run. Under TSCH, once the node has joined, only the active cells; a node that has not joined scans with the radio on. |

`E[mJ] = I[mA] · V[V] · t[s]`, with the draws below and V = 3.0 V. They are the
figures of the Tmote Sky platform (MSP430 and CC2420) that v1 already used. The CPU and
LPM values match those published for that platform in the Contiki energy-estimation
work; the radio values were inherited from v1 and should be checked against the
datasheet before a paper cites them.

| State | I (mA) |
|---|---|
| CPU | 1.8 |
| LPM | 0.0545 |
| TX | 17.4 |
| RX (listen) | 19.7 |

Three things differ between versions.

- **Where it is computed.** v1: on the node, per report, then the node's counters are
  zeroed. A report that is lost takes the energy of its interval with it, and the
  fraction lost (3.2% of reports on average in a sample of 70 CSMA simulations, up to
  9.8%) is missing from `energy`. v2: the node sends cumulative times; the root
  subtracts the previous report's, so a lost report is covered by the next one. A
  report that arrives out of order (TSCH does reorder) adds no energy, because a later
  report already covered its interval.
- **Units.** v1 multiplied time by the current and wrote the result as mJ, which
  leaves out the supply voltage: its values are mA·s, a third of the energy at 3 V.
  v2 includes V.
- **Truncation.** v1 stored each interval's energy in an unsigned integer, dropping
  the fraction every report. The transmitter is on for only a few milliseconds per
  interval, so in a measured run (CSMA, 26 nodes, 180 min) 90% of the intervals had a
  true TX energy below 1 mJ (median 0.3, in v1's units) and were recorded as 0; the TX
  total came out at 26% of the real one (2,778 recorded against 10,627). RX and CPU
  lose under 2%. v2 keeps 3 decimals, and the energy model can be changed (`ENERGY_*`
  in `root.c`) without re-running anything, since the nodes send times, not energy.

**What `energy` is, and is not.** Under CSMA the radio never sleeps, so the energy of
a run is mostly the radio listening: on 28,216 CSMA simulations, 91.5% of `energy`
was RX, 8.3% CPU, 0.20% TX. `nodes × duration` alone explains 82.5% of its variance.
That is physics of an always-on MAC, not an error, but it means `energy` under CSMA
says how many nodes ran for how long far more than how well the network routed.
TSCH, which turns the radio off between cells, is the MAC in which energy depends on
the design.

Nodes that never joined send no report and add no energy. A network that formed
partially therefore looks cheaper than one that formed completely, which is how paper 1
found that TSCH's low-energy region was a formation failure.

**Power, `power_mw`.** For each report the root also prints `energy_time_ms`: the
node's CPU + LPM time over the same interval as the energy columns. The MCU is always in
one of the two, so that is the node's elapsed time over the interval (the Cooja platform
has no deep-sleep state; only the radio driver touches Energest, and the CPU counts from
boot). Summed over rows it is the node-time the energy covers, so

`power_mw = 1000 × Σ total_energy_mj ÷ Σ energy_time_ms`

is the mean power of a node that reported, over all the time those nodes' reports
cover, from boot to each node's last delivered report. A silent stretch is inside it,
because the next report's interval spans it. It is a ratio of totals, not a mean of
per-report powers, which would give a 10-second interval and a one-hour gap the same
weight. Unlike `energy`, it does not grow with the number of nodes heard from or with
how long they kept reporting, so runs with different coverage can be compared on it;
nodes that never reported are not in it, which is what `node_coverage` reports. Under
CSMA it is nearly constant, because the radio never sleeps.

### 5.3 Throughput

`throughput` is the number of reports the root received, summed over the nodes
(`sum_max` of each node's cumulative count). It counts **messages**, not bytes and not a
rate. It measures how much of what the nodes generated got through, and which nodes got
in at all; it is not the capacity of the network.

Two metrics say how much of the network the others describe, since every row is a
report that arrived:

- **`delivery_ratio`**: reports received over reports generated, both summed over
  nodes. A node's generated count is the `seq` of its newest delivered report, so the
  reports a node generated after its last delivered one are not known and not counted;
  a node that goes silent at the end of a run looks better than it is. A node generates
  a report only while it has a route to the root (§5.7), so time spent outside the
  network lowers `throughput`, not this ratio.
- **`node_coverage`**: the fraction of the sensor nodes that delivered at least one
  report. The denominator is the number of motes other than the root that wrote
  anything to the testlog; every v2 node writes from boot, joined or not.

`latency`, `power_mw` and `hops` describe only the reports and nodes these two count.

v1 counted every datagram from a node, pongs included. On a sample of 60 CSMA
simulations pongs were 32% of `throughput` on average (10th to 90th percentile 20%
to 44%, up to 47%), and their number depends on how many pings get through, which
depends on the topology: the metric partly measured the probe, not the application.
v2 has no probes, so every datagram the root counts is a report.

### 5.4 Hops

The root reads the IPv6 hop limit of each report. The node sends it as `UIP_TTL`
(64) and each forwarder takes one off. A direct child therefore arrives with 64.

- v1: `64 − ttl`, the number of **relays**: 0 for a direct child.
- v2: `64 − ttl + 1`, the number of **links**: 1 for a direct child, which is the
  usual meaning of hop count and agrees with the BFS depth of the DODAG.

On the same simulation the mean was 1.27 under v1 and 2.28 under v2. The default
conversion averages over rows, so a node that reports more often weighs more.
`pylib/rpl_dodag.py` uses `hops` to estimate each node's depth (`approx_depth`);
with v2 that estimate is on the scale of the exact tree built from the `[RPL]`
events: they were equal for 22 of the 24 nodes compared in the v2 run above, and
differed by 2 for the other two (the tree uses each node's last parent, the TTL
reflects the path each report took; not investigated further).

### 5.5 Counters and bytes

Cumulative counters are summed per node: `sum_max` in the default conversion,
`sum_last_minus_first` in v1's. Either way the counter must accumulate.

v1 stored in `bytes_tx` the size of the report (constant) and in `bytes_rx` the
size of the last datagram received, so `bytes_tx` and `server_bytes_rx` have a growth
of exactly 0 and `bytes_rx` is noise. v2 accumulates UDP payload bytes: for a node,
the reports it sent (`bytes_tx`); for the root, the report payload received from each
node (`server_bytes_rx`). The two differ by the reports that were lost. Payload only:
no UDP, IPv6, 6LoWPAN or 802.15.4 headers, so these are not on-air bytes.

### 5.6 Link quality

`rssi` and `lqi` are read from the radio when the node generates a report and describe
the last frame the radio saw, of any type and from any neighbour, not a specific link.
(v1 read them when a ping arrived.) Under UDGM with success ratio 1.0 they are nearly
constant. They are not default metrics.

### 5.7 Reports and the DODAG

- A node reports every 10 s plus a random jitter in [0, 1 s). v1 drew the jitter once,
  so each node kept its own period for the whole run and two nodes with close draws
  collided in every period. v2 draws it again for every report.
- A node that is not reachable prints `Not reachable yet` and sends nothing.
- On every change of preferred parent a node prints
  `[RPL] t=<ms> node=<link-local> parent=<link-local>`. `pylib/rpl_dodag.py` rebuilds
  the tree, the join times and the convergence time from these lines.
- The v1 node printed its address every period; v2 prints it once.
- The root keeps counters for 200 nodes (v1: 100, and the 101st dereferenced a null
  pointer). Beyond that it prints `No space for mote counter!` once and ignores the
  extra nodes.
- The report starts with a type byte, so a second kind of packet can be added without
  ambiguity (v1 told its two packets apart by length). Every field has a fixed width; v1 used `unsigned int`/`unsigned long`, whose size
  depends on the platform, so its "packed" layout was not portable.

## 6. Configuration that shapes the measurement

| Setting | Value | Effect |
|---|---|---|
| Report period | 10 s + jitter in [0, 1 s) | Number of rows and the resolution of every counter. It is also the whole traffic of the application. |
| Supply voltage and draws | 3.0 V; 1.8 / 0.0545 / 17.4 / 19.7 mA (`ENERGY_*` in `root.c`) | Absolute energy. Orderings do not depend on them as long as every term is scaled the same way. |
| CSMA neighbour queues | Contiki-NG default, 2 (`CSMA_CONF_MAX_NEIGHBOR_QUEUES`) | A packet for a third neighbour while two queues are in use is dropped, silently (MAC log level is `NONE`). v1's burst of pings reached it constantly; v2 sends only the reports, so it is rarely reached (§7). |
| TSCH schedule | v2: one shared cell every 3 slots of 10 ms (`TSCH_SCHEDULE_CONF_DEFAULT_LENGTH 3` in `project-conf.h`). v1: Contiki-NG's default, every 7 | With 7, at this report rate and 26 nodes, the cell congests and RPL drops routes (§8.1). With 3, TSCH delivers what CSMA delivers (§8). |
| Queue buffers | 64 on the Cooja platform (`QUEUEBUF_CONF_NUM`) | A TSCH node can queue up to 64 packets for one neighbour, so under congestion a report can wait tens of seconds before it is sent. |
| Neighbour and route tables | v2: 64 entries (`NBR_TABLE_CONF_MAX_NEIGHBORS`, `NETSTACK_MAX_ROUTE_ENTRIES` in `project-conf.h`). v1: the Cooja platform's 300 | None up to 63 motes, root included: the output is bit for bit that of the 300-entry build (five instances, both MACs). The tables are smaller only to make Cooja faster. Past 63 motes a full table evicts neighbours (`firmware/README.md`, "Simulation speed"). |
| Radio | UDGM, ranges from the problem, success 1.0 | No random channel loss. |
| Contiki-NG | The one in the `simulation-cooja:v1.1` image (`a044c3203`, November 2024) | The semantics of the seed and of the stack depend on it. Record the image with the results. |

## 7. Known limitations

**Fixed in v2** (measured on this installation's database, 2026-11-13, on firmware v1):

| Finding | Evidence |
|---|---|
| Pings dropped by CSMA; `latency` biased by zeros | On 60 CSMA simulations, 33% of rows (0.2% to 85% per simulation) had `rtt_latency = 0` and 25% of the nodes never received a ping. The correlation between the share of zero rows and `latency` was −0.90. With at most two children of the root, every node received pings; with more, 67% did (80 simulations). |
| Same instance, replayed in Cooja | 26 nodes and the root, seed 987654, v1: 21,404 `could not allocate neighbor, dropping packet`, 6,202 of 27,868 pings received (22%), 11 of 26 nodes answered, `latency` 62.5 ms. Raising the queue limit to 16 with the bursts kept: all 26 answered, `latency` 349.9 ms, and the reports that reached the root fell from 22,977 to 14,183 (the burst saturates the channel). An earlier draft of v2 that still probed, one node at a time: 26 of 26 answered, 0.44% of the rows without a sample, `latency` (RTT/2) 61.2 ms, 22,892 reports logged, 9 queue drops. The v2 that ships sends no probes: 22,741 reports, `latency` 64.4 ms (one-way). |
| The probes starve TSCH | One TSCH instance (topology B of §8), with the earlier draft of v2 in which the root pinged the nodes: **617 reports from 21 nodes**; with the v2 that ships, no probes, **6,416 reports from 20 nodes**. The minimal TSCH schedule has one shared cell per 7-slot slotframe for all traffic (§8). |
| `throughput` includes pongs | 32% on average (§5.3). |
| `bytes_tx`, `server_bytes_rx` always 0; `bytes_rx` noise | §5.5. |
| TX energy truncated | 26% of the real value in the replayed run (§5.2). |
| Energy in mA·s labelled mJ | §5.2. |
| `r2n_latency`, `n2r_latency` garbage on CSMA | §5.1; 65% and 95% of the CSMA simulations have an aggregate above 10¹⁵. The mote clocks start at their own boot. |
| Three latencies, none comparable across MACs | §5.1. |
| `hops` off by one | §5.4. |
| `server_sent` a copy of `throughput` | The default conversion read the wrong column (§9). |
| Energy of lost reports vanishes | §5.2. |

**Not fixed; they belong to the model or the experiment, not to a bug.**

- **Energy under CSMA is mostly idle listening** (§5.2). The `energy` objective is
  informative for TSCH and for comparing node counts, much less for routing quality.
- **Cooja has no sleep model.** The CPU is always on and `lpm_energy_mj` is 0. CPU
  energy is a constant per node and second.
- **No channel loss** (UDGM, success 1.0): variance across seeds is internal.
- **Nodes that never joined are invisible** to `energy`, `power_mw`, `latency` and
  `throughput`. `node_coverage` counts them (§5.3), and the DODAG analysis
  (`expected_nodes`, `n_nodes_joined`) shows when the others joined. Keep coverage next
  to every comparison. In the TSCH runs of paper 1 a median 31% of the nodes never
  joined.
- **TSCH nodes go silent for long stretches under the default minimal schedule.** On
  topology B of §8 the gap between consecutive reports of a node had a median of 10.5 s,
  a 99th percentile of 212 s and a maximum of 3,580 s (topology A: 2,796 s and 7,835 s).
  The cause is congestion of the shared cell, which makes RPL drop routes (§8.1).
- **`latency` is conditional on delivery** (§5.1).
- **`rssi`/`lqi` are not per link** (§5.6).
- Residual CSMA queue drops: 9 in the 180-minute run of the earlier draft (against 21,404).

## 8. Comparing CSMA and TSCH

With v2 the two MACs run the same application from the same sources, and `latency`
is measured the same way under both, so the definitions of the metrics are the same.
What differs is how much of the traffic each MAC lets reach the root, because every
metric is computed from the reports that arrive.

Measured on two topologies from the database (26 nodes and the root; A: seed 987654,
B: seed 123456), 180 min each, replayed with the same sources, so that the MAC and its
schedule are the only difference (2026-11-15; one run per cell; metrics through the
real converter and the default conversion). Three TSCH schedules: Contiki-NG's default
minimal schedule (one shared cell every 7 slots, what v1 built), the same with a shared
cell every 3 slots (`TSCH_SCHEDULE_CONF_DEFAULT_LENGTH 3`, the value of Contiki-NG's own
6TiSCH example, and what `rpl-udp-tsch-v2` builds), and Orchestra with its default rules
(`MODULES += $(CONTIKI_NG_SERVICES_DIR)/orchestra`).

Topology A:

| | CSMA | TSCH, minimal (7) | TSCH, shared cell every 3 slots | TSCH, Orchestra |
|---|---|---|---|---|
| Nodes that reported (of 26) | 26 | 26 | 26 | 25 |
| Reports delivered per reporting node | 875 | 59 | 888 | 286 |
| `delivery_ratio` | 97.7% | 73.7% | 96.8% | 75.2% |
| `latency` median / mean / p95 (ms) | 60 / 64 / 127 | 676 / 2,931 / 13,681 | 87 / 188 / 694 | 233 / 500 / 2,102 |
| `power_mw` | 64.5 | 26.4 | 12.2 | 9.9 |
| Radio on (TX + RX time over time) | 100% | 36% | 11% | 8% |
| Node-time not inside a gap of more than 60 s | 86% | 21% | 88% | 58% |

Topology B:

| | CSMA | TSCH, minimal (7) | TSCH, shared cell every 3 slots | TSCH, Orchestra |
|---|---|---|---|---|
| Nodes that reported (of 26) | 26 | 20 | 26 | 26 |
| Reports delivered per reporting node | 888 | 321 | 886 | 389 |
| `delivery_ratio` | 98.0% | 96.3% | 97.2% | 81.5% |
| `latency` median / mean / p95 (ms) | 68 / 70 / 129 | 186 / 699 / 3,170 | 90 / 184 / 663 | 342 / 479 / 1,495 |
| `power_mw` | 64.5 | 16.8 | 12.4 | 11.7 |
| Radio on (TX + RX time over time) | 100% | 19% | 12% | 11% |
| Node-time not inside a gap of more than 60 s | 88% | 64% | 88% | 76% |

Across seeds. The same two topologies with two more Cooja seeds each (336157 and
667370), so six runs per column; ranges over the six (Orchestra was not repeated):

| | CSMA | TSCH, minimal (7) | TSCH, shared cell every 3 slots |
|---|---|---|---|
| Nodes that reported (of 26) | 26 | 20 – 26 | 26 |
| Reports delivered per reporting node | 875 – 888 | 46 – 321 | 886 – 894 |
| `delivery_ratio` | 97.6 – 98.0% | 73.7 – 96.3% | 96.8 – 97.2% |
| `latency` median (ms) | 60 – 68 | 186 – 692 | 86 – 92 |
| `latency` p95 (ms) | 127 – 130 | 3,170 – 28,035 | 608 – 772 |
| `power_mw` | 64.5 | 16.8 – 28.1 | 12.0 – 12.4 |

What the tables say:

- **The minimal schedule congests at this load** (§8.1), and that, not TSCH, is what
  made the TSCH numbers so different from CSMA's: few reports, long silences, a latency
  tail of tens of seconds, and results that change widely from one seed to the next. It also costs energy: nodes that lose the network scan with
  the radio on, so the minimal schedule spends more power than the one with more
  cells.
- **With a shared cell every 3 slots TSCH delivers what CSMA delivers**, in all six
  runs, and varies across seeds as little as CSMA does: every node, as many reports, about the same delivery ratio and the same
  share of node-time without long gaps. The latency median is 1.3 to 1.5 times CSMA's and the
  95th percentile about 5 times; the power is a fifth. This is why `rpl-udp-tsch-v2`
  uses it: a comparison of the two MACs is then made with both carrying the same traffic.
- **Orchestra with its default rules** spends the least power but delivers less than
  CSMA. Its rules were not tuned here; with reports converging on the root, the root's
  receive cell (one every 17 slots) is a likely limit, not verified.
- **CSMA's power is the same in every run**, 64.5 mW, because its radio never sleeps.

What it means for each metric:

- **`latency`.** The same instrument under every MAC, but each is measured on the
  reports that got through. Compare it only next to `delivery_ratio` and
  `node_coverage`, and with the median as the objective: under the congested minimal
  schedule the mean is 3.8 to 4.3 times the median.
- **`throughput`.** The number of reports delivered. It measures reliability and how many
  nodes got into the network, not the capacity of the MAC.
- **Energy.** Compare `power_mw`, not `energy`. The sum grows with how many nodes
  reported and for how long, and a node that never reported adds nothing (six nodes on
  topology B under the minimal schedule). v2's cumulative counters recover the energy of
  a silent stretch when the node reports again, so `power_mw` is not biased by the gaps.
- **`hops`.** The same definition under every MAC.

What is being compared, in every column: Contiki-NG defaults otherwise, the report rate
of §6, and UDGM with success ratio 1.0, so there is no channel noise for TSCH's channel
hopping to defend against. Six runs per configuration (two topologies of 26 nodes,
three seeds each) bound the variation at this size and load; other network sizes,
densities and report rates were not measured.

### 8.1 Why TSCH nodes go silent

Investigated on 2026-11-15 with a copy of the TSCH v2 firmware, kept out of the
repository, that logs the MAC and RPL warnings and, every second, whether the node is
associated, its time source, whether it has joined the DODAG and has a route to the
root, and the length of its TSCH queue. Topologies A and B of §8, 180 min. The extra
output changes the trajectory of a run (instance B delivered 6,416 reports from 20 nodes
without it and 2,421 from 25 with it), not the mechanism, which was the same in every
instrumented run. Numbers below are topology B; A was the same or worse.

1. **In a gap the node is in the network but has no route.** Of the time inside gaps
   longer than 60 s, the node was associated with TSCH and not reachable 70% of the
   time, not associated 27%, and associated and reachable 3%. A node generates a report
   only while it has a route (§5.7), so without one it goes silent.
2. **Almost every gap contains an RPL local repair by the node itself**: 169 of 176
   (A: 201 of 214). Of the repairs, 974 were `DAO max rtx` and 277 `TSCH leaving`.
   RPL-lite waits for the root's DAO-ACK: five transmissions, each with a timeout drawn
   between 2.5 and 7.5 s. Without an ACK the node repairs locally: it drops its parent,
   and its children lose their route too (2,272 `no parent` warnings).
3. **The DAO gets no ACK because the queues are full.** 6,690 packets were dropped
   with `can't send packet ... queue 63/64` (the Cooja platform sets 64 queue buffers,
   and a single neighbour's queue can take them all). All of them were on the way up,
   toward relays, about half toward the root's direct children, and none toward the
   root. Behind dozens of packets waiting for one shared cell every 70 ms, contended by
   every neighbour, a DAO and its ACK take longer than RPL waits. The same queues
   produce the latency tail of §8.
4. **Keepalives wait in the same queues.** A node without synchronisation for 120 s
   (`TSCH_DESYNC_THRESHOLD`) leaves the network and scans with the radio on: 349 leaves,
   every node at least once; on average a node was associated 69% of the time.
5. **It starts when the network forms and feeds itself.** Every repair adds DIS, DIO and
   DAO messages to the same cell. On B the queues overflowed for the first 150 minutes
   and then nearly stopped (39 drops in the last 30); on A for the whole run. With reports every 60 s instead of 10 s,
   they overflowed during the first hour and not after 90 minutes. Whether and when a
   run gets out depends on its trajectory, which is why replays of the same instance
   differed so much.

**Confirmation.** The same instrumented instance with one shared cell every 3 slots
instead of 7 (`TSCH_SCHEDULE_CONF_DEFAULT_LENGTH 3`, the value Contiki-NG's own 6TiSCH
example uses) dropped no packet for a full queue, had all 26 nodes reporting, 890
reports per node (888 under CSMA), a median latency of 90 ms, and its nodes associated
98% of the time. Without the instrumentation, on both topologies, the same setting
matched CSMA's delivery (§8, tables). Lowering the report rate alone did not prevent the
congestion while the network formed.

## 9. Using results across versions

- **Do not mix v1 and v2 values of the same metric in one comparison.** The names are
  the same and the quantities are not (units, definitions, and for `latency` and
  `throughput` the probe traffic that v1 put on the network). The `fw_version` column
  marks v2 rows; v1 rows have none.
- **v1 columns that are safe to use**: `total_sent`, `total_received`, `server_sent`
  (the CSV column, not the stored metric), the energy columns for *relative*
  comparisons between runs of the same firmware, and `hops` if 1 is added.
- **v1 columns to avoid**: `bytes_tx`, `bytes_rx`, `server_bytes_rx`, `lpm_energy_mj`,
  `r2n_latency` and `n2r_latency`, and `radio_tx_energy_mj` as an absolute number.
  `rtt_latency` (the stored `latency` of every finished experiment) and `throughput` are
  usable but carry the biases of §7, which depend on the topology, so they can reorder
  solutions.
- **The stored `server_sent` of finished experiments is a copy of `throughput`.** The
  CSV still has the right column, so it can be recomputed:
  `sum_last_minus_first` over `server_sent`.
- **Moving an experiment to v2** means launching it again with the `-v2` source
  repositories; nothing about a finished experiment is rewritten. Upload them from
  `firmware/` with `python upload-firmware.py rpl-udp-csma-v2 rpl-udp-tsch-v2`. Compare a v1
  result with a v2 one only after replaying a sample of the v1 individuals under v2: the
  objective values change, and so may the ordering the optimiser saw.
- **Running v1 with the wizard.** The default conversion reads v2. Under v1, in step 5:
  point `latency`, `latency_mean` and `latency_p95` at `rtt_latency`; and in step 4
  replace the `power_mw` objective with `energy`, since v1 prints no `energy_time_ms`.
  Otherwise those metrics are NaN. `delivery_ratio` means nothing under v1: its
  `server_received` counts pongs, so the ratio can exceed 1.

## 10. Checking a metric yourself

Per simulation, the database keeps the raw testlog (`log_cooja_id`), the CSV
(`csv_log_id`) and the evaluated metrics (`network_metrics`). Both files are
zstd-compressed in GridFS. To follow a number: open the CSV, apply the `kind` of
the metric to its `column` (§2.1), and compare with `network_metrics`. For the
chain from the log to the metrics, `pylib/cooja_builder/csv_converter.py` and
`pylib/statistics.py` are the only code involved.

`pylib/tests/test_firmware_v2.py` runs real v2 lines through that chain, with the
deployed nodes counted in the log as the master-node does, and checks that every default
metric is finite, that `latency` is the median of the per-report delays, that
throughput counts every report and only reports, that the byte counters accumulate,
that `power_mw` is energy over the time it covers, that `node_coverage` counts the nodes
heard from over the nodes deployed, and that there is exactly one latency column,
measured on the simulation clock, with no probes. It also checks that the GUI and
`util/create_exp.py` define the same default conversion and that the root prints every
column it reads.
