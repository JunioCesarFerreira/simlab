"""Guards for the firmware shipped in firmware/.

Two things here are easy to break by accident and expensive to notice late:

* v1 (rpl-udp-csma, rpl-udp-tsch) is what every finished experiment ran. Their
  snapshots hold these exact digests and three papers cite the results, so an
  edit in place would make the repository stop matching what the numbers came
  from. Changes go in a new directory.
* v2 makes the CSMA and TSCH variants differ in the MAC layer and nothing else,
  and writes the JSON columns the default data conversion reads. A column that
  goes missing does not fail a run: it becomes a NaN metric, quietly.

Metric definitions: docs/markdown/FIRMWARE_METRICS.md
"""
import ast
import hashlib
import logging
import math
import re
from pathlib import Path

import pandas as pd
import pytest

from pylib import statistics
from pylib.cooja_builder.csv_converter import cooja_log_to_csv
from pylib.rpl_dodag import count_deployed_nodes

REPO = Path(__file__).resolve().parents[2]
FIRMWARE = REPO / "firmware"
CSMA_V2 = FIRMWARE / "rpl-udp-csma-v2"
TSCH_V2 = FIRMWARE / "rpl-udp-tsch-v2"

# sha256 of every v1 file, as captured in the firmware snapshots of the
# experiments run so far.
V1_DIGESTS = {
    "rpl-udp-csma/Makefile": "629377f94707703167b02d7f103d0421d2b352e401d9e68cfb890012bd178355",
    "rpl-udp-csma/metrics-packet.h": "8b9f335361e964941e1f0d05e08c5697edce010be0dd40e12aa25cc489608a08",
    "rpl-udp-csma/node.c": "dce041d92ff16e3e56f4bab7f99f5f2f47508def3cb61d6d9a6d5c6c239efc80",
    "rpl-udp-csma/project-conf.h": "ae7994f45d2bcb2715bf62dd98f99443fc2e1fec99acd729878aabf43374d4e1",
    "rpl-udp-csma/root.c": "f4dc0175f82ec9d1bc636a51bb3f8c2cef9416475f2f6ce4e914190d966c57f0",
    "rpl-udp-tsch/Makefile": "92a97925b543e42eaf3cc6a192cdd73b6a68712df5a5a289763aa4062bccfa43",
    "rpl-udp-tsch/metrics-packet.h": "8b9f335361e964941e1f0d05e08c5697edce010be0dd40e12aa25cc489608a08",
    "rpl-udp-tsch/node.c": "1cee964153e67729ab37000ccc6d4716fd3628af6a24ae9b652a5f37e3075752",
    "rpl-udp-tsch/project-conf.h": "ae7994f45d2bcb2715bf62dd98f99443fc2e1fec99acd729878aabf43374d4e1",
    "rpl-udp-tsch/root.c": "efc29f9c65caa320a64ac56b83b90698e71c3006ae4b202c986fb024d645b77f",
}

# Metrics whose name is deliberately not the name of the column they read.
DELIBERATE_ALIASES = {
    "energy": "total_energy_mj",
    "power_mw": "total_energy_mj",
    "throughput": "server_received",
    "delivery_ratio": "server_received",
    "latency_mean": "latency",
    "latency_p95": "latency",
    "node_coverage": "node",
}


def _default_conversion():
    """The default data conversion, as util/create_exp.py defines it."""
    tree = ast.parse((REPO / "util" / "create_exp.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "data_conversion_config":
                    return ast.literal_eval(value)["metrics"]
    raise AssertionError("could not find data_conversion_config in util/create_exp.py")


def _gui_default_conversion():
    """The same list in gui/simlab/src/services/defaultDataConversion.ts, one metric per line."""
    text = (REPO / "gui" / "simlab" / "src" / "services" / "defaultDataConversion.ts").read_text()
    items = []
    for line in re.findall(r"^\s*\{ (name: .*) \},\s*$", text, re.M):
        pairs = re.findall(r'(\w+): ("[^"]*"|[\d.]+)', line)
        items.append({k: (v.strip('"') if v.startswith('"') else float(v)) for k, v in pairs})
    assert items, "could not read the metrics in defaultDataConversion.ts"
    return items


def test_the_gui_and_create_exp_have_the_same_default_conversion():
    def norm(items):
        return sorted((sorted((k, float(v) if isinstance(v, (int, float)) else v) for k, v in i.items())
                       for i in items))
    assert norm(_gui_default_conversion()) == norm(_default_conversion())


def test_v1_firmware_is_untouched():
    for rel, expected in V1_DIGESTS.items():
        digest = hashlib.sha256((FIRMWARE / rel).read_bytes()).hexdigest()
        assert digest == expected, (
            f"{rel} changed. Finished experiments and the papers refer to this exact "
            f"firmware (see docs/markdown/FIRMWARE_TRACEABILITY.md); put the change in a "
            f"new directory instead of editing v1."
        )


@pytest.mark.parametrize("name", ["node.c", "root.c", "metrics-packet.h", "project-conf.h"])
def test_v2_variants_share_their_sources(name):
    assert (CSMA_V2 / name).read_bytes() == (TSCH_V2 / name).read_bytes(), (
        f"{name} differs between rpl-udp-csma-v2 and rpl-udp-tsch-v2. The two must run the "
        f"same application so that the MAC layer is the only difference between them."
    )


def test_v2_variants_differ_only_in_the_mac():
    def mac(makefile: Path):
        lines = [l.strip() for l in makefile.read_text().splitlines() if not l.strip().startswith("#")]
        return [l for l in lines if l.startswith("MAKE_MAC")]

    assert mac(CSMA_V2 / "Makefile") == []
    assert mac(TSCH_V2 / "Makefile") == ["MAKE_MAC = MAKE_MAC_TSCH"]


def test_v2_tsch_has_a_shared_cell_every_3_slots():
    """Contiki-NG's default, one shared cell every 7 slots, congests with 26 nodes reporting every
    10 s: RPL drops routes and nodes go silent for minutes (FIRMWARE_METRICS.md, section 8.1)."""
    conf = (TSCH_V2 / "project-conf.h").read_text()
    assert re.search(r"^#define TSCH_SCHEDULE_CONF_DEFAULT_LENGTH 3\s*$", conf, re.M)


def test_v2_root_prints_every_column_the_default_conversion_reads():
    keys = set(re.findall(r'\\"(\w+)\\":', (CSMA_V2 / "root.c").read_text()))
    needed = {"node", "root_time_now"}
    for item in _default_conversion():
        needed |= {item["column"]} | ({item["denominator"]} if "denominator" in item else set())
    assert needed <= keys, f"root.c does not print: {sorted(needed - keys)}"


def test_default_conversion_names_match_the_column_they_read():
    """`server_sent` once read `server_received` and was a copy of `throughput`."""
    for item in _default_conversion():
        expected = DELIBERATE_ALIASES.get(item["name"], item["name"])
        assert item["column"] == expected, (
            f"metric '{item['name']}' reads column '{item['column']}', expected '{expected}'"
        )


# Lines from a real v2 Cooja run (CSMA, 26 nodes, 180 min), cut down: the first line each of
# three motes wrote, and the root's lines for two of them, each one's first two reports and its
# 101st and 102nd. The third mote's reports are left out, as if they had never reached the root.
_V2_LOG = r"""
[t_us=851000] [Mote:7] [INFO: Main      ] Starting Contiki-NG-develop/v5.0
[t_us=161000] [Mote:12] [INFO: Main      ] Starting Contiki-NG-develop/v5.0
[t_us=30000] [Mote:9] [INFO: Main      ] - Routing: RPL Lite
[t_us=20918008] [Mote:1] {"node":"fd00::207:7:7:7", "fw_version":2, "cpu_energy_mj":108.259, "lpm_energy_mj":0.000, "radio_tx_energy_mj":1.935, "radio_rx_energy_mj":1182.646, "total_energy_mj":1292.840, "energy_time_ms":20048.000, "tx_time_ms":37.072, "rx_time_ms":20010.928, "total_sent":1, "bytes_tx":52, "latency":19.008, "hops":1, "lqi":100, "rssi":-58, "server_received":1, "server_bytes_rx":52, "reports_lost":0, "root_time_now":20010}
[t_us=21534920] [Mote:1] {"node":"fd00::20c:c:c:c", "fw_version":2, "cpu_energy_mj":115.193, "lpm_energy_mj":0.000, "radio_tx_energy_mj":1.992, "radio_rx_energy_mj":1258.465, "total_energy_mj":1375.651, "energy_time_ms":21332.000, "tx_time_ms":38.168, "rx_time_ms":21293.832, "total_sent":1, "bytes_tx":52, "latency":41.920, "hops":1, "lqi":100, "rssi":-93, "server_received":1, "server_bytes_rx":52, "reports_lost":0, "root_time_now":20626}
[t_us=31521008] [Mote:1] {"node":"fd00::207:7:7:7", "fw_version":2, "cpu_energy_mj":57.105, "lpm_energy_mj":0.000, "radio_tx_energy_mj":2.694, "radio_rx_energy_mj":621.933, "total_energy_mj":681.731, "energy_time_ms":10575.000, "tx_time_ms":88.672, "rx_time_ms":30534.328, "total_sent":2, "bytes_tx":104, "latency":47.008, "hops":1, "lqi":100, "rssi":-45, "server_received":2, "server_bytes_rx":104, "reports_lost":0, "root_time_now":30613}
[t_us=32030008] [Mote:1] {"node":"fd00::20c:c:c:c", "fw_version":2, "cpu_energy_mj":56.495, "lpm_energy_mj":0.000, "radio_tx_energy_mj":2.869, "radio_rx_energy_mj":615.056, "total_energy_mj":674.420, "energy_time_ms":10462.000, "tx_time_ms":93.128, "rx_time_ms":31700.872, "total_sent":2, "bytes_tx":104, "latency":75.008, "hops":1, "lqi":100, "rssi":-93, "server_received":2, "server_bytes_rx":104, "reports_lost":0, "root_time_now":31122}
[t_us=1069309008] [Mote:1] {"node":"fd00::207:7:7:7", "fw_version":2, "cpu_energy_mj":58.136, "lpm_energy_mj":0.000, "radio_tx_energy_mj":0.642, "radio_rx_energy_mj":635.543, "total_energy_mj":694.322, "energy_time_ms":10766.000, "tx_time_ms":1767.288, "rx_time_ms":1066671.712, "total_sent":101, "bytes_tx":5252, "latency":19.008, "hops":1, "lqi":100, "rssi":-45, "server_received":101, "server_bytes_rx":5252, "reports_lost":0, "root_time_now":1068401}
[t_us=1070041008] [Mote:1] {"node":"fd00::20c:c:c:c", "fw_version":2, "cpu_energy_mj":56.624, "lpm_energy_mj":0.000, "radio_tx_energy_mj":2.163, "radio_rx_energy_mj":617.274, "total_energy_mj":676.061, "energy_time_ms":10486.000, "tx_time_ms":4679.072, "rx_time_ms":1065162.928, "total_sent":101, "bytes_tx":5252, "latency":38.008, "hops":1, "lqi":100, "rssi":-93, "server_received":101, "server_bytes_rx":5252, "reports_lost":0, "root_time_now":1069133}
[t_us=1079816008] [Mote:1] {"node":"fd00::207:7:7:7", "fw_version":2, "cpu_energy_mj":56.597, "lpm_energy_mj":0.000, "radio_tx_energy_mj":0.848, "radio_rx_energy_mj":618.467, "total_energy_mj":675.912, "energy_time_ms":10481.000, "tx_time_ms":1783.536, "rx_time_ms":1077136.464, "total_sent":102, "bytes_tx":5304, "latency":45.008, "hops":1, "lqi":100, "rssi":-45, "server_received":102, "server_bytes_rx":5304, "reports_lost":0, "root_time_now":1078908}
[t_us=1080425008] [Mote:1] {"node":"fd00::20c:c:c:c", "fw_version":2, "cpu_energy_mj":56.144, "lpm_energy_mj":0.000, "radio_tx_energy_mj":2.426, "radio_rx_energy_mj":611.716, "total_energy_mj":670.286, "energy_time_ms":10397.000, "tx_time_ms":4725.544, "rx_time_ms":1075513.456, "total_sent":102, "bytes_tx":5304, "latency":25.008, "hops":1, "lqi":100, "rssi":-93, "server_received":102, "server_bytes_rx":5304, "reports_lost":0, "root_time_now":1079517}
"""


@pytest.fixture
def v2_metrics(tmp_path):
    """The master-node's path: testlog -> CSV -> default conversion, with the deployed nodes counted in the log."""
    log = tmp_path / "COOJA.testlog"
    log.write_text(_V2_LOG.lstrip("\n"))
    csv = tmp_path / "sim_result.csv"
    cooja_log_to_csv(log, csv)
    df = pd.read_csv(csv)
    cfg = {"node_col": "node", "time_col": "root_time_now", "metrics": _default_conversion()}
    return statistics.evaluate_config(
        df, cfg, logging.getLogger("test"), deployed_nodes=count_deployed_nodes(log))


def test_v2_log_gives_a_finite_value_for_every_default_metric(v2_metrics):
    bad = {k: v for k, v in v2_metrics.items() if not math.isfinite(v)}
    assert not bad, f"non-finite metrics: {bad}"


def test_latency_is_the_median_one_way_delay_of_the_reports(v2_metrics):
    # one value per report, in ms: 19.008, 41.92, 47.008, 75.008, 19.008, 38.008, 45.008, 25.008
    assert v2_metrics["latency"] == pytest.approx((38.008 + 41.92) / 2)
    assert v2_metrics["latency_mean"] == pytest.approx(38.747)


def test_throughput_counts_every_report_and_nothing_else(v2_metrics):
    # each node's newest count, summed: 102 reports from each of the two nodes
    assert v2_metrics["throughput"] == 204
    assert v2_metrics["total_sent"] == 204
    assert v2_metrics["delivery_ratio"] == 1.0


def test_byte_counters_are_cumulative(v2_metrics):
    # 102 reports of 52 bytes from each node. v1 wrote the size of the last packet instead.
    assert v2_metrics["bytes_tx"] == 2 * 5304
    assert v2_metrics["server_bytes_rx"] == 2 * 5304


def test_power_is_energy_over_the_time_it_covers(v2_metrics):
    # 6,741.223 mJ over 104,547 ms of node time. Under CSMA the radio never sleeps:
    # (19.7 mA radio listening + 1.8 mA CPU) x 3 V = 64.5 mW, plus a little transmitting.
    assert v2_metrics["power_mw"] == pytest.approx(1000 * 6741.223 / 104547)


def test_coverage_counts_the_nodes_heard_from_over_the_nodes_deployed(v2_metrics):
    assert v2_metrics["node_coverage"] == pytest.approx(2 / 3)


# --- one latency ----------------------------------------------------------------------------------
#
# v1 had three (rtt_latency, r2n_latency, n2r_latency), all from probes the root sent to the nodes.
# On CSMA the motes' clocks start at their own boot, up to <motedelay_us> apart, so a subtraction
# between two of them is not a delay; and a probe is traffic the two MACs carry very differently.
# The one that is left is the one-way delay of the reports, on the simulation clock.

def test_the_root_prints_one_latency_column():
    keys = set(re.findall(r'\\"(\w+)\\":', (CSMA_V2 / "root.c").read_text()))
    assert {k for k in keys if "latency" in k} == {"latency"}


def test_every_latency_metric_reads_the_one_latency_column():
    items = [i for i in _default_conversion() if "latency" in i["name"] or "latency" in i["column"]]
    assert {i["column"] for i in items} == {"latency"}
    # The objective is the median: under TSCH a few reports wait seconds and the mean follows them.
    assert {"name": "latency", "kind": "median", "column": "latency"} in items


def test_v2_sends_no_probes():
    """Probes are what made the other latencies; without them the root has nothing to send."""
    for name in ("node.c", "root.c", "metrics-packet.h"):
        text = (CSMA_V2 / name).read_text()
        assert "PKT_PING" not in text and "PKT_PONG" not in text, name
    assert "simple_udp_sendto" not in (CSMA_V2 / "root.c").read_text()


def test_latency_uses_the_simulation_clock_on_both_ends():
    """clock_time() counts from each mote's own boot; only the rtimer is the simulation clock."""
    node = (CSMA_V2 / "node.c").read_text()
    root = (CSMA_V2 / "root.c").read_text()
    header = (CSMA_V2 / "metrics-packet.h").read_text()
    assert "r->send_us       = fw_sim_us();" in node
    assert "rx_us - r.send_us" in root
    assert "RTIMER_NOW()" in header.split("fw_sim_us(void)")[1].split("}")[0]
