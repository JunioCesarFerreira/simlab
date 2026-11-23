"""A run in which no report reached the root is a result, and it can be read.

The converter wrote one blank line, which is not empty by size, and reading it
back raised EmptyDataError: the run went through the collection retry and ended
in Error, its testlog discarded.
"""
import logging
import math

import pandas as pd

from pylib.cooja_builder.csv_converter import cooja_log_to_csv
from pylib.statistics import evaluate_config, unavailable

LOG = logging.getLogger(__name__)
CONFIG = {"node_col": "node", "time_col": "root_time_now", "metrics": [
    {"name": "delivered", "kind": "sum_last_minus_first", "column": "server_received"},
    {"name": "energy", "kind": "sum_all", "column": "total_energy_mj"},
    {"name": "coverage", "kind": "node_coverage", "column": "node"},
    {"name": "latency", "kind": "median", "column": "latency"},
    {"name": "power", "kind": "ratio_sum", "column": "e", "denominator": "t"},
]}


def test_a_log_without_root_reports_converts_to_an_empty_file(tmp_path):
    log = tmp_path / "COOJA.testlog"
    log.write_text("00:00.000\tID:2\tStarting\n00:01.000\tID:1\tno json here\n")
    csv = tmp_path / "out.csv"

    df = cooja_log_to_csv(log, csv)

    assert df.empty and csv.stat().st_size == 0


def test_a_log_with_reports_still_converts_as_before(tmp_path):
    log = tmp_path / "COOJA.testlog"
    log.write_text('00:01.000\tID:1\t[Mote:1] {"node":"a","root_time_now":5,"latency":3}\n')
    csv = tmp_path / "out.csv"

    cooja_log_to_csv(log, csv)

    assert pd.read_csv(csv).to_dict("records") == [{"node": "a", "root_time_now": 5, "latency": 3}]


def test_nothing_reported_counts_zero_and_leaves_the_rest_without_value():
    metrics = evaluate_config(pd.DataFrame(), CONFIG, LOG, deployed_nodes=12, nothing_reported=True)

    assert metrics["delivered"] == 0.0 and metrics["energy"] == 0.0 and metrics["coverage"] == 0.0
    assert math.isnan(metrics["latency"]) and math.isnan(metrics["power"])
    assert unavailable(metrics) == ["latency", "power"]


def test_without_the_number_of_nodes_coverage_has_no_value():
    metrics = evaluate_config(pd.DataFrame(), CONFIG, LOG, nothing_reported=True)
    assert math.isnan(metrics["coverage"])
