import json
import re
import pandas as pd
from pathlib import Path

# Root (Mote:1) JSON metrics line, e.g. "[Mote:1] {"node":"...", ...}".
_ROOT_JSON_PATTERN = re.compile(r'\[Mote:1\].*?(\{.*?\})')


def read_root_records(cooja_log_input: Path) -> list[dict]:
    """Extract the root's per-node JSON metric records from a COOJA testlog.

    Each record is one metrics report received by the root (Mote:1) from a
    sensor node, carrying at least ``node`` and ``root_time_now``. Lines that
    are not root JSON, or that fail to parse, are skipped.
    """
    rows: list[dict] = []
    with cooja_log_input.open(encoding="utf-8") as f:
        for line in f:
            m = _ROOT_JSON_PATTERN.search(line)
            if not m:
                continue
            try:
                rows.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                continue
    return rows


def cooja_log_to_csv(cooja_log_input: Path, csv_output: Path) -> pd.DataFrame:
    rows = read_root_records(cooja_log_input)

    # -------------------------- DataFrame bruto --------------------------------
    df = pd.DataFrame(rows)

    REQUIRED_COLUMNS = {"node", "root_time_now"}
    missing = REQUIRED_COLUMNS - set(df.columns)
    if not missing:
        df.sort_values(["node", "root_time_now"], inplace=True)

    df.to_csv(csv_output, index=False)
    return df
