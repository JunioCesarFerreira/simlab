"""The sink is Cooja's mote 1, fixed or mobile.

Mote 1 runs the root firmware and the log reader takes the root's lines as
Mote:1. The builder made the first fixed mote the root, so P4, whose sink is
mobile, ran the root firmware on a sensor.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from pylib import cooja_files
from pylib.cooja_builder.parse_json_pos_dat import mote_order

TEMPLATE = Path(__file__).resolve().parents[2] / "mo-engine" / "simulation_template.xml"


def fixed(name, x, y, source="node.c"):
    return {"name": name, "sourceCode": source, "position": [x, y],
            "radiusOfReach": 50.0, "radiusOfInter": 60.0}


def mobile(name, source, path):
    return {"name": name, "sourceCode": source, "functionPath": path, "isClosed": True,
            "isRoundTrip": False, "speed": 1.0, "timeStep": 1.0,
            "radiusOfReach": 50.0, "radiusOfInter": 60.0}


def config(elements):
    return {"name": "t", "duration": 1, "randomSeed": 7, "radiusOfReach": 50.0,
            "radiusOfInter": 60.0, "region": [0, 0, 100, 100], "simulationElements": elements}


def build(tmp_path, elements):
    csc, pos = tmp_path / "sim.csc", tmp_path / "pos.dat"
    cooja_files.convert_simulation_files(config(elements), str(TEMPLATE), str(csc), str(pos))
    root = ET.parse(csc).getroot()
    motes = {}
    for kind in ("server", "client"):
        for mote in root.find(f".//motetype[description='{kind}']").findall("mote"):
            p = mote.find("interface_config/pos")
            motes[int(mote.find("interface_config/id").text)] = (kind, float(p.get("x")), float(p.get("y")))
    rows = pos.read_text().splitlines() if pos.exists() else []
    return motes, rows


def test_a_mobile_sink_is_mote_1(tmp_path):
    elements = {
        "fixedMotes": [fixed("node_0", 10.0, 0.0), fixed("node_1", 20.0, 0.0)],
        "mobileMotes": [mobile("sink", "sink.c", [["5 + 10 * t", "5"], ["15", "5 + 10 * t"]])],
    }

    motes, rows = build(tmp_path, elements)

    assert motes[1] == ("server", 5.0, 5.0)
    assert motes[2] == ("client", 10.0, 0.0) and motes[3] == ("client", 20.0, 0.0)
    # positions.dat numbers motes from 0 in the same order.
    assert "1 0.00000000 10.00 0.00" in rows and "2 0.00000000 20.00 0.00" in rows
    trajectory = [r for r in rows if r.startswith("0 ")]
    assert trajectory[0] == "0 0.00000000 5.00 5.00" and len(trajectory) > 2


def test_a_fixed_sink_listed_first_keeps_the_order(tmp_path):
    elements = {
        "fixedMotes": [fixed("sink", 0.0, 0.0, "sink.c"), fixed("relay_0", 10.0, 0.0)],
        "mobileMotes": [mobile("m", "node.c", [["10 * t", "20"]])],
    }

    assert mote_order(elements) == [("fixed", 0), ("fixed", 1), ("mobile", 0)]
    motes, _ = build(tmp_path, elements)
    assert motes[1] == ("server", 0.0, 0.0)


def test_a_fixed_sink_listed_later_becomes_mote_1(tmp_path):
    elements = {"fixedMotes": [fixed("relay_0", 10.0, 0.0), fixed("sink", 0.0, 0.0, "sink.c")],
                "mobileMotes": []}

    motes, rows = build(tmp_path, elements)

    assert motes[1] == ("server", 0.0, 0.0) and motes[2] == ("client", 10.0, 0.0)
    assert rows == []  # no mobile mote: no positions file


def test_without_a_sink_the_first_mote_is_the_root(tmp_path):
    elements = {"fixedMotes": [fixed("node0", 1.0, 1.0), fixed("node1", 2.0, 2.0)],
                "mobileMotes": []}
    motes, _ = build(tmp_path, elements)
    assert motes[1] == ("server", 1.0, 1.0)
