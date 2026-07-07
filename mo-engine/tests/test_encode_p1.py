from lib.problem.resolve import build_test_adapter
from lib.problem.p1_continuous_mobility import ProblemP1
from lib.problem.chromosomes import ChromosomeP1

def test_p1_encode_basic():
    problem: ProblemP1 = {
        "name": "problem1",
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink":(0.0, 0.0),
        "mobile_nodes":[
            {
                "path_segments": [("0 + 10*t", "0")],
                "is_closed": False,
                "is_round_trip": False,
                "speed": 1.0,
                "time_step": 1.0,
            }
        ],
        "number_of_relays": 2,
        "radius_of_reach": 50.0,
        "radius_of_inter": 60.0,
    }

    adapter = build_test_adapter(problem)

    chrom = ChromosomeP1(
        mac_protocol=0,
        relays=[(10.0, 0.0), (20.0, 0.0)]
    )

    sim = adapter.encode_simulation_input(chrom)

    assert "fixedMotes" in sim
    assert "mobileMotes" in sim
    assert len(sim["fixedMotes"]) == 1 + 2   # sink + relays
    assert len(sim["mobileMotes"]) == 1
    assert sim["fixedMotes"][0]["name"] == "sink"
    assert sim["fixedMotes"][1]["name"] == "relay_0"
    assert sim["fixedMotes"][2]["name"] == "relay_1"
    assert sim["fixedMotes"][1]["position"] == [10.0, 0.0]
    assert sim["fixedMotes"][2]["position"] == [20.0, 0.0]
    assert sim["mobileMotes"][0]["name"] == "mobile_0"
    assert sim["mobileMotes"][0]["functionPath"] == [("0 + 10*t", "0")]
    assert sim["mobileMotes"][0]["speed"] == 1.0


def test_p1_synthetic_vehicle_config():
    """Regression: the synthetic-instances editor encodes a benchmark as a P1
    problem with no mobile nodes and zero coverage requirement. It MUST use the
    canonical name 'problem1' so the adapter resolves, and must encode as
    (sink + N relays) with no mobile motes."""
    problem: ProblemP1 = {
        "name": "problem1",            # NOT 'benchmark' — resolver key
        "region": [-100.0, -100.0, 100.0, 100.0],
        "sink": (0.0, 0.0),
        "mobile_nodes": [],            # no trajectory to cover
        "min_coverage_percentage": 0.0,
        "number_of_relays": 5,
        "radius_of_reach": 200.0,
        "radius_of_inter": 200.0,
    }

    adapter = build_test_adapter(problem)

    chrom = ChromosomeP1(
        mac_protocol=0,
        relays=[(10.0, 20.0), (-30.0, 40.0), (50.0, -60.0), (70.0, 80.0), (-90.0, 15.0)],
    )
    sim = adapter.encode_simulation_input(chrom)

    assert len(sim["fixedMotes"]) == 1 + 5   # sink + relays
    assert len(sim["mobileMotes"]) == 0
    assert sim["fixedMotes"][0]["name"] == "sink"
    assert [m["name"] for m in sim["fixedMotes"][1:]] == [f"relay_{i}" for i in range(5)]
