from lib.problem.resolve import build_test_adapter
from lib.problem.p2_discrete_mobility import ProblemP2
from lib.problem.chromosomes import ChromosomeP2


def test_p2_encode_basic():
    problem: ProblemP2 = {
        "name": "problem2",
        "region": [-100.0, 100.0, -100.0, 100.0],
        "sink": (0.0, 0.0),
        "candidates": [
            (10.0, 0.0),
            (20.0, 0.0),
            (30.0, 0.0),
        ],
        "mobile_nodes": [
            {
                "path_segments": [("0 + 5*t", "0")],
                "is_closed": False,
                "is_round_trip": False,
                "speed": 1.0,
                "time_step": 1.0,
            }
        ],
        "radius_of_reach": 50.0,
        "radius_of_inter": 60.0,
    }

    adapter = build_test_adapter(problem)

    # Seleciona apenas os candidatos 0 e 2
    chrom =  ChromosomeP2(
        mac_protocol = 0,
        mask = [1, 0, 1]
    )

    sim = adapter.encode_simulation_input(chrom)

    # -------------------------------------------------
    # Estrutura geral
    # -------------------------------------------------
    assert "fixedMotes" in sim
    assert "mobileMotes" in sim

    # sink + 2 relays
    assert len(sim["fixedMotes"]) == 1 + 2
    assert len(sim["mobileMotes"]) == 1

    # -------------------------------------------------
    # Fixed motes
    # -------------------------------------------------
    assert sim["fixedMotes"][0]["name"] == "sink"

    assert sim["fixedMotes"][1]["name"] == "relay_0"
    assert sim["fixedMotes"][1]["position"] == [10.0, 0.0]

    assert sim["fixedMotes"][2]["name"] == "relay_2"
    assert sim["fixedMotes"][2]["position"] == [30.0, 0.0]

    # -------------------------------------------------
    # Mobile motes (predefinidos no problema)
    # -------------------------------------------------
    assert sim["mobileMotes"][0]["name"] == "mobile_0"
    assert sim["mobileMotes"][0]["functionPath"] == [("0 + 5*t", "0")]
    assert sim["mobileMotes"][0]["speed"] == 1.0
