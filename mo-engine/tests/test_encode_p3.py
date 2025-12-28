from strategy.problem.resolve import build_test_adapter
from strategy.problem.p3_target_coverage import ProblemP3


def test_p3_encode_basic():
    problem: ProblemP3 = {
        "name": "problem3",
        "region": [-100.0, 100.0, -100.0, 100.0],
        "sink": (0.0, 0.0),
        "candidates": [
            (10.0, 0.0),
            (20.0, 0.0),
            (30.0, 0.0),
        ],
        "targets": [
            (15.0, 10.0),
        ],
        "k_required": 1,
        "radius_of_reach": 50.0,
        "radius_of_inter": 60.0,
        "radius_of_cover": 30.0,
    }

    adapter = build_test_adapter(problem)

    # Seleciona apenas os candidatos 1 e 2
    chrom = [0, 1, 1]

    sim = adapter.encode_simulation_input(chrom)

    # -------------------------------------------------
    # Estrutura geral
    # -------------------------------------------------
    assert "fixedMotes" in sim
    assert "mobileMotes" in sim

    # sink + 2 relays
    assert len(sim["fixedMotes"]) == 1 + 2
    assert len(sim["mobileMotes"]) == 0

    # -------------------------------------------------
    # Fixed motes
    # -------------------------------------------------
    assert sim["fixedMotes"][0]["name"] == "sink"
    assert sim["fixedMotes"][0]["position"] == [0.0, 0.0]

    assert sim["fixedMotes"][1]["name"] == "relay_1"
    assert sim["fixedMotes"][1]["position"] == [20.0, 0.0]

    assert sim["fixedMotes"][2]["name"] == "relay_2"
    assert sim["fixedMotes"][2]["position"] == [30.0, 0.0]
