from lib.problem.resolve import build_test_adapter
from lib.problem.p4_mobile_sink_collection import ProblemP4
from lib.problem.chromosomes import ChromosomeP4


def _extract_start_point(expr: str) -> float:
    if "+" in expr:
        return float(expr.split("+")[0].strip())
    return float(expr.strip())

def _extract_end_point(expr: str) -> float:
    expr = expr.strip()
    if "* t" not in expr:
        return float(expr)

    left, right = expr.split("+", 1)
    a = float(left.strip())
    b = float(right.replace("* t", "").strip())
    return a + b



def test_p4_encode_path_with_multiple_sojourns():
    problem: ProblemP4 = {
        "name": "problem4",
        "region": [-100.0, 100.0, -100.0, 100.0],
        "radius_of_reach": 50.0,
        "radius_of_inter": 60.0,

        "nodes": [
            (10.0, 0.0),
            (20.0, 0.0),
            (20.0, 4.0),
            (15.0, 8.4),
            (21.0, 3.0),
            (12.0, 5.0),
        ],

        "sink_base": (0.0, 0.0),
        "initial_energy": 100.0,
        "buffer_capacity": 50.0,
        "data_rate": 1.0,

        "speed": 10.0,
        "time_step": 1.0,

        "sojourns": [
            {
                "id": 0,
                "position": (0.0, 0.0),
                "adjacency": [1],
                "visibleNodes": [0, 1],
            },
            {
                "id": 1,
                "position": (30.0, 0.0),
                "adjacency": [2],
                "visibleNodes": [1],
            },
            {
                "id": 2,
                "position": (30.0, 30.0),
                "adjacency": [0],
                "visibleNodes": [0],
            },
        ],
    }

    adapter = build_test_adapter(problem)

    # -------------------------------------------------
    # Base -> S1 -> S2 -> Base
    # -------------------------------------------------
    chrom = ChromosomeP4(
        mac_protocol=0,
        route=[0, 1, 2, 0],
        sojourn_times=[0.0, 2.0, 3.0, 0.0],
    )

    sim = adapter.encode_simulation_input(chrom)

    # -------------------------------------------------
    # mobile sink
    # -------------------------------------------------
    sink = sim["mobileMotes"][0]
    path = sink["functionPath"]
    
    assert len(path) > 0

    starts = [
        (
            _extract_start_point(seg[0]),
            _extract_start_point(seg[1]),
        )
        for seg in path
    ]
    
    assert starts[0] == (0.0, 0.0)

    assert (30.0, 0.0) in starts
    assert (30.0, 30.0) in starts

    last_x = _extract_end_point(path[-1][0])
    last_y = _extract_end_point(path[-1][1])
    assert (last_x, last_y) == (0.0, 0.0)

    # -------------------------------------------------
    # Fixed motes
    # -------------------------------------------------
    assert sim["fixedMotes"][0]["name"] == "node_0"
    assert sim["fixedMotes"][0]["position"] == [10.0, 0.0]

    assert sim["fixedMotes"][1]["name"] == "node_1"
    assert sim["fixedMotes"][1]["position"] == [20.0, 0.0]

    assert sim["fixedMotes"][2]["name"] == "node_2"
    assert sim["fixedMotes"][2]["position"] == [20.0, 4.0]