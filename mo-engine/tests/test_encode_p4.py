from strategy.problem.resolve import build_test_adapter
from strategy.problem.p4_mobile_sink_collection import ProblemP4
from strategy.problem.chromosomes import ChromosomeP4

def test_p4_encode_basic():
    problem: ProblemP4 = {
        "name": "problem4",
        "region": [-100.0, 100.0, -100.0, 100.0],

        # -------------------------------------------------
        # Sensores fixos (nodes N)
        # -------------------------------------------------
        "nodes": [
            (10.0, 0.0),
            (20.0, 0.0),
        ],

        # -------------------------------------------------
        # Base do sink
        # -------------------------------------------------
        "sink_base": (0.0, 0.0),

        # -------------------------------------------------
        # Parâmetros de energia/buffer
        # -------------------------------------------------
        "initial_energy": 100.0,
        "buffer_capacity": 50.0,
        "data_rate": 1.0,

        # -------------------------------------------------
        # Mobilidade do sink
        # -------------------------------------------------
        "speed": 10.0,
        "time_step": 1.0,

        # -------------------------------------------------
        # Sojourn locations (L, A)
        # -------------------------------------------------
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
                "adjacency": [0],
                "visibleNodes": [1],
            },
        ],

        # -------------------------------------------------
        # Parâmetros específicos do P4
        # -------------------------------------------------
        "problem_parameters": {
            "base_index": 0,
            "L_stops": [
                (0.0, 0.0),
                (30.0, 0.0),
            ],
            "A_edges": [
                (0, 1),
                (1, 0),
            ],
            "max_route_len": 4,
            "tau_min": 0.0,
            "tau_max": 10.0,
        },

        # -------------------------------------------------
        # Homogêneos
        # -------------------------------------------------
        "radius_of_reach": 50.0,
        "radius_of_inter": 60.0,
    }

    adapter = build_test_adapter(problem)

    # -------------------------------------------------
    # Cromossomo mínimo válido:
    # rota base -> sojourn 1 -> base
    # -------------------------------------------------
    chrom = ChromosomeP4 (
        mac_protocol = 0,
        route = [0, 1, 0],
        sojourn_times = [0.0, 5.0, 0.0],
    )

    sim = adapter.encode_simulation_input(chrom)

    # -------------------------------------------------
    # Estrutura geral
    # -------------------------------------------------
    assert "fixedMotes" in sim
    assert "mobileMotes" in sim

    # -------------------------------------------------
    # Fixed motes: sensores
    # -------------------------------------------------
    assert len(sim["fixedMotes"]) == 2

    assert sim["fixedMotes"][0]["name"] == "node_0"
    assert sim["fixedMotes"][0]["position"] == [10.0, 0.0]

    assert sim["fixedMotes"][1]["name"] == "node_1"
    assert sim["fixedMotes"][1]["position"] == [20.0, 0.0]

    # -------------------------------------------------
    # Mobile motes: sink móvel
    # -------------------------------------------------
    assert len(sim["mobileMotes"]) == 1

    sink = sim["mobileMotes"][0]
    assert sink["name"] == "sink"
    assert sink["isClosed"] is True
    assert sink["speed"] == 10.0
    assert sink["timeStep"] == 1.0

    # functionPath deve existir e ser não vazio
    assert "functionPath" in sink
    assert isinstance(sink["functionPath"], list)
    assert len(sink["functionPath"]) > 0

    # Cada segmento deve ser uma tupla (x(t), y(t))
    for seg in sink["functionPath"]:
        assert isinstance(seg, tuple)
        assert len(seg) == 2
        assert isinstance(seg[0], str)
        assert isinstance(seg[1], str)
