from fastapi import APIRouter

router = APIRouter()

# Static catalogue of supported benchmark functions.
# Extend this list when new benchmarks are added to synthetic_data.py.
_BENCHMARKS = [
    {
        "id": "DTLZ2",
        "label": "DTLZ2",
        "min_objectives": 2,
        "max_objectives": None,
        "description": (
            "Scalable benchmark with a Pareto front on the unit hypersphere segment "
            "in the first orthant.  Supports any number of objectives M ≥ 2 and "
            "requires n ≥ M-1 decision variables.  "
            "Reference: Deb et al. (2005)."
        ),
        "n_min_formula": "n ≥ M-1",
    },
    {
        "id": "ZDT1",
        "label": "ZDT1",
        "min_objectives": 2,
        "max_objectives": 2,
        "description": (
            "Two-objective benchmark with a convex Pareto front.  "
            "f1 = x1,  g = 1 + 9·∑(x2..xn)/(n-1),  f2 = g·(1 - √(f1/g)).  "
            "Reference: Zitzler, Deb & Thiele (2000)."
        ),
        "n_min_formula": "n ≥ 2",
    },
    {
        "id": "SCH1",
        "label": "SCH1",
        "min_objectives": 2,
        "max_objectives": 2,
        "description": (
            "Schaffer's two-objective problem with one effective decision variable.  "
            "f1 = x²,  f2 = (x-2)².  The Pareto front lies on the curve connecting "
            "the two parabola vertexes.  "
            "Reference: Schaffer (1985)."
        ),
        "n_min_formula": "n ≥ 1",
    },
]


@router.get("/benchmarks")
def list_benchmarks() -> list[dict]:
    """Return the catalogue of available synthetic benchmark functions."""
    return _BENCHMARKS
