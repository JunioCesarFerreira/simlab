"""What a ``batch`` experiment must bring, checked when it is created.

The engine takes the chromosomes as they come: a missing mask reads as "no
relays" and a bit of 2 as "not installed", so a wrong input would run as
something else or fail with the generation already open. Only the shape is
checked here; feasibility stays with the engine, which penalises. Kept free of
the mo-engine and of third-party packages, so the REST API can use it.
"""
from typing import Any, Mapping, Optional

MAX_SEED = 2**31 - 1
#: Beyond it BSON cannot store an integer. NaN and infinity fail it as well.
_MAX_NUMBER = 2**63 - 1
#: ``mac_protocol`` -> key in ``source_repository_options``.
_MAC_KEYS = ("csma", "tsch")


def _is_whole(value: Any, low: int = 0, high: int = _MAX_NUMBER) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def _is_number(value: Any, low: float = -_MAX_NUMBER, high: float = _MAX_NUMBER) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high


def _is_position(value: Any) -> bool:
    return isinstance(value, dict) and _is_number(value.get("x")) and _is_number(value.get("y"))


def _require(condition: bool, field: str, expected: str) -> None:
    if not condition:
        raise ValueError(f"{field} must be {expected}")


def _size_of(problem: Mapping[str, Any]) -> int:
    """How many genes a chromosome of this problem has (for P4, stops to pick from)."""
    name = problem.get("name")
    if name == "problem0":
        size = problem.get("n", problem.get("number_of_variables"))
        _require(_is_whole(size, 1), "parameters.problem.n", "a whole number of at least 1")
    elif name == "problem1":
        size = problem.get("number_of_relays")
        _require(_is_whole(size, 1), "parameters.problem.number_of_relays",
                 "a whole number of at least 1")
    elif name in ("problem2", "problem3", "problem4"):
        key = "sojourns" if name == "problem4" else "candidates"
        items = problem.get(key)
        _require(isinstance(items, list) and len(items) > 0,
                 f"parameters.problem.{key}", "a non-empty list")
        size = len(items)
    else:
        raise ValueError("parameters.problem.name must be one of problem0 to problem4")
    return size


def _check_genes(name: str, size: int, chromosome: Mapping[str, Any], at: str) -> None:
    if name == "problem0":
        x = chromosome.get("x")
        _require(isinstance(x, list) and len(x) == size and all(_is_number(v, 0, 1) for v in x),
                 f"{at}.x", f"a list of {size} numbers in [0, 1]")
        return

    _require(_is_whole(chromosome.get("mac_protocol"), 0, 1),
             f"{at}.mac_protocol", "0 (csma) or 1 (tsch)")
    if name == "problem1":
        relays = chromosome.get("relays")
        _require(isinstance(relays, list) and len(relays) == size and all(map(_is_position, relays)),
                 f"{at}.relays", f'a list of {size} positions {{"x": .., "y": ..}}, one per relay')
    elif name in ("problem2", "problem3"):
        mask = chromosome.get("mask")
        _require(isinstance(mask, list) and len(mask) == size and all(_is_whole(v, 0, 1) for v in mask),
                 f"{at}.mask", f"a list of {size} integers 0 or 1, one per candidate")
    else:
        route, times = chromosome.get("route"), chromosome.get("sojourn_times")
        _require(isinstance(route, list) and len(route) > 0
                 and all(_is_whole(v, 0, size - 1) for v in route),
                 f"{at}.route", f"a non-empty list of sojourn indices in [0, {size - 1}]")
        _require(isinstance(times, list) and len(times) == len(route)
                 and all(_is_number(v, 0) for v in times),
                 f"{at}.sojourn_times", f"a list of {len(route)} non-negative numbers, one per stop")


def validate_batch_input(
    parameters: Mapping[str, Any],
    source_options: Optional[Mapping[str, Any]],
) -> None:
    """Raise ``ValueError``, naming the field, for input that does not fit its problem.

    The source rule is the engine's: a single source runs every chromosome,
    whatever its MAC gene; with several, each gene needs the source of its key.
    """
    if "random_seed" in parameters:
        _require(_is_whole(parameters["random_seed"], 0, MAX_SEED),
                 "parameters.random_seed", f"a whole number in [0, {MAX_SEED}]")

    chromosomes = parameters.get("chromosomes")
    _require(isinstance(chromosomes, list) and len(chromosomes) > 0,
             "parameters.chromosomes", "a non-empty list")

    problem = parameters.get("problem") or {}
    size = _size_of(problem)
    name = problem["name"]

    synthetic = (parameters.get("simulation") or {}).get("synthetic")
    simulated = not (isinstance(synthetic, dict) and synthetic.get("enabled"))
    sources = {key for key, value in (source_options or {}).items() if value}
    if simulated:
        _require(bool(sources), "source_repository_options",
                 "given, because the batch is not synthetic")

    for index, chromosome in enumerate(chromosomes):
        at = f"parameters.chromosomes[{index}]"
        _require(isinstance(chromosome, dict), at, "an object")
        _check_genes(name, size, chromosome, at)
        if simulated and name != "problem0" and len(sources) > 1:
            needed = _MAC_KEYS[chromosome["mac_protocol"]]
            _require(needed in sources, f"{at}.mac_protocol",
                     f"the gene of one of the sources: there is no '{needed}' in "
                     f"source_repository_options")
