"""Seed aggregator (Ψa): collapses the per-seed results of one individual into a
single objective value.

Kept free of pandas so the REST API, which does not install it, can validate
an experiment's ``parameters.simulation.aggregator`` when it is created.
"""
import math
from typing import Any, Callable

AggregatorFn = Callable[[list[float], dict[str, Any]], float]


def _agg_mean(values: list[float], params: dict[str, Any]) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")
    return sum(values) / len(values)


def _agg_median(values: list[float], params: dict[str, Any]) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _agg_trimmed_mean(values: list[float], params: dict[str, Any]) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")
    trim = float(params.get("trim", 0.0))
    if not (0.0 <= trim < 0.5):
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    n = len(values)
    k = math.floor(trim * n)
    s = sorted(values)
    trimmed = s[k: n - k] if k > 0 else s
    if not trimmed:
        return _agg_mean(values, {})
    return sum(trimmed) / len(trimmed)


def _agg_min(values: list[float], params: dict[str, Any]) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")
    return min(values)


def _agg_max(values: list[float], params: dict[str, Any]) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")
    return max(values)


AGGREGATOR_DISPATCH: dict[str, AggregatorFn] = {
    "mean": _agg_mean,
    "median": _agg_median,
    "trimmed_mean": _agg_trimmed_mean,
    "min": _agg_min,
    "max": _agg_max,
}

AGGREGATOR_PARAMS: dict[str, frozenset[str]] = {
    "trimmed_mean": frozenset({"trim"}),
}


def _validate_params(kind: str, params: dict[str, Any]) -> None:
    # A misspelt key would otherwise be dropped and the run would silently
    # use the default, e.g. ``{"kind": "trimmed_mean", "trimm": 0.1}`` → mean.
    unknown = set(params) - AGGREGATOR_PARAMS.get(kind, frozenset())
    if unknown:
        raise ValueError(f"Unknown parameter(s) for aggregator '{kind}': {sorted(unknown)}")
    if "trim" in params:
        trim = params["trim"]
        if isinstance(trim, bool) or not isinstance(trim, (int, float)):
            raise ValueError(f"trim must be a number, got {trim!r}")
        if not (0.0 <= trim < 0.5):
            raise ValueError(f"trim must be in [0, 0.5), got {trim}")


def resolve_aggregator(spec: "str | dict[str, Any]") -> "tuple[str, dict[str, Any]]":
    """Normalize aggregator spec to (kind, params).

    'mean' → ('mean', {})
    {'kind': 'trimmed_mean', 'trim': 0.1} → ('trimmed_mean', {'trim': 0.1})
    Raises ValueError for unknown kinds or invalid parameters.
    """
    if isinstance(spec, str):
        kind, extra = spec, {}
    elif isinstance(spec, dict):
        kind = spec.get("kind", "mean")
        extra = {k: v for k, v in spec.items() if k != "kind"}
    else:
        raise TypeError(f"aggregator spec must be str or dict, got {type(spec)}")
    if kind not in AGGREGATOR_DISPATCH:
        raise ValueError(f"Unknown aggregator '{kind}'. Valid: {list(AGGREGATOR_DISPATCH)}")
    _validate_params(kind, extra)
    return kind, extra


def aggregate_seed_values(values: list[float], aggregator: "str | dict[str, Any]" = "mean") -> float:
    """Collapse a list of per-seed values into a single float using the given aggregator."""
    kind, params = resolve_aggregator(aggregator)
    return AGGREGATOR_DISPATCH[kind](values, params)
