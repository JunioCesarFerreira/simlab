import numpy as np
import pandas as pd
from logging import Logger
from typing import Callable, Any

def quantile(series: pd.Series, q: float) -> float:
    """
    Calculates the quantile of a pandas series, handling missing values.
    
    Args:
        series: Data series
        q: Desired quantile (0.0 to 1.0)
        
    Returns:
        Quantile value or NaN if series is empty
    """
    s = series.dropna().astype(float)
    return float(np.percentile(s, q * 100)) if len(s) else float("nan")

def mean(series: pd.Series) -> float:
    """
    Calculates the mean of a pandas series, handling missing values.
    
    Args:
        series: Data series
        
    Returns:
        Mean or NaN if series is empty
    """
    s = series.dropna().astype(float)
    return float(s.mean()) if len(s) else float("nan")

def median(series: pd.Series) -> float:
    """
    Calculates the median of a pandas series, handling missing values.
    
    Args:
        series: Data series
        
    Returns:
        Median or NaN if series is empty
    """
    s = series.dropna().astype(float)
    return float(s.median()) if len(s) else float("nan")

def sum_last_minus_first(df: pd.DataFrame, value_col: str, 
                        node_col="node", time_col="root_time_now") -> float:
    """
    Calculates the sum of differences (last - first value) by node/group.
    
    Useful for calculating accumulated growth or total variation over time
    for different entities (nodes) in an experiment.
    
    Args:
        df: DataFrame with temporal data
        value_col: Column with values to analyze
        node_col: Column identifying groups/nodes
        time_col: Temporal column for ordering
        
    Returns:
        Total sum of positive variations by node
    """
    if value_col not in df.columns:
        return float("nan")
    
    # Sort by node and time, then group by node
    g = df.sort_values([node_col, time_col]).groupby(node_col)[value_col]
    
    # Get first and last value of each group
    start = g.first()
    end = g.last()
    
    # Calculate difference (last - first) and remove negative values
    per_node = (end - start).clip(lower=0)
    
    return float(per_node.sum())

def sum_rate(df: pd.DataFrame, volume_col: str, 
            time_col="root_time_now") -> float:
    """
    Calculates the average accumulation rate (volume per second).
    
    Args:
        df: DataFrame with temporal data
        volume_col: Column with volume/accumulated values
        time_col: Temporal column in milliseconds
        
    Returns:
        Average rate (total volume / total duration in seconds)
    """
    if volume_col not in df.columns:
        return float("nan")
    
    # Calculate total duration in seconds (convert ms to s)
    duration_s = (df[time_col].max() - df[time_col].min()) / 1000.0
    
    # Total volume sum
    v = float(df[volume_col].dropna().sum())
    
    return float(v / duration_s) if duration_s > 0 else float("nan")

def inverse_of(value: float, scale: float = 1.0) -> float:
    """
    Calculates the inverse of a value with protection against division by zero.
    
    Useful for transforming metrics where higher values are worse
    into metrics where higher values are better.
    
    Args:
        value: Original value
        scale: Scaling factor for the inverse
        
    Returns:
        scale / (value + epsilon) to avoid division by zero
    """
    return float(scale / (value + 1e-9))

def sum_all(df: pd.DataFrame, value_col: str) -> float:
    """
    Sum the whole column (treating non-numeric as NaN).
    """
    if value_col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[value_col], errors="coerce")
    return float(s.sum(skipna=True))


StatFn = Callable[[pd.DataFrame, dict[str, Any], dict[str, Any]], float]

def _require_column(df: pd.DataFrame, col: str, name: str, log: Logger) -> bool:
    if col not in df.columns:
        log.warning(f"Column '{col}' not found for '{name}'.")
        return False
    return True


STAT_DISPATCH: dict[str, StatFn] = {
    "quantile": lambda df, item, ctx: quantile(
        df[item["column"]], item.get("q", 0.5)
    ),
    "mean": lambda df, item, ctx: mean(df[item["column"]]),
    "median": lambda df, item, ctx: median(df[item["column"]]),
    "sum_all": lambda df, item, ctx: sum_all(df, item["column"]),
    "sum_last_minus_first": lambda df, item, ctx: sum_last_minus_first(
        df,
        item["column"],
        ctx["node_col"],
        ctx["time_col"],
    ),
    "sum_rate": lambda df, item, ctx: sum_rate(
        df,
        item["column"],
        ctx["time_col"],
    ),
    "inverse_median": lambda df, item, ctx: inverse_of(
        median(df[item["column"]]),
        item.get("scale", 1.0),
    ),
}

# ---------------------------------------------------------------------------
# Seed aggregator (Ψa) — used by generation repository to collapse per-seed
# simulation results into a single objective value per individual.
# ---------------------------------------------------------------------------

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
    k = int(np.floor(trim * n))
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


def resolve_aggregator(spec: "str | dict[str, Any]") -> "tuple[str, dict[str, Any]]":
    """Normalize aggregator spec to (kind, params).

    'mean' → ('mean', {})
    {'kind': 'trimmed_mean', 'trim': 0.1} → ('trimmed_mean', {'trim': 0.1})
    Raises ValueError for unknown kinds.
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
    return kind, extra


def aggregate_seed_values(values: list[float], aggregator: "str | dict[str, Any]" = "mean") -> float:
    """Collapse a list of per-seed values into a single float using the given aggregator."""
    kind, params = resolve_aggregator(aggregator)
    return AGGREGATOR_DISPATCH[kind](values, params)


def _evaluate_items(
    df: pd.DataFrame,
    items: list[dict[str, Any]],
    ctx: dict[str, Any],
    log: Logger
) -> dict[str, float]:
    results: dict[str, float] = {}

    for item in items:
        name = item["name"]
        col = item.get("column")
        kind = item.get("kind")

        if not col or not _require_column(df, col, name, log):
            results[name] = float("nan")
            continue

        fn = STAT_DISPATCH.get(kind)
        if fn is None:
            log.warning(f"Unknown kind '{kind}' for '{name}'.")
            results[name] = float("nan")
            continue

        try:
            value = fn(df, item, ctx)

            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"Non-finite result: {value}")

            results[name] = float(value)

        except Exception as e:
            log.exception(
                f"Error evaluating '{name}' (kind={kind}, col={col}): {e}"
            )
            results[name] = float("nan")

    return results


def evaluate_config(
    df: pd.DataFrame, 
    cfg: dict[str, list[dict]], 
    log: Logger
    ) -> dict[str, float]:
    """
    Evaluates a DataFrame against a configuration of objectives or metrics.
    
    Processes a JSON/YAML configuration that defines which metrics to calculate
    and how to calculate them, returning a dictionary of results.
    
    Args:
        df: DataFrame with data to evaluate
        cfg: Configuration dictionary with:
            - metrics: List of metrics for monitoring or objectives to optimize
            - node_col: Grouping column name (optional)
            - time_col: Temporal column name (optional)
            
    Returns:
        Dictionary with numerical results for each objective or metric defined in the config.
    """
    ctx = {
        "node_col": cfg.get("node_col", "node"),
        "time_col": cfg.get("time_col", "root_time_now"),
    }

    metrics = _evaluate_items(
        df,
        cfg.get("metrics", []),
        ctx,
        log
    )

    return metrics
