import numpy as np
import pandas as pd
from logging import Logger
from typing import Callable, Any

def james_stein(means: np.ndarray, variances: np.ndarray) -> float:
    """
    Applies the James-Stein estimator to combine multiple estimates.
    
    The James-Stein estimator "shrinks" individual means toward the global mean,
    resulting in an estimate with lower mean squared error when k >= 3.
    
    Args:
        means: Array of individual means for each group
        variances: Array of variances for each group
        
    Returns:
        Combined mean using the James-Stein estimator
    """
    k = means.size
    if k < 3:
        # James-Stein requires at least 3 groups to be effective
        return float(means.mean())
    
    mu_bar = means.mean()  # Global mean of all means
    diff2 = np.sum((means - mu_bar) ** 2)  # Sum of squared differences
    sigma2 = np.mean(variances)  # Average variance
    
    if diff2 <= 0 or sigma2 <= 0:
        # Degenerate cases: return simple mean
        return float(mu_bar)
    
    # Calculate shrinkage weights: (1 - (k-2)*variance/sum_of_squares)
    weights = (1.0 - ((k - 2) * variances) / max(diff2, 1e-9)).clip(0, 1)
    wsum = weights.sum()
    
    if wsum <= 0:
        return float(mu_bar)
    
    # Return weighted mean using James-Stein weights
    return float(np.sum(weights * means) / wsum)

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

def js_node_mean(df: pd.DataFrame, value_col: str, node_col="node") -> float:
    """
    Applies the James-Stein estimator to node/group means.
    
    Combines means from different nodes using James-Stein shrinkage,
    resulting in a more robust estimate of the global mean.
    
    Args:
        df: DataFrame with data grouped by node
        value_col: Column with values to aggregate
        node_col: Column identifying groups/nodes
        
    Returns:
        Combined mean using James-Stein
    """
    # Group by node and calculate statistics
    groups = df[[node_col, value_col]].dropna().groupby(node_col)[value_col]
    node_means = groups.mean().astype(float).values  # Means by node
    node_vars = groups.var(ddof=1).astype(float).values  # Variances by node
    node_counts = groups.count().astype(float).values  # Counts by node
    
    # Calculate variance of the mean (variance / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_of_mean = np.where(node_counts > 0, node_vars / node_counts, np.nan)
    
    # Fill NaNs with variance mean (if available)
    var_of_mean = np.nan_to_num(
        var_of_mean, 
        nan=np.nanmean(var_of_mean) if np.isfinite(np.nanmean(var_of_mean)) else 0.0
    )
    
    return james_stein(node_means, var_of_mean)

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
    "js_node_mean": lambda df, item, ctx: js_node_mean(
        df,
        item["column"],
        ctx["node_col"],
    ),
}

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
