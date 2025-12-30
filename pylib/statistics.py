import numpy as np
import pandas as pd
from logging import Logger

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


def evaluate_config(df: pd.DataFrame, cfg: dict[str, list[dict]], log: Logger) -> tuple[dict[str, float], dict[str, float]]:
    """
    Evaluates a DataFrame against a configuration of objectives and metrics.
    
    Processes a JSON/YAML configuration that defines which metrics to calculate
    and how to calculate them, returning dictionaries of results.
    
    Args:
        df: DataFrame with data to evaluate
        cfg: Configuration dictionary with:
            - objectives: List of objectives to optimize
            - metrics: List of metrics for monitoring
            - node_col: Grouping column name (optional)
            - time_col: Temporal column name (optional)
            
    Returns:
        Tuple: (objectives_dict, metrics_dict) with numerical results
    """
    obj, met = {}, {}  # Dictionaries for objectives and metrics
    
    # Process objectives (typically used for optimization)
    for item in cfg.get("objectives", []):
        col, kind, name = item["column"], item["kind"], item["name"]
        
        if col not in df.columns:
            log.warning(f"Column '{col}' not found in DataFrame for objective '{name}'.")
            obj[name] = float("nan")
            continue
            
        # Select aggregation function based on type
        if kind == "quantile":
            obj[name] = quantile(df[col], item.get("q", 0.5))
        elif kind == "mean":
            obj[name] = mean(df[col])
        elif kind == "median":
            obj[name] = median(df[col])
        elif kind == "sum_last_minus_first":
            obj[name] = sum_last_minus_first(df, col, 
                                           cfg.get("node_col", "node"), 
                                           cfg.get("time_col", "root_time_now"))
        elif kind == "sum_all":
            obj[name] = sum_all(df, col)
        elif kind == "inverse_median":
            obj[name] = inverse_of(median(df[col]), item.get("scale", 1.0))
        elif kind == "js_node_mean":
            obj[name] = js_node_mean(df, col, cfg.get("node_col", "node"))
        else:
            log.warning(f"Unknown objective kind '{kind}' for objective '{name}'.")
            obj[name] = float("nan")
    
    # Process metrics (typically used for monitoring)
    for item in cfg.get("metrics", []):
        col, kind, name = item["column"], item["kind"], item["name"]
        
        if col not in df.columns:
            log.warning(f"Column '{col}' not found in DataFrame for metric '{name}'.")
            met[name] = float("nan")
            continue
            
        if kind == "quantile":
            met[name] = quantile(df[col], item.get("q", 0.5))
        elif kind == "mean":
            met[name] = mean(df[col])
        elif kind == "median":
            met[name] = median(df[col])
        elif kind == "sum_all":
            obj[name] = sum_all(df, col)
        elif kind == "sum_last_minus_first":
            met[name] = sum_last_minus_first(df, col, 
                                           cfg.get("node_col", "node"), 
                                           cfg.get("time_col", "root_time_now"))
        elif kind == "sum_rate":
            met[name] = sum_rate(df, col, cfg.get("time_col", "root_time_now"))
        else:
            log.warning(f"Unknown metric kind '{kind}' for metric '{name}'.")
            met[name] = float("nan")
    
    return obj, met