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


def sum_max(df: pd.DataFrame, value_col: str, node_col: str = "node") -> float:
    """
    Per node, the largest value of the column; summed over nodes.

    The total of a counter that only grows (counted since boot, never reset),
    e.g. reports generated or received. Unlike sum_last_minus_first it does not
    drop each node's first report, and unlike "last" it does not depend on the
    order rows arrived in: a report that reaches the root after a later one
    (TSCH reorders) carries a smaller count than the row before it.
    """
    if value_col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[value_col], errors="coerce")
    per_node = s.groupby(df[node_col]).max()
    return float(per_node.sum(skipna=True))


def _ratio(numerator: float, denominator: float, scale: float) -> float:
    """scale * numerator / denominator; NaN, not infinity, when there is nothing to divide by."""
    if not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(scale * numerator / denominator)


def ratio_sum(df: pd.DataFrame, value_col: str, denominator_col: str, scale: float = 1.0) -> float:
    """
    scale * (sum of the column) / (sum of the denominator column), over all rows.

    For a quantity whose rows add up, divided by the rows' extent, e.g. energy
    over the time it was spent in: total_energy_mj / energy_time_ms * 1000 is
    the mean power in mW over every interval the rows cover.
    """
    if value_col not in df.columns or denominator_col not in df.columns:
        return float("nan")
    return _ratio(sum_all(df, value_col), sum_all(df, denominator_col), scale)


def ratio_sum_max(df: pd.DataFrame, value_col: str, denominator_col: str,
                  node_col: str = "node", scale: float = 1.0) -> float:
    """
    scale * sum_max(column) / sum_max(denominator): the ratio of two counters'
    totals, e.g. reports received over reports generated (delivery ratio).
    """
    if value_col not in df.columns or denominator_col not in df.columns:
        return float("nan")
    return _ratio(sum_max(df, value_col, node_col), sum_max(df, denominator_col, node_col), scale)


def node_coverage(df: pd.DataFrame, node_col: str, deployed_nodes: int | None) -> float:
    """
    Fraction of the deployed sensor nodes that appear in the column at least once.

    The CSV only has rows for nodes whose reports reached the root, so the
    number deployed has to come from elsewhere (the master-node counts the
    motes that wrote to the testlog). NaN when it is not known.
    """
    if node_col not in df.columns or not deployed_nodes:
        return float("nan")
    return float(df[node_col].dropna().nunique() / deployed_nodes)


StatFn = Callable[[pd.DataFrame, dict[str, Any], dict[str, Any]], float]

def _require_column(df: pd.DataFrame, col: str, name: str, log: Logger) -> bool:
    if col not in df.columns:
        log.warning(f"Column '{col}' not found for '{name}'.")
        return False
    return True


#: Kinds that divide by a second column, named in the item's "denominator".
RATIO_KINDS = {"ratio_sum", "ratio_sum_max"}

#: Kinds that count or add up what was reported: with no report, 0.
COUNTING_KINDS = {"sum_all", "sum_max", "sum_last_minus_first", "sum_rate"}


def _nothing_reported(item: dict[str, Any], ctx: dict[str, Any]) -> float:
    """The value of a metric over a run in which no report reached the root.

    Counts are 0, and so is the share of nodes heard from. A mean, median,
    quantile or ratio of nothing has no value.
    """
    kind = item.get("kind")
    if kind in COUNTING_KINDS:
        return 0.0
    if kind == "node_coverage" and ctx.get("deployed_nodes"):
        return 0.0
    return float("nan")


def unavailable(metrics: dict[str, float]) -> list[str]:
    """Names of the metrics that could not be computed (stored as NaN)."""
    return sorted(k for k, v in metrics.items()
                  if not isinstance(v, (int, float)) or not np.isfinite(float(v)))


STAT_DISPATCH: dict[str, StatFn] = {
    "quantile": lambda df, item, ctx: quantile(
        df[item["column"]], item.get("q", 0.5)
    ),
    "mean": lambda df, item, ctx: mean(df[item["column"]]),
    "median": lambda df, item, ctx: median(df[item["column"]]),
    "sum_all": lambda df, item, ctx: sum_all(df, item["column"]),
    "sum_max": lambda df, item, ctx: sum_max(df, item["column"], ctx["node_col"]),
    "ratio_sum": lambda df, item, ctx: ratio_sum(
        df, item["column"], item["denominator"], item.get("scale", 1.0)
    ),
    "ratio_sum_max": lambda df, item, ctx: ratio_sum_max(
        df, item["column"], item["denominator"], ctx["node_col"], item.get("scale", 1.0)
    ),
    "node_coverage": lambda df, item, ctx: node_coverage(
        df, item["column"], ctx.get("deployed_nodes")
    ),
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

        if ctx.get("nothing_reported"):
            results[name] = _nothing_reported(item, ctx)
            continue

        if not col or not _require_column(df, col, name, log):
            results[name] = float("nan")
            continue

        fn = STAT_DISPATCH.get(kind)
        if fn is None:
            log.warning(f"Unknown kind '{kind}' for '{name}'.")
            results[name] = float("nan")
            continue

        if kind in RATIO_KINDS:
            den = item.get("denominator")
            if not den:
                log.warning(f"'{name}' is a {kind} and names no denominator column.")
                results[name] = float("nan")
                continue
            if not _require_column(df, den, name, log):
                results[name] = float("nan")
                continue

        if kind == "node_coverage" and not ctx.get("deployed_nodes"):
            log.warning(f"'{name}' needs the number of deployed nodes, which is not known.")
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
    log: Logger,
    deployed_nodes: int | None = None,
    nothing_reported: bool = False,
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
        deployed_nodes: Sensor nodes in the simulation, for the node_coverage
            kind; the CSV alone cannot tell, it has no row for a node whose
            reports never reached the root.
        nothing_reported: No report reached the root (the CSV is empty):
            counting kinds and node_coverage are 0, the others NaN.

    Returns:
        Dictionary with numerical results for each objective or metric defined in the config.
    """
    ctx = {
        "node_col": cfg.get("node_col", "node"),
        "time_col": cfg.get("time_col", "root_time_now"),
        "deployed_nodes": deployed_nodes,
        "nothing_reported": nothing_reported,
    }

    metrics = _evaluate_items(
        df,
        cfg.get("metrics", []),
        ctx,
        log
    )

    return metrics
