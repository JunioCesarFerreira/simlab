"""How many simulation seeds an instance needs, measured instead of assumed.

A seed study runs one simulation per seed, with everything else fixed, and
reports how each metric is distributed. "How many seeds" turns out to be two
questions with very different answers, so they are kept apart:

* **To estimate** the value of one solution within a target error (``n*``,
  :func:`n_star`). Depends on the noise of that solution alone.
* **To rank** solutions during an optimisation (``m*``, :func:`m_star`).
  Depends on the noise *relative to the difference between solutions*, so it
  needs a panel of chromosomes and has no answer with a single one.

Measured on P1 with 156 individuals and 43 seeds, the first asks for about 244
seeds for the latency and the second for 6. Merging them into one number would
be wrong for one of the two uses.

Every function here is pure: same input, same numbers. Resampling draws from a
generator derived from the study's seed list (:func:`study_rng`), so a report
can be reproduced. Only ``numpy`` is imported at module level; ``scipy`` is
loaded by the functions that need it, so that the seed generator stays
available without the scientific stack.
"""
import json
import math
import random
from hashlib import blake2b
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pylib.seed_aggregation import resolve_aggregator

#: Bumped when a number could change for the same data.
ANALYSIS_VERSION = 1

SEED_MIN = 1
#: Exclusive. Cooja computes ``seed + moteID`` in an ``int``, so the top of the
#: range is left free for the mote ids.
SEED_LIMIT = 2**31 - 2**16
#: The firmware truncates ``seed + moteID`` to an ``unsigned short``, so two
#: seeds only differ for it through their low 16 bits.
LOW16 = 2**16
#: Minimum circular distance between the low 16 bits of two generated seeds.
#: Seeds closer than the number of motes share part of their firmware seeds.
DEFAULT_GUARD = 64

MIN_SEEDS = 3
MAX_SEEDS = 500
MAX_CHROMOSOMES = 100
MAX_SIMULATIONS = 5000
#: Below this, ``m*`` swings too much between panels to be worth reporting
#: without a warning (section 4.4 of the proposal: 5 to 52 with ``K = 5``).
MIN_PANEL = 10

_MAX_DRAWS_PER_SEED = 1000


def generate_seeds(count: int, master_seed: int, guard: int = DEFAULT_GUARD) -> list[int]:
    """A list of ``count`` seeds, stable by prefix.

    The first ``n`` values of a list of ``n + k`` are the list of ``n``, which
    is what lets a study be extended and lets an optimisation use "the first
    ``m``" of a list whose order was fixed before any result existed. That is
    why this walks a single stream and skips the rejects, instead of
    ``random.sample``, whose values depend on the size asked for.

    Values are distinct, and so are their low 16 bits, at least ``guard``
    apart on the circle. :func:`random.random` is the one method whose
    sequence the standard library promises to keep across Python versions.
    """
    if not 1 <= count <= MAX_SEEDS:
        raise ValueError(f"count must be between 1 and {MAX_SEEDS}, got {count}")
    if guard < 1:
        raise ValueError(f"guard must be at least 1, got {guard}")

    stream = random.Random(master_seed)
    blocked = bytearray(LOW16)
    seeds: list[int] = []
    for _ in range(count * _MAX_DRAWS_PER_SEED):
        candidate = SEED_MIN + int(stream.random() * (SEED_LIMIT - SEED_MIN))
        low = candidate % LOW16
        if blocked[low]:
            continue
        seeds.append(candidate)
        for offset in range(1 - guard, guard):
            blocked[(low + offset) % LOW16] = 1
        if len(seeds) == count:
            return seeds
    raise ValueError(
        f"could not place {count} seeds at a distance of {guard} in the low 16 bits")


def validate_seeds(values: Sequence[Any]) -> list[int]:
    """An imported list as integers, or a ``ValueError`` saying what is wrong.

    A duplicate is refused rather than dropped: it would be collapsed in
    silence by the count per seed, and the study would run fewer replicas than
    the list it was given.
    """
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("seeds must be a non-empty list")
    seeds: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"seeds must be whole numbers, got {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"seeds must be whole numbers, got {value!r}")
        seed = int(value)
        if not SEED_MIN <= seed < SEED_LIMIT:
            raise ValueError(
                f"seeds must be in [{SEED_MIN}, {SEED_LIMIT}), got {seed}")
        seeds.append(seed)
    if not MIN_SEEDS <= len(seeds) <= MAX_SEEDS:
        raise ValueError(
            f"a study takes between {MIN_SEEDS} and {MAX_SEEDS} seeds, got {len(seeds)}")
    repeated = sorted({s for s in seeds if seeds.count(s) > 1})
    if repeated:
        raise ValueError(f"duplicate seed(s): {repeated[:5]}")
    return seeds


def diagnose_seeds(seeds: Sequence[int], guard: int = DEFAULT_GUARD) -> dict[str, int]:
    """How much of the list is not independent as far as the firmware can tell.

    Warnings, not refusals: an imported list may come from an earlier
    experiment that someone wants to reproduce as it was.
    """
    low = np.asarray(seeds, dtype=np.int64) % LOW16
    upper = np.triu_indices(low.size, 1)
    apart = np.abs(low[upper[0]] - low[upper[1]])
    apart = np.minimum(apart, LOW16 - apart)
    return {
        "low16_collisions": int((apart == 0).sum()),
        "consecutive_pairs": int(((apart > 0) & (apart < guard)).sum()),
        "guard": guard,
    }


def study_rng(seeds: Sequence[int], *labels: Any) -> np.random.Generator:
    """The generator of one resampling, derived from the study's seed list.

    The labels keep the streams of different metrics and chromosomes apart.
    """
    content = json.dumps([list(seeds), [str(label) for label in labels]])
    digest = blake2b(content.encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# ── one metric on one chromosome ───────────────────────────────────────────


def _number(value: Any) -> Optional[float]:
    """A float JSON can carry: a result here may be read back from a browser."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def describe(values: np.ndarray) -> dict[str, Optional[float]]:
    """Descriptive statistics. Quantiles are type 7, as in ``pylib.statistics``."""
    from scipy import stats

    n = int(values.size)
    out: dict[str, Optional[float]] = {
        key: None for key in ("min", "max", "mean", "sd", "cv", "median", "iqr",
                              "p5", "p95", "skewness", "excess_kurtosis")}
    out["n"] = n
    if n == 0:
        return out
    mean = float(values.mean())
    p5, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    out.update(min=float(values.min()), max=float(values.max()), mean=mean,
               median=float(median), iqr=float(q3 - q1), p5=float(p5), p95=float(p95))
    if n >= 2:
        sd = float(values.std(ddof=1))
        out["sd"] = sd
        out["cv"] = _number(sd / abs(mean)) if mean else None
        if sd > 0 and n >= 3:
            out["skewness"] = _number(stats.skew(values, bias=False))
        if sd > 0 and n >= 4:
            out["excess_kurtosis"] = _number(stats.kurtosis(values, bias=False))
    return out


def ci_mean_t(values: np.ndarray, confidence: float) -> Optional[list[float]]:
    n = values.size
    if n < 2:
        return None
    from scipy import stats

    half = (stats.t.ppf((1 + confidence) / 2, n - 1)
            * values.std(ddof=1) / math.sqrt(n))
    mean = float(values.mean())
    return [mean - float(half), mean + float(half)]


def bca_interval(values: np.ndarray, statistic: str, confidence: float,
                 rng: np.random.Generator, resamples: int = 9999) -> Optional[list[float]]:
    """A bias-corrected and accelerated interval of the mean or of the sd."""
    if values.size < 3:
        return None
    if values.std() == 0:
        constant = float(values.mean()) if statistic == "mean" else 0.0
        return [constant, constant]
    from scipy import stats

    if statistic == "mean":
        function = np.mean
    else:
        def function(sample, axis=-1):
            return np.std(sample, axis=axis, ddof=1)
    result = stats.bootstrap(
        (values,), function, confidence_level=confidence, n_resamples=resamples,
        method="BCa", rng=rng, vectorized=True)
    low, high = result.confidence_interval
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return [float(low), float(high)]


def median_interval(values: np.ndarray, confidence: float) -> Optional[list[float]]:
    """Distribution-free interval of the median, from order statistics.

    ``None`` when the sample is too small for any pair of order statistics to
    reach the confidence asked for.
    """
    from scipy import stats

    n = values.size
    alpha = 1 - confidence
    lower = int(stats.binom.ppf(alpha / 2, n, 0.5))
    upper = int(stats.binom.ppf(1 - alpha / 2, n, 0.5)) + 1
    if lower < 1 or upper > n:
        return None
    ordered = np.sort(values)
    return [float(ordered[lower - 1]), float(ordered[upper - 1])]


def quantile_upper_limit(values: np.ndarray, q: float,
                         confidence: float) -> Optional[float]:
    """An upper confidence limit of the quantile ``q``, distribution-free.

    The smallest order statistic that exceeds the quantile with the confidence
    asked for. ``None`` below the size that makes one exist, which is the
    size the sample maximum alone starts to qualify: 29 seeds for p90 and 59
    for p95 at 95%.
    """
    from scipy import stats

    n = values.size
    position = int(stats.binom.ppf(confidence, n, q)) + 1
    if position > n:
        return None
    return float(np.sort(values)[position - 1])


def clopper_pearson(failures: int, trials: int, confidence: float) -> Optional[list[float]]:
    """Exact interval of a rate, meaningful with no failure at all.

    0 failures in 30 still leave an upper limit of 11.6% at 95%.
    """
    if trials <= 0:
        return None
    from scipy import stats

    alpha = 1 - confidence
    low = 0.0 if failures == 0 else stats.beta.ppf(
        alpha / 2, failures, trials - failures + 1)
    high = 1.0 if failures == trials else stats.beta.ppf(
        1 - alpha / 2, failures + 1, trials - failures)
    return [float(low), float(high)]


def holm(p_values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Holm-adjusted p-values, leaving the missing ones where they are."""
    present = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[Optional[float]] = [None] * len(p_values)
    running = 0.0
    for position, (p, index) in enumerate(present):
        running = max(running, min(1.0, (len(present) - position) * p))
        adjusted[index] = running
    return adjusted


def shapiro_p(values: np.ndarray) -> Optional[float]:
    """Shapiro–Wilk p-value. A diagnostic: it never selects a method.

    With few seeds it has no power, and with many it rejects departures that
    do not matter.
    """
    if values.size < 3 or values.std() == 0:
        return None
    from scipy import stats

    return _number(stats.shapiro(values).pvalue)


def relative_half_width(mean: float, gamma: float) -> float:
    """The half-width that bounds the relative error on the *true* value by γ.

    If ``|ȳ−μ| ≤ γ'|ȳ|`` then ``|ȳ−μ| ≤ γ'(|μ|+|ȳ−μ|)``, so
    ``|ȳ−μ|/|μ| ≤ γ'/(1−γ')``, which is ``γ`` for ``γ' = γ/(1+γ)``.
    """
    return gamma / (1 + gamma) * abs(mean)


def n_star(sd: float, half_width: float, confidence: float) -> Optional[int]:
    """Smallest ``n ≥ 2`` whose t interval is no wider than ``half_width``."""
    if half_width <= 0:
        return None
    if sd == 0:
        return 2
    from scipy import stats

    level = (1 + confidence) / 2
    # The t quantile is above the normal one, so this never starts past the answer.
    n = max(2, math.floor((stats.norm.ppf(level) * sd / half_width) ** 2))
    while stats.t.ppf(level, n - 1) * sd / math.sqrt(n) > half_width:
        n += 1
    return n


def stein_n(sd: float, half_width: float, confidence: float, pilot: int) -> Optional[int]:
    """Stein's two-stage total, which accounts for ``sd`` being an estimate."""
    if half_width <= 0 or pilot < 2:
        return None
    from scipy import stats

    t = stats.t.ppf((1 + confidence) / 2, pilot - 1)
    return max(pilot, math.ceil((t * sd / half_width) ** 2))


def aggregate_rows(matrix: np.ndarray, spec: "str | dict[str, Any]") -> np.ndarray:
    """Ψa over the last axis.

    The vectorised twin of ``AGGREGATOR_DISPATCH``: resampling calls it
    millions of times. A parity test keeps it equal to what the engine computes.
    """
    kind, params = resolve_aggregator(spec)
    if kind == "mean":
        return matrix.mean(axis=-1)
    if kind == "median":
        return np.median(matrix, axis=-1)
    if kind == "min":
        return matrix.min(axis=-1)
    if kind == "max":
        return matrix.max(axis=-1)
    size = matrix.shape[-1]
    drop = math.floor(float(params.get("trim", 0.0)) * size)
    return np.sort(matrix, axis=-1)[..., drop:size - drop].mean(axis=-1)


def _resample_sizes(n: int) -> list[int]:
    """Every size up to 60 seeds, and about 30 more points up to ``n``."""
    sizes = list(range(2, min(n, 60) + 1))
    if n > 60:
        sizes += [int(k) for k in np.unique(np.geomspace(61, n, 30).round())]
    return sizes


def precision_curve(values: np.ndarray, spec: "str | dict[str, Any]", tolerance: float,
                    confidence: float, rng: np.random.Generator,
                    resamples: int = 2000) -> dict[str, Any]:
    """P(Ψ of ``k`` seeds is within ``tolerance`` of Ψ of all of them), by ``k``.

    The non-parametric second opinion on ``n*``, and the only answer for an
    aggregator with no interval of its own, since it applies the real
    estimator. Limited to ``k ≤ n``: resampling cannot speak beyond the
    sample. ``expected`` is the mean of Ψ at each ``k``, which is what shows
    the target of a minimum or a maximum moving with the number of seeds.
    """
    reference = float(aggregate_rows(values, spec))
    curve: list[dict[str, Any]] = []
    needed: Optional[int] = None
    for k in _resample_sizes(values.size):
        estimates = aggregate_rows(values[rng.integers(0, values.size, (resamples, k))], spec)
        within = float((np.abs(estimates - reference) <= tolerance).mean())
        curve.append({"k": k, "p": within, "expected": float(estimates.mean())})
        if needed is None and within >= confidence:
            needed = k
    return {"curve": curve, "n_emp": needed}


def histogram(values: np.ndarray) -> list[dict[str, float]]:
    """Freedman–Diaconis bins, or unit bins for a metric that only takes integers."""
    if values.size == 0:
        return []
    low, high = float(values.min()), float(values.max())
    if low == high:
        return [{"from": low, "to": high, "count": int(values.size)}]
    q1, q3 = np.percentile(values, [25, 75])
    width = 2 * float(q3 - q1) * values.size ** (-1 / 3)
    if not width > 0:
        width = (high - low) / math.sqrt(values.size)
    integers = bool(np.all(values == np.round(values)))
    if integers:
        width = max(1.0, float(round(width)))
        low -= 0.5
        high += 0.5
    bins = min(60, max(1, math.ceil((high - low) / width)))
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return [{"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(bins)]


def boxplot(values: np.ndarray) -> Optional[dict[str, float]]:
    if values.size == 0:
        return None
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    fence = 1.5 * (q3 - q1)
    inside = values[(values >= q1 - fence) & (values <= q3 + fence)]
    return {"low": float(inside.min()), "q1": float(q1), "median": float(median),
            "q3": float(q3), "high": float(inside.max())}


def outlier_flags(values: np.ndarray) -> list[Optional[str]]:
    """Which values stand out, and by which rule. They are never removed.

    Tukey's fences and the robust z of 3.5 on the median absolute deviation
    disagree on skewed data, so a value says which one flagged it.
    """
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    fence = 1.5 * (q3 - q1)
    deviation = float(np.median(np.abs(values - median)))
    flags: list[Optional[str]] = []
    for value in values:
        tukey = value < q1 - fence or value > q3 + fence
        robust = deviation > 0 and abs(0.6745 * (value - median) / deviation) > 3.5
        flags.append("both" if tukey and robust else
                     "tukey" if tukey else "mad" if robust else None)
    return flags


def qq_points(values: np.ndarray) -> list[list[float]]:
    """``[normal quantile, sorted value]``, at Blom's plotting positions."""
    from scipy import stats

    n = values.size
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    return [[float(z), float(y)]
            for z, y in zip(stats.norm.ppf(positions), np.sort(values))]


def running_interval(values: np.ndarray, confidence: float) -> list[dict[str, float]]:
    """The mean and its t interval after each seed, in the order of the list."""
    from scipy import stats

    points: list[dict[str, float]] = []
    for i in range(2, values.size + 1):
        head = values[:i]
        half = stats.t.ppf((1 + confidence) / 2, i - 1) * head.std(ddof=1) / math.sqrt(i)
        mean = float(head.mean())
        points.append({"n": i, "mean": mean, "low": mean - float(half),
                       "high": mean + float(half)})
    return points


# ── ranking solutions: a panel of chromosomes × seeds ──────────────────────


def balanced_matrix(values: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """The largest-looking complete block of a chromosome × seed matrix.

    The crossed model needs every cell. A hole is closed by removing the row
    or column with the highest share of missing cells, one at a time, so one
    chromosome that failed on most seeds does not cost the study those seeds.
    Returns the block and the row and column indices kept.
    """
    rows = list(range(values.shape[0]))
    columns = list(range(values.shape[1]))
    missing = np.isnan(values)
    while rows and columns:
        block = missing[np.ix_(rows, columns)]
        if not block.any():
            break
        by_row, by_column = block.mean(axis=1), block.mean(axis=0)
        if by_row.max() >= by_column.max():
            rows.pop(int(by_row.argmax()))
        else:
            columns.pop(int(by_column.argmax()))
    return values[np.ix_(rows, columns)], rows, columns


def variance_components(values: np.ndarray) -> dict[str, Any]:
    """The crossed model ``y = μ + aᵢ + bₛ + eᵢₛ``, one observation per cell.

    Negative estimates are set to zero, as usual for random effects. ``e``
    holds the interaction too: with one run per cell they cannot be told
    apart, and the determinism of the simulator is what makes that harmless.
    """
    K, n = values.shape
    grand = values.mean()
    by_chromosome = values.mean(axis=1) - grand
    by_seed = values.mean(axis=0) - grand
    total = float(((values - grand) ** 2).sum())
    ss_chromosome = float(n * (by_chromosome ** 2).sum())
    ss_seed = float(K * (by_seed ** 2).sum())
    ss_residual = max(0.0, total - ss_chromosome - ss_seed)
    ms_chromosome = ss_chromosome / (K - 1)
    ms_seed = ss_seed / (n - 1)
    ms_residual = ss_residual / ((K - 1) * (n - 1))
    return {
        "individual": max(0.0, (ms_chromosome - ms_residual) / n),
        "seed": max(0.0, (ms_seed - ms_residual) / K),
        "residual": ms_residual,
        "shares": {
            "individual": ss_chromosome / total if total else 0.0,
            "seed": ss_seed / total if total else 0.0,
            "residual": ss_residual / total if total else 0.0,
        },
    }


def reliability_at(components: Mapping[str, Any], m: int) -> tuple[Optional[float], Optional[float]]:
    """``Eρ²`` and ``Φ`` with ``m`` seeds (the D study of generalizability theory).

    ``Eρ²`` is for ordering solutions evaluated with the same seeds, which is
    how the strategies work. ``Φ`` is for comparing one with an absolute
    threshold, so the main effect of the seed counts against it too.
    """
    individual = components["individual"]
    if individual <= 0:
        return None, None
    residual = components["residual"]
    seed = components["seed"]
    return (individual / (individual + residual / m),
            individual / (individual + (seed + residual) / m))


def m_star(components: Mapping[str, Any], reliability: float) -> Optional[int]:
    """Smallest ``m`` with ``Eρ²(m) ≥ reliability``, ``None`` with no signal.

    Rounded after a small allowance so a value that is whole up to floating
    point does not gain a seed.
    """
    individual = components["individual"]
    if individual <= 0:
        return None
    ratio = reliability / (1 - reliability) * components["residual"] / individual
    return max(1, math.ceil(ratio - 1e-9))


def pigeonhole_interval(values: np.ndarray, reliability: float, confidence: float,
                        rng: np.random.Generator,
                        resamples: int = 1000) -> Optional[list[Optional[int]]]:
    """Interval of ``m*`` by resampling chromosomes and seeds independently.

    The crossed design has two sources of sampling, so resampling only one of
    them understates the error; the pigeonhole bootstrap errs on the wide
    side. ``None`` for the upper end means some resamples had no signal left.
    """
    K, n = values.shape
    estimates = np.empty(resamples)
    for b in range(resamples):
        rows = rng.integers(0, K, K)
        columns = rng.integers(0, n, n)
        sample = values[np.ix_(rows, columns)]
        found = m_star(variance_components(sample), reliability)
        estimates[b] = math.inf if found is None else found
    alpha = 1 - confidence
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2], method="lower")
    return [int(low) if math.isfinite(low) else None,
            int(high) if math.isfinite(high) else None]


def _disjoint_sizes(n: int) -> list[int]:
    """Set sizes for comparing two disjoint sets of seeds, Fibonacci up to n/2."""
    sizes: list[int] = []
    a, b = 1, 2
    while a <= n // 2:
        sizes.append(a)
        a, b = b, a + b
    if n // 2 >= 1 and (not sizes or sizes[-1] != n // 2):
        sizes.append(n // 2)
    return sizes


def _disjoint_draws(n: int, k: int, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(rng.random((draws, n)), axis=1)
    return order[:, :k], order[:, k:2 * k]


def rank_stability(values: np.ndarray, spec: "str | dict[str, Any]", rng: np.random.Generator,
                   draws: int = 200) -> list[dict[str, float]]:
    """Spearman correlation between the rankings of two disjoint sets of ``k`` seeds.

    Rests on no variance model, so it is the check on ``Eρ²``, which is a
    *mean* reliability for a panel whose variance is known not to be
    homogeneous, and the only answer for an aggregator other than the mean.
    """
    from scipy import stats

    K, n = values.shape
    out: list[dict[str, float]] = []
    for k in _disjoint_sizes(n):
        first, second = _disjoint_draws(n, k, draws, rng)
        a = stats.rankdata(aggregate_rows(values[:, first], spec), axis=0).T
        b = stats.rankdata(aggregate_rows(values[:, second], spec), axis=0).T
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        scale = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        usable = scale > 0
        if usable.any():
            correlation = (a[usable] * b[usable]).sum(axis=1) / scale[usable]
            out.append({"k": k, "spearman": float(correlation.mean())})
    return out


def first_reliable(stability: Sequence[Mapping[str, float]], reliability: float) -> Optional[int]:
    """The smallest size whose rankings agree at least this much."""
    return next((int(row["k"]) for row in stability if row["spearman"] >= reliability), None)


def _front_mask(points: np.ndarray) -> np.ndarray:
    """Non-dominated rows of ``points``, every column minimised."""
    no_worse = (points[:, None, :] <= points[None, :, :]).all(axis=2)
    better = (points[:, None, :] < points[None, :, :]).any(axis=2)
    return ~(no_worse & better).any(axis=0)


def front_stability(cube: np.ndarray, specs: Sequence["str | dict[str, Any]"],
                    goals: Sequence[int], rng: np.random.Generator,
                    draws: int = 100) -> list[dict[str, float]]:
    """Jaccard similarity of the Pareto fronts of two disjoint sets of seeds.

    A diagnostic, never a stopping rule: with many near-ties, membership in
    the front keeps changing after the ranking has settled (0.57 with 21
    seeds in the data the proposal was measured on, while Spearman was 0.96).

    ``cube`` is objectives × chromosomes × seeds; ``goals`` is +1 to minimise.
    """
    n = cube.shape[2]
    out: list[dict[str, float]] = []
    for k in _disjoint_sizes(n):
        first, second = _disjoint_draws(n, k, draws, rng)
        scores: list[float] = []
        for b in range(draws):
            front_a = _front_mask(np.stack(
                [goal * aggregate_rows(layer[:, first[b]], spec)
                 for layer, spec, goal in zip(cube, specs, goals)], axis=1))
            front_b = _front_mask(np.stack(
                [goal * aggregate_rows(layer[:, second[b]], spec)
                 for layer, spec, goal in zip(cube, specs, goals)], axis=1))
            union = (front_a | front_b).sum()
            scores.append(float((front_a & front_b).sum() / union) if union else 1.0)
        out.append({"k": k, "jaccard": float(np.mean(scores))})
    return out


def analyse_panel(values: np.ndarray, *, reliability: float, confidence: float,
                  spec: "str | dict[str, Any]", rng: np.random.Generator,
                  with_interval: bool) -> Optional[dict[str, Any]]:
    """Components, reliability curve and ``m*`` of one metric, or ``None``.

    ``values`` is chromosomes × seeds, with NaN where nothing was measured.
    The projection is a statement about the mean, so another aggregator is
    answered by the ranking of disjoint sets instead.
    """
    block, rows, columns = balanced_matrix(values)
    if block.shape[0] < 2 or block.shape[1] < 2:
        return None
    components = variance_components(block)
    needed = m_star(components, reliability)
    limit = min(100, max(block.shape[1], needed or 0))
    curve = []
    for m in range(1, limit + 1):
        e_rho2, phi = reliability_at(components, m)
        curve.append({"m": m, "e_rho2": e_rho2, "phi": phi})
    stability = rank_stability(block, spec, rng)
    by_rank = first_reliable(stability, reliability)
    kind, _ = resolve_aggregator(spec)
    if kind == "mean":
        value, method = needed, "generalizability"
    else:
        value, method = by_rank, "rank_stability"
    interval = None
    if with_interval and kind == "mean" and needed is not None:
        interval = pigeonhole_interval(block, reliability, confidence, rng)
    return {
        "used": {
            "chromosomes": int(block.shape[0]), "seeds": int(block.shape[1]),
            "dropped_chromosomes": sorted(set(range(values.shape[0])) - set(rows)),
            "dropped_seeds": sorted(set(range(values.shape[1])) - set(columns)),
        },
        "components": {k: components[k] for k in ("individual", "seed", "residual")},
        "shares": components["shares"],
        "reliability": curve,
        "m_star": {"value": value, "ci": interval, "method": method},
        "m_rank": by_rank,
        "rank_stability": stability,
        "signal": components["individual"] > 0,
    }
