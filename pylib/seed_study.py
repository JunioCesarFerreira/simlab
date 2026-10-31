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


def _shape(values: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """Bias-corrected skewness and excess kurtosis (G₁ and G₂).

    Written out because a study takes these once per metric and chromosome, and
    through ``scipy.stats`` they were half the time of the whole analysis.
    """
    n = values.size
    centred = values - values.mean()
    m2 = float((centred ** 2).mean())
    if n < 3 or m2 == 0:
        return None, None
    skewness = float((centred ** 3).mean()) / m2 ** 1.5 * math.sqrt(n * (n - 1)) / (n - 2)
    if n < 4:
        return _number(skewness), None
    kurtosis = float((centred ** 4).mean()) / m2 ** 2 - 3
    return _number(skewness), _number(
        (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * kurtosis + 6))


def describe(values: np.ndarray) -> dict[str, Optional[float]]:
    """Descriptive statistics. Quantiles are type 7, as in ``pylib.statistics``."""
    n = int(values.size)
    out: dict[str, Optional[float]] = {
        key: None for key in ("min", "max", "mean", "sd", "cv", "median", "q1", "q3", "iqr",
                              "p5", "p95", "skewness", "excess_kurtosis")}
    out["n"] = n
    if n == 0:
        return out
    mean = float(values.mean())
    p5, q1, median, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    out.update(min=float(values.min()), max=float(values.max()), mean=mean,
               median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
               p5=float(p5), p95=float(p95))
    if n >= 2:
        sd = float(values.std(ddof=1))
        out["sd"] = sd
        out["cv"] = _number(sd / abs(mean)) if mean else None
        out["skewness"], out["excess_kurtosis"] = _shape(values)
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


# ── a whole study ──────────────────────────────────────────────────────────

#: A metric at this size is an unsigned subtraction that wrapped around, not a
#: measurement: ``n2r_latency`` subtracts clocks that are not synchronised.
OVERFLOW_LIMIT = 2.0**63
#: What each number of seeds allows a report to claim.
MIN_FOR_INTERVALS = 10
MIN_FOR_SHAPE = 30
#: A value of ``n*`` past this multiple of ``n`` is a long extrapolation.
INDICATIVE_FACTOR = 3
#: How far ``n*`` and the resampling estimate may disagree before it is said.
DIVERGENCE = 0.25
RESAMPLES_SUMMARY = 1000
RESAMPLES_DETAIL = 2000

DEFAULT_TARGETS = {"confidence": 0.95, "relative_error": 0.05, "reliability": 0.90}

_NO_PRECISION = ("min", "max")
_INFRASTRUCTURE = ("could not reach", "could not fetch", "ssh", "recovered:", "disappeared")


def _warning(code: str, message: str, metric: Optional[str] = None) -> dict[str, Any]:
    return {"code": code, "metric": metric, "message": message}


def _observed(row: np.ndarray) -> np.ndarray:
    return row[~np.isnan(row)]


def metric_status(matrix: np.ndarray) -> str:
    """What a metric's numbers can be used for, before any statistic is taken."""
    observed = matrix[~np.isnan(matrix)]
    if observed.size == 0 or int((~np.isnan(matrix)).sum(axis=1).max()) < 2:
        return "no_data"
    if (np.abs(observed) >= OVERFLOW_LIMIT).any():
        return "overflow_suspect"
    if all(_observed(row).std() == 0 for row in matrix if _observed(row).size):
        return "constant"
    return "ok"


def _fingerprint(matrix: np.ndarray) -> bytes:
    return np.nan_to_num(matrix, nan=0.0).tobytes() + np.isnan(matrix).tobytes()


def group_aliases(values: Mapping[str, np.ndarray], order: Sequence[str]) -> dict[str, list[str]]:
    """Metrics with identical numbers, under the first of them in ``order``.

    ``latency`` and ``rtt_latency`` read one column, and counting both would
    present one measurement as two. Constants are left out: three metrics that
    are always zero are not the same metric.
    """
    groups: dict[bytes, list[str]] = {}
    for name in order:
        if metric_status(values[name]) == "ok":
            groups.setdefault(_fingerprint(values[name]), []).append(name)
    return {names[0]: names[1:] for names in groups.values()}


def _tolerance(y: np.ndarray, spec: "str | dict[str, Any]", gamma: float,
               absolute: Optional[float]) -> float:
    """The half-width resampling has to land in: the one ``n*`` is computed for.

    The same ``γ'`` on both sides, or the two estimates of one quantity would
    differ by construction and their disagreement would mean nothing.
    """
    if absolute is not None:
        return absolute
    return relative_half_width(float(aggregate_rows(y, spec)), gamma)


def _precision_of(y: np.ndarray, *, spec: "str | dict[str, Any]", confidence: float,
                  gamma: float, absolute: Optional[float], planned: int,
                  rng: Optional[np.random.Generator]) -> dict[str, Any]:
    """How many seeds to estimate this metric on this chromosome.

    ``rng`` asks for the resampling estimate as well, which only objectives
    pay for.
    """
    out: dict[str, Any] = {"value": None, "stein": None, "empirical": None,
                           "method": "t", "status": "ok"}
    kind, _ = resolve_aggregator(spec)
    if kind in _NO_PRECISION:
        # The expected extreme moves with the number of seeds, so there is no
        # fixed value for k seeds to be close to.
        out.update(method="none", status="not_applicable")
        return out
    if y.size < 2:
        out["status"] = "too_few"
        return out
    mean, sd = float(y.mean()), float(y.std(ddof=1))
    interval = ci_mean_t(y, confidence)
    if absolute is not None:
        half_width = absolute
    elif interval[0] <= 0 <= interval[1] and sd > 0:
        out["status"] = "near_zero_mean"
        return out
    else:
        half_width = relative_half_width(mean, gamma)

    if kind == "mean":
        out["value"] = n_star(sd, half_width, confidence)
        out["stein"] = stein_n(sd, half_width, confidence, y.size)
    else:
        out["method"] = "empirical"
    if rng is not None and y.size >= 3:
        out["empirical"] = precision_curve(
            y, spec, _tolerance(y, spec, gamma, absolute), confidence, rng,
            RESAMPLES_SUMMARY)["n_emp"]
    if out["method"] == "empirical":
        out["value"] = out["empirical"]
    if out["value"] is not None:
        if out["value"] > INDICATIVE_FACTOR * max(y.size, 1) or y.size < MIN_FOR_INTERVALS:
            out["status"] = "indicative"
        elif out["value"] > max(y.size, planned):
            out["status"] = "extrapolated"
    elif out["method"] == "empirical":
        out["status"] = "beyond_sample"
    return out


def _chromosome_row(genome_hash: str, y: np.ndarray, confidence: float) -> dict[str, Any]:
    row: dict[str, Any] = {"genome_hash": genome_hash, **describe(y)}
    row["ci_mean_t"] = ci_mean_t(y, confidence) if y.size >= MIN_FOR_INTERVALS else None
    row["shapiro_p"] = shapiro_p(y) if y.size >= MIN_FOR_INTERVALS else None
    row["shapiro_p_holm"] = None
    row["n_star"] = None
    return row


def _spread(numbers: Sequence[Optional[float]]) -> Optional[dict[str, float]]:
    present = [float(v) for v in numbers if v is not None]
    if not present:
        return None
    return {"p50": float(np.percentile(present, 50)), "p90": float(np.percentile(present, 90)),
            "max": max(present), "min": min(present)}


def analyse_study(values: Mapping[str, np.ndarray], *, seeds: Sequence[int],
                  chromosomes: Sequence[str], objectives: Sequence[Mapping[str, Any]],
                  targets: Optional[Mapping[str, float]] = None,
                  absolute: Optional[Mapping[str, float]] = None) -> dict[str, Any]:
    """Every metric of a study, and the two answers to "how many seeds".

    ``values`` maps a metric to a chromosomes × seeds matrix, with NaN where
    nothing was measured yet, so a running study can be read too.
    ``absolute`` gives a metric an absolute half-width, the only target that
    makes sense when its mean is close to zero.
    """
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    confidence = targets["confidence"]
    gamma = targets["relative_error"]
    reliability = targets["reliability"]
    absolute = dict(absolute or {})
    K, n = len(chromosomes), len(seeds)

    goal_of = {o["metric_name"]: o.get("goal", "min") for o in objectives}
    spec_of = {o["metric_name"]: o.get("aggregator", "mean") for o in objectives}
    objective_names = [o["metric_name"] for o in objectives if o["metric_name"] in values]
    order = objective_names + [m for m in values if m not in goal_of]
    aliases = group_aliases(values, order)
    alias_of = {alias: main for main, names in aliases.items() for alias in names}

    warnings: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for name in order:
        if name in alias_of:
            continue
        matrix = values[name]
        status = metric_status(matrix)
        observed = [_observed(row) for row in matrix]
        is_objective = name in goal_of
        spec = spec_of.get(name, "mean")

        rows: list[dict[str, Any]] = []
        for i, genome_hash in enumerate(chromosomes):
            y = observed[i]
            row = _chromosome_row(genome_hash, y, confidence)
            if status == "ok":
                row["n_star"] = _precision_of(
                    y, spec=spec, confidence=confidence, gamma=gamma,
                    absolute=absolute.get(name), planned=n,
                    rng=study_rng(seeds, name, genome_hash) if is_objective else None)
            rows.append(row)

        panel = None
        if status == "ok" and K >= 2 and n >= 2:
            panel = analyse_panel(matrix, reliability=reliability, confidence=confidence,
                                  spec=spec, rng=study_rng(seeds, name, "panel"),
                                  with_interval=is_objective)
        metrics.append({
            "name": name, "status": status, "aliases": aliases.get(name, []),
            "is_objective": is_objective, "goal": goal_of.get(name), "aggregator": spec,
            "observed": int(sum(y.size for y in observed)),
            "planned": K * n,
            "all_zero": status == "constant" and not np.nan_to_num(matrix).any(),
            "by_chromosome": rows,
            "panel": panel,
            "summary": {
                "n_star": _spread([(r["n_star"] or {}).get("value") for r in rows]),
                "empirical": _spread([(r["n_star"] or {}).get("empirical") for r in rows]),
                "cv": _spread([r["cv"] for r in rows]),
                "statuses": sorted({(r["n_star"] or {}).get("status", status) for r in rows}),
            },
        })

    # Holm across metrics, per chromosome: the family is what one reader of one
    # table is testing at once.
    for i in range(K):
        adjusted = holm([m["by_chromosome"][i]["shapiro_p"] for m in metrics])
        for metric, p in zip(metrics, adjusted):
            metric["by_chromosome"][i]["shapiro_p_holm"] = p
    for metric in metrics:
        tested = [r["shapiro_p_holm"] for r in metric["by_chromosome"]
                  if r["shapiro_p_holm"] is not None]
        metric["summary"]["normality_rejected"] = (
            sum(p < 1 - confidence for p in tested) / len(tested) if tested else None)

    by_name = {m["name"]: m for m in metrics}
    by_name.update({alias: by_name[main] for alias, main in alias_of.items() if main in by_name})

    report: dict[str, Any] = {}
    optimise: dict[str, Any] = {}
    for objective in objective_names:
        metric = by_name.get(objective)
        if metric is None:
            continue
        spread = metric["summary"]["n_star"]
        kind, _ = resolve_aggregator(metric["aggregator"])
        method = "none" if kind in _NO_PRECISION else (
            "t" if kind == "mean" else "empirical")
        report[objective] = {
            "value": math.ceil(spread["p50"]) if spread else None,
            "p90": math.ceil(spread["p90"]) if spread else None,
            "max": math.ceil(spread["max"]) if spread else None,
            "method": method,
            "status": _report_status(metric, spread, n),
        }
        panel = metric["panel"]
        optimise[objective] = {
            "value": panel["m_star"]["value"] if panel else None,
            "ci": panel["m_star"]["ci"] if panel else None,
            "method": panel["m_star"]["method"] if panel else None,
        }
        _warn_objective(warnings, objective, metric, report[objective], n, K)

    joint = _joint_precision(values, objective_names, spec_of, seeds, confidence, gamma, absolute)
    needed = [row["value"] for row in optimise.values() if row["value"] is not None]
    if K >= 2 and K < MIN_PANEL:
        warnings.append(_warning(
            "small_panel",
            f"The panel has {K} chromosomes. The number of seeds to rank solutions swings "
            f"between panels of this size; use at least {MIN_PANEL}."))
    if n < MIN_FOR_SHAPE:
        warnings.append(_warning(
            "few_seeds",
            f"{n} seeds. Below {MIN_FOR_SHAPE} the shape of a distribution is not "
            "reliable and every n* is only an indication."))

    front = None
    usable = [by_name[o] for o in objective_names if o in by_name]
    if K >= 2 and len(usable) >= 2 and all(m["status"] == "ok" for m in usable):
        blocks = [values[o] for o in objective_names]
        stacked = np.stack(blocks)
        complete = ~np.isnan(stacked).any(axis=(0, 1))
        if int(complete.sum()) >= 2:
            front = front_stability(
                stacked[:, :, complete],
                [spec_of[o] for o in objective_names],
                [1 if goal_of[o] == "min" else -1 for o in objective_names],
                study_rng(seeds, "front"))

    return {
        "analysis_version": ANALYSIS_VERSION,
        "targets": targets,
        "shape": {"chromosomes": K, "seeds": n},
        "metrics": metrics,
        "warnings": warnings,
        "front_stability": front,
        "recommendation": {
            "report": {"by_objective": report, "joint": joint},
            "optimise": {"by_objective": optimise, "max": max(needed) if needed else None},
        },
    }


def _report_status(metric: Mapping[str, Any], spread: Optional[Mapping[str, float]],
                   n: int) -> str:
    """How far to trust the number reported, read on that number.

    On a panel, one chromosome near zero must not mark the typical answer as a
    long extrapolation, so the rule applies to the median across chromosomes.
    """
    if metric["status"] != "ok":
        return metric["status"]
    if spread is None:
        statuses = metric["summary"]["statuses"]
        return next((s for s in ("not_applicable", "near_zero_mean", "beyond_sample",
                                 "too_few") if s in statuses), "ok")
    if spread["p50"] > INDICATIVE_FACTOR * n or n < MIN_FOR_INTERVALS:
        return "indicative"
    return "extrapolated" if spread["p50"] > n else "ok"


def _estimates_diverge(rows: Sequence[Mapping[str, Any]], n: int) -> bool:
    """Whether ``n*`` and resampling disagree, where both can speak.

    Only inside the sample: resampling stops at ``n``, so comparing it with an
    ``n*`` past that compares a number with its own cap. A difference of a seed
    or two says nothing when the answer itself is three.
    """
    gaps = []
    for row in rows:
        precision = row.get("n_star") or {}
        value, empirical = precision.get("value"), precision.get("empirical")
        if value and empirical and value <= n:
            gaps.append((abs(empirical - value), abs(empirical - value) / value))
    if not gaps:
        return False
    return (float(np.median([gap[0] for gap in gaps])) > 2
            and float(np.median([gap[1] for gap in gaps])) > DIVERGENCE)


def _warn_objective(warnings: list[dict[str, Any]], name: str, metric: Mapping[str, Any],
                    report: Mapping[str, Any], n: int, K: int) -> None:
    """What a reader of the recommendation has to know about one objective."""
    status = metric["status"]
    if status == "overflow_suspect":
        warnings.append(_warning(
            status, f"{name} holds values near 2⁶³: an unsigned subtraction wrapped "
            "around. It is a measurement defect, so no number of seeds is recommended.", name))
    elif status == "constant":
        warnings.append(_warning(
            status, f"{name} does not change with the seed, so seeds add nothing to it.", name))
    elif report["status"] == "not_applicable":
        warnings.append(_warning(
            "aggregator_no_precision",
            f"{name} uses {resolve_aggregator(metric['aggregator'])[0]}: the expected extreme "
            "moves with the number of seeds, so a precision target has no fixed value to "
            "be close to. See how it moves in the detail.", name))
    elif "near_zero_mean" in metric["summary"]["statuses"]:
        warnings.append(_warning(
            "near_zero_mean",
            f"The interval of {name} contains zero, so a relative error has no meaning. "
            "It needs an absolute target.", name))
    elif report["value"] is not None and report["value"] > n:
        warnings.append(_warning(
            "extrapolated",
            f"Estimating {name} takes about {report['value']} seeds, more than the {n} "
            "that ran: the number is an extrapolation.", name))

    if report["method"] == "t" and _estimates_diverge(metric["by_chromosome"], n):
        warnings.append(_warning(
            "divergent_estimates",
            f"For {name}, the t formula and resampling disagree by more than "
            f"{DIVERGENCE:.0%}: the distribution is far from normal.", name))
    panel = metric["panel"]
    if panel and (panel["used"]["dropped_chromosomes"] or panel["used"]["dropped_seeds"]):
        warnings.append(_warning(
            "panel_unbalanced",
            f"For {name}, the panel analysis left out "
            f"{len(panel['used']['dropped_chromosomes'])} chromosome(s) and "
            f"{len(panel['used']['dropped_seeds'])} seed(s) with missing results.", name))
    if K >= 2 and panel and not panel["signal"]:
        warnings.append(_warning(
            "no_signal",
            f"The panel does not separate solutions on {name}: their difference is "
            "smaller than the noise, so no number of seeds ranks them.", name))


def _joint_precision(values: Mapping[str, np.ndarray], objectives: Sequence[str],
                     spec_of: Mapping[str, Any], seeds: Sequence[int], confidence: float,
                     gamma: float, absolute: Mapping[str, float]) -> Optional[dict[str, Any]]:
    """Seeds for every objective to hold at once, by Bonferroni over ``J``."""
    eligible = [o for o in objectives
                if resolve_aggregator(spec_of.get(o, "mean"))[0] == "mean"
                and metric_status(values[o]) == "ok"]
    if not eligible:
        return None
    joint = 1 - (1 - confidence) / len(eligible)
    worst: list[float] = []
    for objective in eligible:
        needed = []
        for row in values[objective]:
            y = _observed(row)
            if y.size < 2:
                continue
            half = absolute.get(objective, relative_half_width(float(y.mean()), gamma))
            needed.append(n_star(float(y.std(ddof=1)), half, joint))
        present = [v for v in needed if v is not None]
        if present:
            worst.append(float(np.percentile(present, 50)))
    if not worst:
        return None
    return {"value": math.ceil(max(worst)), "confidence": joint, "objectives": eligible}


def metric_detail(y_by_seed: Sequence[Optional[float]], *, seeds: Sequence[int],
                  simulation_ids: Sequence[Optional[str]], genome_hash: str, metric: str,
                  spec: "str | dict[str, Any]" = "mean",
                  targets: Optional[Mapping[str, float]] = None,
                  absolute: Optional[float] = None) -> dict[str, Any]:
    """One metric on one chromosome, with everything its charts need.

    Kept apart from :func:`analyse_study` because its resampling is what costs:
    a study has a few of these open at a time, not one per metric and chromosome.
    """
    targets = {**DEFAULT_TARGETS, **(targets or {})}
    confidence = targets["confidence"]
    gamma = targets["relative_error"]
    order = [i for i, value in enumerate(y_by_seed) if value is not None]
    y = np.array([y_by_seed[i] for i in order], dtype=float)
    rng = study_rng(seeds, metric, genome_hash, "detail")
    out: dict[str, Any] = {
        "metric": metric, "genome_hash": genome_hash, "aggregator": spec,
        "summary": describe(y), "values": [], "histogram": [], "box": None, "qq": [],
        "running": [], "precision": None, "n_star": None,
        "intervals": {"mean_t": None, "mean_bca": None, "sd_bca": None, "median": None,
                      "p90_upper": None, "p95_upper": None, "p99_upper": None},
    }
    if y.size == 0:
        return out
    flags = outlier_flags(y)
    out["values"] = [
        {"seed": seeds[i], "order": i, "value": float(value),
         "simulation_id": simulation_ids[i], "flag": flag}
        for i, value, flag in zip(order, y, flags)]
    out["histogram"] = histogram(y)
    out["box"] = boxplot(y)
    out["qq"] = qq_points(y)
    out["running"] = running_interval(y, confidence)
    if y.size < 2:
        return out

    sd_interval = bca_interval(y, "sd", confidence, rng)
    out["intervals"] = {
        "mean_t": ci_mean_t(y, confidence),
        "mean_bca": bca_interval(y, "mean", confidence, rng),
        "sd_bca": sd_interval,
        "median": median_interval(y, confidence),
        "p90_upper": quantile_upper_limit(y, 0.90, confidence),
        "p95_upper": quantile_upper_limit(y, 0.95, confidence),
        "p99_upper": quantile_upper_limit(y, 0.99, confidence),
    }
    precision = _precision_of(y, spec=spec, confidence=confidence, gamma=gamma,
                              absolute=absolute, planned=len(seeds), rng=None)
    tolerance = _tolerance(y, spec, gamma, absolute)
    if precision["value"] is not None and sd_interval:
        precision["range"] = [n_star(sd_interval[0], tolerance, confidence),
                              n_star(sd_interval[1], tolerance, confidence)]
    if y.size >= 3:
        curve = precision_curve(y, spec, tolerance, confidence, rng, RESAMPLES_DETAIL)
        out["precision"] = curve
        precision["empirical"] = curve["n_emp"]
        if precision["method"] == "empirical":
            precision["value"] = curve["n_emp"]
    out["n_star"] = precision
    return out


def compare_control(main: Mapping[str, Mapping[str, Any]],
                    control: Mapping[str, Sequence[Mapping[str, Any]]],
                    metrics: Sequence[str]) -> dict[str, Any]:
    """Whether repeating a seed reproduced every number.

    Everything else in a study rests on this: if another source of noise
    exists, the distribution measured is not the distribution of the seed.
    ``main`` holds a chromosome's result on the repeated seed, ``control`` the
    repetitions of it.
    """
    compared = 0
    divergent: list[dict[str, Any]] = []
    replicates = 0
    for genome_hash, runs in control.items():
        reference = main.get(genome_hash) or {}
        replicates = max(replicates, len(runs))
        for metric in metrics:
            numbers = [float(run[metric]) for run in (reference, *runs) if metric in run]
            numbers = [v for v in numbers if math.isfinite(v)]
            if len(numbers) < 2:
                continue
            compared += 1
            if any(not math.isclose(v, numbers[0], rel_tol=1e-12) for v in numbers[1:]):
                divergent.append({"genome_hash": genome_hash, "metric": metric,
                                  "values": numbers})
    return {"checked": compared > 0, "identical": compared > 0 and not divergent,
            "replicates": replicates, "compared": compared, "divergent": divergent[:50],
            "divergent_total": len(divergent)}


def failure_kind(message: Optional[str]) -> str:
    """Whether a failed run says something about the simulation or the pool.

    A failure of the infrastructure is independent of the seed. A run that
    keeps failing on its own is censoring that may depend on it, and the
    distribution is then conditional on the seeds that finished.
    """
    text = (message or "").lower()
    return "infrastructure" if any(mark in text for mark in _INFRASTRUCTURE) else "execution"


def failure_summary(attempts: Sequence[Mapping[str, Any]], confidence: float) -> dict[str, Any]:
    """The failure rate of a study's runs, from every attempt recorded."""
    finished = [a for a in attempts if a.get("status") in ("Done", "Error")]
    failed = [a for a in finished if a.get("status") == "Error"]
    by_kind = {"infrastructure": 0, "execution": 0}
    for attempt in failed:
        by_kind[failure_kind(attempt.get("system_message"))] += 1
    return {"attempts": len(finished), "failed": len(failed), "by_kind": by_kind,
            "rate_ci": clopper_pearson(len(failed), len(finished), confidence)}
