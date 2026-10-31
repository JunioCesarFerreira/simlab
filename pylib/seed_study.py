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
