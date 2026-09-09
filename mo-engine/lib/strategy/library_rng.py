"""Reproducible seeding for the optional DEAP / pymoo selection backends.

Neither library accepts SimLab's ``random.Random``, and each escapes it a
different way:

* DEAP's NSGA-III niching shuffles through the process-wide ``numpy.random``
  module. There is no argument to pass — the global stream is the only way in.
* pymoo's survival operators build ``np.random.default_rng(None)`` whenever no
  ``random_state`` is given, which draws from OS entropy. ``np.random.seed``
  does not reach it at all.

Both are driven from the experiment's ``algorithm.random_seed``, through the
same ``random.Random`` the rest of the loop already uses. That is deliberate:
every library seed is drawn from that one generator, so the reproducibility of
a whole run reduces to a single piece of state — which is also the only thing a
checkpoint has to persist.
"""
from __future__ import annotations

import contextlib
import random

import numpy as np

# numpy.random.seed accepts a 32-bit value.
_SEED_SPACE = 2 ** 32


def derive_seed(rng: random.Random) -> int:
    """Draw the next library seed from SimLab's own generator."""
    return rng.randrange(_SEED_SPACE)


def derive_generator(rng: random.Random) -> "np.random.Generator":
    """A fresh numpy Generator for one pymoo call, seeded from *rng*.

    Deliberately per-call rather than one long-lived generator: a Generator's
    state would then be a second thing to checkpoint, and getting it out of step
    with ``rng`` would silently break resume reproducibility.
    """
    return np.random.default_rng(derive_seed(rng))


@contextlib.contextmanager
def numpy_global_seed(rng: random.Random):
    """Seed the process-wide numpy RNG for a DEAP call, then put it back.

    Restoring the previous state on exit keeps this out of anything else running
    in the process — the engine shares its interpreter with other work.
    """
    state = np.random.get_state()
    np.random.seed(derive_seed(rng))
    try:
        yield
    finally:
        np.random.set_state(state)


def dump_random_state(rng: random.Random) -> dict:
    """A BSON-safe snapshot of *rng*, for a checkpoint.

    ``getstate`` returns ``(version, tuple_of_ints, gauss_next)``; Mongo stores
    lists rather than tuples. Because every library seed is derived from this
    one generator, this snapshot is the whole random state of a run.
    """
    version, internal, gauss_next = rng.getstate()
    return {
        "version": int(version),
        "internal": [int(value) for value in internal],
        "gauss_next": gauss_next,
    }


def load_random_state(rng: random.Random, snapshot: dict | None) -> bool:
    """Restore *rng* from a :func:`dump_random_state` snapshot.

    Returns False — leaving *rng* untouched — for a checkpoint written before
    the field existed, or one that cannot be read back. The caller decides what
    to say about it; a run that resumes on a fresh stream is still valid, it
    just no longer reproduces the uninterrupted one.
    """
    if not snapshot:
        return False
    try:
        rng.setstate(
            (
                int(snapshot["version"]),
                tuple(int(value) for value in snapshot["internal"]),
                snapshot.get("gauss_next"),
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    return True
