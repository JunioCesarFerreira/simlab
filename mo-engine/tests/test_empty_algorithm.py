"""``parameters.algorithm: {}`` runs on the defaults.

``build_adapter`` handed the generator over only with a non-empty
configuration, so every loop died at start on ``_rng``.
"""
from __future__ import annotations

import logging
import random
from unittest.mock import MagicMock

import pytest
from bson import ObjectId

from lib.problem.resolve import build_adapter
from lib.strategy.nsga2 import NSGA2LoopStrategy
from lib.strategy.nsga3 import NSGA3LoopStrategy
from lib.strategy.random_search import RandomSearchStrategy
from tests.resume_harness import PROBLEMS


@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_the_adapter_gets_the_generator_without_settings(problem):
    adapter = build_adapter(PROBLEMS[problem], {}, random.Random(1))

    parents = adapter.random_individual_generator(2)

    assert len(adapter.crossover(parents)) == 2
    assert adapter.mutate(parents[0]) is not None


@pytest.mark.parametrize("problem", list(PROBLEMS))
def test_empty_is_the_same_as_the_defaults(problem):
    empty = build_adapter(PROBLEMS[problem], {}, random.Random(5))
    other = build_adapter(PROBLEMS[problem], {"population_size": 20}, random.Random(5))

    assert empty.random_individual_generator(4) == other.random_individual_generator(4)


def test_no_settings_are_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        build_adapter(PROBLEMS["analytical"], {}, random.Random(1))

    assert not caplog.records


@pytest.mark.parametrize("cls", [NSGA2LoopStrategy, NSGA3LoopStrategy, RandomSearchStrategy],
                         ids=lambda cls: cls.__name__)
def test_a_strategy_starts_with_an_empty_algorithm(cls):
    mongo = MagicMock()
    mongo.generation_repo.find_by_experiment.return_value = []
    mongo.genome_cache_repo.get_all_by_experiment.return_value = []
    strategy = cls({
        "_id": ObjectId(),
        "parameters": {
            "problem": PROBLEMS["analytical"],
            "algorithm": {},
            "simulation": {"synthetic": {"enabled": True}, "random_seeds": [1]},
            "objectives": [{"metric_name": "a", "goal": "min"},
                           {"metric_name": "b", "goal": "min"}],
        },
    }, mongo)
    for name in ("_start_watcher", "_start_generation_poll", "_reconcile_current_generation",
                 "_upload_topology_async", "_fire_generation_done"):
        setattr(strategy, name, lambda *_a, **_kw: None)

    strategy.start()

    assert len(strategy._current_population) == 20
