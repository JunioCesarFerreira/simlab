"""Unit tests for pylib.batch_input — what a batch must bring."""
import copy

import pytest

from pylib.batch_input import validate_batch_input

PROBLEMS = {
    "problem0": ({"n": 2}, {"x": [0.0, 1]}),
    "problem1": ({"number_of_relays": 2},
                 {"mac_protocol": 0, "relays": [{"x": -1.5, "y": 2}, {"x": 3, "y": 4.25}]}),
    "problem2": ({"candidates": [[0, 0], [1, 1]]}, {"mac_protocol": 1, "mask": [0, 1]}),
    "problem3": ({"candidates": [[0, 0], [1, 1]]}, {"mac_protocol": 0, "mask": [1, 1]}),
    "problem4": ({"sojourns": [{"id": 0}, {"id": 1}]},
                 {"mac_protocol": 0, "route": [0, 1, 0], "sojourn_times": [0, 2.5, 1]}),
}
ONE = {"csma": "a"}
BOTH = {"csma": "a", "tsch": "b"}


def parameters(name="problem2", *chromosomes, **more):
    problem, chromosome = PROBLEMS[name]
    return {
        "strategy": "batch",
        "problem": {"name": name, **copy.deepcopy(problem)},
        "simulation": {},
        "chromosomes": list(chromosomes) or [copy.deepcopy(chromosome)],
        **more,
    }


def refused(params, field, sources=ONE):
    with pytest.raises(ValueError) as error:
        validate_batch_input(params, sources)
    assert str(error.value).startswith(field + " "), str(error.value)


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_each_problem_accepts_its_own_shape(name):
    validate_batch_input(parameters(name), BOTH)


def test_p0_takes_either_name_for_its_size():
    params = parameters("problem0")
    params["problem"] = {"name": "problem0", "number_of_variables": 2}
    validate_batch_input(params, ONE)


def test_other_keys_of_a_chromosome_are_left_alone():
    validate_batch_input(parameters("problem2", {"mac_protocol": 0, "mask": [1, 0], "label": "A"}), ONE)


@pytest.mark.parametrize("value", [None, [], {}, "x", 3])
def test_a_batch_needs_chromosomes(value):
    refused(parameters(chromosomes=value), "parameters.chromosomes")


@pytest.mark.parametrize("value", [None, [], "x", 3])
def test_a_chromosome_is_an_object(value):
    refused(parameters("problem2", value), "parameters.chromosomes[0]")


NOT_WHOLE = [None, True, 1.0, "1"]
NOT_NUMBERS = [None, True, "1", float("nan"), float("inf"), 10 ** 400]


@pytest.mark.parametrize("name", ["problem1", "problem2", "problem3", "problem4"])
@pytest.mark.parametrize("value", [*NOT_WHOLE, 2, -1, [0]])
def test_the_mac_gene_is_0_or_1(name, value):
    params = parameters(name)
    params["chromosomes"][0]["mac_protocol"] = value
    refused(params, "parameters.chromosomes[0].mac_protocol")


@pytest.mark.parametrize("value", [None, [], [0.5], [0, 0, 0], [0, 1.01], [-0.01, 0],
                                   *[[0, v] for v in NOT_NUMBERS]])
def test_p0_x_has_n_values_in_the_unit_interval(value):
    refused(parameters("problem0", {"x": value}), "parameters.chromosomes[0].x")


@pytest.mark.parametrize("value", [None, [], [{"x": 0, "y": 0}], [{"x": 0, "y": 0}] * 3,
                                   [{"x": 0}, {"x": 0, "y": 0}], [[0, 0], [1, 1]],
                                   *[[{"x": 0, "y": v}] * 2 for v in NOT_NUMBERS]])
def test_p1_has_one_position_per_relay(value):
    refused(parameters("problem1", {"mac_protocol": 0, "relays": value}),
            "parameters.chromosomes[0].relays")


@pytest.mark.parametrize("name", ["problem2", "problem3"])
@pytest.mark.parametrize("value", [None, [], [1], [1, 0, 1], [1, 2], [1, -1], "10",
                                   *[[1, v] for v in NOT_WHOLE]])
def test_a_mask_has_one_bit_per_candidate(name, value):
    refused(parameters(name, {"mac_protocol": 0, "mask": value}),
            "parameters.chromosomes[0].mask")


@pytest.mark.parametrize("value", [None, [], [0, 2], [-1, 0], *[[0, v] for v in NOT_WHOLE]])
def test_p4_route_visits_known_sojourns(value):
    refused(parameters("problem4", {"mac_protocol": 0, "route": value, "sojourn_times": [1, 1]}),
            "parameters.chromosomes[0].route")


@pytest.mark.parametrize("value", [None, [], [1], [1, 1, 1], [1, -1], *[[1, v] for v in NOT_NUMBERS]])
def test_p4_has_one_time_per_stop(value):
    refused(parameters("problem4", {"mac_protocol": 0, "route": [0, 1], "sojourn_times": value}),
            "parameters.chromosomes[0].sojourn_times")


@pytest.mark.parametrize("name, key, value, field", [
    ("problem0", "n", None, "parameters.problem.n"),
    ("problem0", "n", 0, "parameters.problem.n"),
    ("problem1", "number_of_relays", None, "parameters.problem.number_of_relays"),
    ("problem1", "number_of_relays", 0, "parameters.problem.number_of_relays"),
    ("problem2", "candidates", None, "parameters.problem.candidates"),
    ("problem3", "candidates", [], "parameters.problem.candidates"),
    ("problem4", "sojourns", None, "parameters.problem.sojourns"),
    ("problem2", "name", "problem9", "parameters.problem.name"),
    ("problem2", "name", ["problem2"], "parameters.problem.name"),
])
def test_the_problem_must_give_the_size(name, key, value, field):
    params = parameters(name)
    params["problem"][key] = value
    refused(params, field)


@pytest.mark.parametrize("value", [*NOT_WHOLE, -1, 2 ** 31])
def test_the_seed_is_a_whole_number_in_range(value):
    refused(parameters(random_seed=value), "parameters.random_seed")


@pytest.mark.parametrize("value", [0, 42, 2 ** 31 - 1])
def test_a_seed_in_range_is_accepted(value):
    validate_batch_input(parameters(random_seed=value), ONE)


class TestSources:
    """The rule of ``get_source_by_mac_protocol`` in the mo-engine."""

    @pytest.mark.parametrize("gene", [0, 1])
    @pytest.mark.parametrize("sources", [{"csma": "a"}, {"tsch": "a"}, {"csma": "a", "tsch": ""}])
    def test_a_single_source_takes_either_gene(self, gene, sources):
        validate_batch_input(parameters("problem2", {"mac_protocol": gene, "mask": [1, 0]}), sources)

    @pytest.mark.parametrize("sources", [None, {}, {"csma": ""}])
    def test_without_a_source_nothing_runs(self, sources):
        refused(parameters(), "source_repository_options", sources)

    def test_with_several_the_gene_must_find_its_own(self):
        refused(parameters("problem2", {"mac_protocol": 1, "mask": [1, 0]}),
                "parameters.chromosomes[0].mac_protocol", {"csma": "a", "csma-v2": "b"})

    def test_synthetic_needs_none(self):
        params = parameters(simulation={"synthetic": {"enabled": True}})
        validate_batch_input(params, {})
