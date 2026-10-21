"""The scalability benchmark strategy, against a strict in-memory repository.

The real problem adapter is used — a P1 problem — because the workload has
to be a network the adapter would actually simulate. Only the database and
GridFS are stood in for, and the stand-ins record every write, because what
the strategy owes the master-node is a precise sequence of documents.
"""
from contextlib import contextmanager
from typing import Any

import pytest
from bson import ObjectId

from pylib.db.models.enums import EnumStatus
from lib.strategy import usl_benchmark as module
from lib.strategy.usl_benchmark import (
    UslBenchmarkStrategy,
    build_waves,
    choose_workload,
)

EXP_ID = ObjectId("507f1f77bcf86cd799460001")
CSMA = ObjectId("507f1f77bcf86cd799460011")
TSCH = ObjectId("507f1f77bcf86cd799460012")
OTHER_EXP = ObjectId("507f1f77bcf86cd799460099")

PROBLEM = {
    "name": "problem1",
    "region": [-100.0, -100.0, 100.0, 100.0],
    "sink": (0.0, 0.0),
    "mobile_nodes": [{
        "path_segments": [("0 + 10*t", "0")],
        "is_closed": False, "is_round_trip": False,
        "speed": 1.0, "time_step": 1.0,
    }],
    "number_of_relays": 3,
    "radius_of_reach": 80.0,
    "radius_of_inter": 120.0,
    "min_coverage_percentage": 0.0,
}


class Store:
    """Just enough of MongoRepository, and strict about it."""

    def __init__(self):
        self.experiments: dict[str, dict] = {}
        self.generations: list[dict] = []
        self.individuals: list[dict] = []
        self.simulations: list[dict] = []
        outer = self

        class Generations:
            def insert(self, gen):
                outer.generations.append(dict(gen))
                return gen["_id"]

            def find_by_experiment(self, exp_id):
                return [g for g in outer.generations if g["experiment_id"] == exp_id]

            def get(self, gen_id):
                return next((g for g in outer.generations
                             if str(g["_id"]) == str(gen_id)), None)

            def watch_status_terminal(self, callback):
                return None

        class Individuals:
            def insert(self, ind):
                outer.individuals.append(dict(ind))
                return ObjectId()

        class Collection:
            def count_documents(self, flt):
                return sum(1 for s in outer.simulations
                           if s["generation_id"] == flt["generation_id"]
                           and s["status"] == flt["status"])

        class Db(dict):
            def __getitem__(self, name):
                return Collection()

        class Connection:
            @contextmanager
            def connect(self):
                yield Db()

        class Simulations:
            connection = Connection()

            def insert(self, sim):
                doc = dict(sim)
                doc["_id"] = ObjectId()
                outer.simulations.append(doc)
                return doc["_id"]

        class Experiments:
            def get(self, exp_id):
                return outer.experiments.get(str(exp_id))

            def update(self, exp_id, fields):
                outer.experiments.setdefault(str(exp_id), {}).update(fields)
                return True

        self.generation_repo = Generations()
        self.individual_repo = Individuals()
        self.simulation_repo = Simulations()
        self.experiment_repo = Experiments()
        self.fs_handler = object()

    # helpers for the tests -------------------------------------------------
    def wave_simulations(self, gen_id):
        return [s for s in self.simulations if s["generation_id"] == gen_id]

    def close(self, gen_id, sim_status=EnumStatus.DONE):
        """What the master-node does when a wave's simulations finish."""
        for sim in self.wave_simulations(gen_id):
            sim["status"] = sim_status
        gen = self.generation_repo.get(gen_id)
        gen["status"] = (EnumStatus.DONE if sim_status == EnumStatus.DONE
                         else EnumStatus.ERROR)
        return {"fullDocument": {"_id": gen["_id"],
                                 "experiment_id": gen["experiment_id"],
                                 "status": gen["status"]}}


def experiment(levels=(1, 2, 4), repetitions=2, **benchmark) -> dict:
    return {
        "_id": EXP_ID,
        "parameters": {
            "strategy": "usl_benchmark",
            "random_seed": 7,
            "problem": PROBLEM,
            "simulation": {"duration": 60, "random_seeds": [4242]},
            "objectives": [{"metric_name": "latency", "goal": "min"}],
            "benchmark": {"levels": list(levels), "repetitions": repetitions,
                          **benchmark},
        },
        # The keys are protocol names, as the launch wizard writes them.
        "source_repository_options": {"csma": str(CSMA), "tsch": str(TSCH)},
    }


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(module, "create_files", lambda config, fs: {
        "csc_file_id": ObjectId(), "pos_file_id": ObjectId()})
    # No threads: the tests deliver generation events themselves, one at a
    # time, which is the only way to assert the order of what follows.
    monkeypatch.setattr(UslBenchmarkStrategy, "_start_watchers", lambda self: None)
    return Store()


def started(store, **kwargs) -> UslBenchmarkStrategy:
    strategy = UslBenchmarkStrategy(experiment(**kwargs), store)
    strategy.start()
    return strategy


# ── the plan ────────────────────────────────────────────────────────────────

class TestTheWaves:
    def test_levels_are_sorted_and_deduplicated(self):
        assert build_waves([4, 1, 2, 2], 1) == [(1, 0), (2, 0), (4, 0)]

    def test_repetitions_alternate_direction(self):
        """A drift in the machines lands on every level instead of biasing
        the small ones against the large ones."""
        assert build_waves([1, 2, 4], 3) == [
            (1, 0), (2, 0), (4, 0),
            (4, 1), (2, 1), (1, 1),
            (1, 2), (2, 2), (4, 2),
        ]

    def test_non_positive_levels_are_dropped(self):
        assert build_waves([0, -1, 3], 1) == [(3, 0)]

    def test_no_usable_level_is_refused(self):
        with pytest.raises(ValueError):
            build_waves([0], 1)

    def test_no_repetition_is_refused(self):
        with pytest.raises(ValueError):
            build_waves([1, 2], 0)


class TestChoosingTheWorkload:
    def test_the_first_feasible_draw_is_taken(self):
        class Adapter:
            draws = 0

            def random_individual_generator(self, size):
                Adapter.draws += 1
                return [f"c{Adapter.draws}"]

            def penalty_objectives(self, chromosome, n):
                return None if chromosome == "c3" else [1e9] * n

        chosen, attempts = choose_workload(Adapter(), None, 1)

        assert chosen == "c3" and attempts == 3

    def test_a_problem_with_no_feasible_draw_is_refused(self):
        """Fifty infeasible draws means the generator does not produce
        simulable networks; looping further would only hide that."""
        class Adapter:
            def random_individual_generator(self, size):
                return ["x"]

            def penalty_objectives(self, chromosome, n):
                return [1e9] * n

        with pytest.raises(ValueError, match="no feasible chromosome"):
            choose_workload(Adapter(), None, 1, attempts=5)


# ── running it ──────────────────────────────────────────────────────────────

class TestTheFirstWave:
    def test_the_experiment_starts_running_with_its_workload_recorded(self, store):
        started(store)

        exp = store.experiments[str(EXP_ID)]
        assert exp["status"] == EnumStatus.RUNNING
        workload = exp["benchmark_workload"]
        assert workload["source"] == "generated"
        assert workload["chromosome"] and workload["genome_hash"]
        assert workload["seed"] == 4242

    def test_it_is_one_generation_tagged_with_its_level(self, store):
        started(store)

        assert len(store.generations) == 1
        gen = store.generations[0]
        assert gen["benchmark"] == {"level": 1, "repetition": 0, "wave": 0}
        assert gen["status"] == EnumStatus.WAITING

    def test_it_runs_exactly_level_simulations(self, store):
        started(store, levels=(3, 5))

        gen = store.generations[0]
        assert gen["benchmark"]["level"] == 3
        assert len(store.wave_simulations(gen["_id"])) == 3

    def test_the_generation_is_written_before_its_simulations(self, store):
        """A fast worker can finish a simulation before the last one is
        inserted; the close has to land on a document that exists."""
        order: list[str] = []
        insert_gen = store.generation_repo.insert
        insert_sim = store.simulation_repo.insert
        store.generation_repo.insert = lambda g: order.append("gen") or insert_gen(g)
        store.simulation_repo.insert = lambda s: order.append("sim") or insert_sim(s)

        started(store, levels=(2,))

        assert order == ["gen", "sim", "sim"]


class TestTheWorkIsIdentical:
    """The only thing allowed to change between simulations is how many run
    at once. Different work would show up as different timing and be read as
    a property of the infrastructure."""

    def test_one_seed_and_one_network_for_every_simulation(self, store):
        strategy = started(store, levels=(1, 2), repetitions=1)
        store_event = store.close(store.generations[0]["_id"])
        strategy.event_generation_done(store_event)

        sims = store.simulations
        assert len(sims) == 3
        assert {s["random_seed"] for s in sims} == {4242}
        assert {s["individual_id"] for s in sims} == {
            store.experiments[str(EXP_ID)]["benchmark_workload"]["genome_hash"]}
        elements = [s["parameters"]["simulationElements"] for s in sims]
        assert all(e == elements[0] for e in elements)

    def test_every_wave_has_one_individual_of_the_same_chromosome(self, store):
        strategy = started(store, levels=(1, 2), repetitions=1)
        strategy.event_generation_done(store.close(store.generations[0]["_id"]))

        assert len(store.individuals) == 2
        assert store.individuals[0]["chromosome"] == store.individuals[1]["chromosome"]


class TestAdvancing:
    def test_a_closed_wave_enqueues_the_next(self, store):
        strategy = started(store, levels=(1, 2, 4), repetitions=1)

        strategy.event_generation_done(store.close(store.generations[0]["_id"]))

        assert [g["benchmark"]["level"] for g in store.generations] == [1, 2]

    def test_the_same_event_delivered_twice_advances_once(self, store):
        """The change stream, the polling fallback and the resume path may
        all deliver one generation."""
        strategy = started(store, levels=(1, 2, 4), repetitions=1)
        event = store.close(store.generations[0]["_id"])

        strategy.event_generation_done(event)
        strategy.event_generation_done(event)

        assert len(store.generations) == 2

    def test_another_experiments_generation_is_ignored(self, store):
        """The watcher sees every generation of every experiment."""
        strategy = started(store, levels=(1, 2), repetitions=1)
        foreign = {"fullDocument": {"_id": store.generations[0]["_id"],
                                    "experiment_id": OTHER_EXP,
                                    "status": EnumStatus.DONE}}

        strategy.event_generation_done(foreign)

        assert len(store.generations) == 1

    def test_an_old_waves_late_event_does_not_advance(self, store):
        strategy = started(store, levels=(1, 2, 4), repetitions=1)
        first = store.generations[0]["_id"]
        strategy.event_generation_done(store.close(first))
        store.generations[0]["status"] = EnumStatus.DONE

        strategy.event_generation_done({"fullDocument": {
            "_id": first, "experiment_id": EXP_ID, "status": EnumStatus.DONE}})

        assert len(store.generations) == 2

    def test_the_last_wave_completes_the_experiment(self, store):
        strategy = started(store, levels=(1, 2), repetitions=1)
        strategy.event_generation_done(store.close(store.generations[0]["_id"]))
        strategy.event_generation_done(store.close(store.generations[1]["_id"]))

        exp = store.experiments[str(EXP_ID)]
        assert exp["status"] == EnumStatus.DONE
        assert "2 waves" in exp["system_message"]
        assert len(store.generations) == 2

    def test_a_wave_with_a_failure_is_not_a_reason_to_stop(self, store):
        """One failure is a measurement that did not happen; the analysis
        reports the wave as invalid and the benchmark carries on."""
        strategy = started(store, levels=(2, 4), repetitions=1)
        gen = store.generations[0]["_id"]
        sims = store.wave_simulations(gen)
        sims[0]["status"] = EnumStatus.DONE
        sims[1]["status"] = EnumStatus.ERROR
        store.generation_repo.get(gen)["status"] = EnumStatus.ERROR

        strategy.event_generation_done({"fullDocument": {
            "_id": gen, "experiment_id": EXP_ID, "status": EnumStatus.ERROR}})

        assert len(store.generations) == 2

    def test_a_wave_with_no_result_at_all_stops_it(self, store):
        """Nothing produced a result: the workload does not run here, and
        every further wave would only measure failures."""
        strategy = started(store, levels=(1, 2, 4), repetitions=1)

        strategy.event_generation_done(
            store.close(store.generations[0]["_id"], EnumStatus.ERROR))

        exp = store.experiments[str(EXP_ID)]
        assert exp["status"] == EnumStatus.ERROR
        assert "none produced a result" in exp["system_message"]
        assert len(store.generations) == 1


class TestResuming:
    def test_a_restart_keeps_the_recorded_workload(self, store):
        first = started(store, levels=(1, 2), repetitions=1)
        recorded = store.experiments[str(EXP_ID)]["benchmark_workload"]

        resumed = UslBenchmarkStrategy(experiment(levels=(1, 2), repetitions=1), store)
        resumed.start()

        assert resumed._workload.get_hash() == recorded["genome_hash"]
        assert len(store.generations) == 1, "nothing new until the wave closes"
        del first

    def test_a_restart_after_a_wave_closed_advances(self, store):
        started(store, levels=(1, 2), repetitions=1)
        store.close(store.generations[0]["_id"])

        UslBenchmarkStrategy(experiment(levels=(1, 2), repetitions=1), store).start()

        assert [g["benchmark"]["level"] for g in store.generations] == [1, 2]

    def test_waves_without_a_recorded_workload_are_refused(self, store):
        """Choosing a new chromosome now would mix two measurements in one
        benchmark."""
        store.generations.append({
            "_id": ObjectId(), "experiment_id": EXP_ID, "index": 0,
            "status": EnumStatus.RUNNING,
            "benchmark": {"level": 1, "repetition": 0, "wave": 0}})

        UslBenchmarkStrategy(experiment(), store).start()

        exp = store.experiments[str(EXP_ID)]
        assert exp["status"] == EnumStatus.ERROR
        assert "mix two measurements" in exp["system_message"]


class TestAProvidedChromosome:
    def test_it_is_used_instead_of_a_random_one(self, store):
        given = {"mac_protocol": 0, "relays": [{"x": 10.0, "y": 0.0},
                                               {"x": 20.0, "y": 0.0},
                                               {"x": 30.0, "y": 0.0}]}

        started(store, chromosome=given)

        workload = store.experiments[str(EXP_ID)]["benchmark_workload"]
        assert workload["source"] == "provided"
        assert workload["chromosome"]["relays"][0] in ({"x": 10.0, "y": 0.0}, [10.0, 0.0])


class TestTheRecordedWorkloadIsTheOneThatRuns:
    """With a single firmware available, resolving the source *rewrites* the
    chromosome's MAC gene to match it. Resolving per simulation and keeping
    the original recorded one network and ran another: the workload, every
    ``individual_id`` and the protocol simulated must be the same thing."""

    def test_with_one_firmware_everything_agrees(self, store):
        # CSMA requested, only TSCH available: the rewrite is guaranteed. A
        # randomly drawn chromosome would only exercise it when the seed
        # happened to draw the other protocol — which is how an earlier
        # version of this test passed against the defect.
        exp = experiment(levels=(2,), repetitions=1, chromosome={
            "mac_protocol": 0,
            "relays": [{"x": 10.0, "y": 0.0}, {"x": 20.0, "y": 0.0},
                       {"x": 30.0, "y": 0.0}]})
        exp["source_repository_options"] = {"tsch": str(TSCH)}

        UslBenchmarkStrategy(exp, store).start()

        workload = store.experiments[str(EXP_ID)]["benchmark_workload"]
        assert workload["chromosome"]["mac_protocol"] == 1, "rewritten to TSCH"
        assert {s["individual_id"] for s in store.simulations} == {
            workload["genome_hash"]}
        assert {s["source_repository_id"] for s in store.simulations} == {TSCH}
        assert store.individuals[0]["individual_id"] == workload["genome_hash"]
