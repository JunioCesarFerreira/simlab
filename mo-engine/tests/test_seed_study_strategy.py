"""The seed study strategy, against a strict in-memory repository.

The real P1 adapter is used, as in the scalability benchmark: the panel has to
be made of networks the adapter would simulate. Only the database stands in,
and it records every write, because what the strategy owes the master-node is
a precise sequence of documents.
"""
from contextlib import contextmanager

import pytest
from bson import ObjectId

from pylib.db.models.enums import EnumStatus
from pylib.db.repositories.generation import _completeness_of
from lib.strategy import batch as batch_module
from lib.strategy.batch import BatchStrategy
from lib.strategy.seed_study import SeedStudyStrategy, choose_panel

EXP_ID = ObjectId("507f1f77bcf86cd799460001")
CSMA = ObjectId("507f1f77bcf86cd799460011")
TSCH = ObjectId("507f1f77bcf86cd799460012")

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

SEEDS = [336157, 667370, 35239]


class Store:
    """Just enough of MongoRepository, and strict about it."""

    def __init__(self):
        self.experiments: dict[str, dict] = {str(EXP_ID): {"_id": EXP_ID, "status": "Running"}}
        self.generations: list[dict] = []
        self.individuals: list[dict] = []
        self.simulations: list[dict] = []
        self.cancelled = False
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

            def record_plan(self, gen_id, seeds):
                self.get(gen_id)["expected_seeds"] = list(seeds)

            def pending_retry_round(self, gen_id):
                return None

            def any_simulation_active(self, gen_id):
                return any(s["generation_id"] == gen_id
                           and s["status"] in (EnumStatus.WAITING, EnumStatus.RUNNING)
                           for s in outer.simulations)

            def replica_completeness(self, gen_id, keys):
                gen = self.get(gen_id)
                grouped: dict[str, list] = {}
                for sim in outer.simulations:
                    if sim["generation_id"] == gen_id:
                        grouped.setdefault(sim["individual_id"], []).append(sim)
                return {key: _completeness_of(runs, keys, gen.get("expected_seeds"))
                        for key, runs in grouped.items()}

            def retries_used(self, gen_id):
                return int(self.get(gen_id).get("replica_retries", 0))

            def claim_retry_round(self, gen_id, max_retries, replicas):
                gen = self.get(gen_id)
                gen["replica_retries"] = gen.get("replica_retries", 0) + 1
                return gen["replica_retries"]

            def finish_retry_round(self, gen_id, round_number):
                self.get(gen_id)["status"] = EnumStatus.RUNNING
                return True

            def close_if_finished(self, gen_id):
                return None

            def get_simulations_metrics_by_individual(self, generation_id, metrics, aggregators):
                out: dict[str, dict] = {}
                for sim in outer.simulations:
                    if (sim["generation_id"] == generation_id
                            and sim["status"] == EnumStatus.DONE):
                        out.setdefault(sim["individual_id"], dict(sim["network_metrics"]))
                return out

            def watch_status_terminal(self, callback):
                return None

        class Individuals:
            def insert(self, ind):
                outer.individuals.append(dict(ind))
                return ObjectId()

            def update_evaluation(self, individual_id, generation_id, objectives, record):
                for ind in outer.individuals:
                    if (ind["individual_id"] == individual_id
                            and ind["generation_id"] == generation_id):
                        ind["objectives"] = objectives
                        ind["completeness"] = record

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
                doc["parameters"] = dict(sim["parameters"])
                doc["_id"] = ObjectId()
                outer.simulations.append(doc)
                return doc["_id"]

            def find_terminal_by_generation(self, gen_id):
                return [s for s in outer.simulations if s["generation_id"] == gen_id
                        and s["status"] == EnumStatus.ERROR]

            def clone_for_retry(self, simulation_id, reason, retry_round):
                source = next(s for s in outer.simulations if s["_id"] == simulation_id)
                clone = {**source, "_id": ObjectId(), "status": EnumStatus.WAITING}
                outer.simulations.append(clone)
                return clone["_id"]

        class Experiments:
            def get(self, exp_id):
                return outer.experiments.get(str(exp_id))

            def update(self, exp_id, fields):
                outer.experiments.setdefault(str(exp_id), {}).update(fields)
                return True

            def is_cancelled(self, exp_id):
                return outer.cancelled

        self.generation_repo = Generations()
        self.individual_repo = Individuals()
        self.simulation_repo = Simulations()
        self.experiment_repo = Experiments()
        self.fs_handler = object()

    # helpers for the tests -------------------------------------------------
    @property
    def experiment(self) -> dict:
        return self.experiments[str(EXP_ID)]

    def of(self, gen_id) -> list[dict]:
        return [s for s in self.simulations if s["generation_id"] == gen_id]

    def close(self, gen_id, failed=()):
        """What the master-node does when a generation's simulations finish.

        ``failed`` lists the seeds that end in error.
        """
        for sim in self.of(gen_id):
            if sim["status"] in (EnumStatus.WAITING, EnumStatus.RUNNING):
                bad = sim["random_seed"] in failed
                sim["status"] = EnumStatus.ERROR if bad else EnumStatus.DONE
                if not bad:
                    sim["network_metrics"] = {"latency": 200.0, "energy": 5.0}
        gen = self.generation_repo.get(gen_id)
        gen["status"] = EnumStatus.DONE
        return {"fullDocument": {"_id": gen["_id"], "status": gen["status"]}}


def experiment(*, size=3, seeds=SEEDS, firmware=None, chromosomes=None, control=2,
               simulation=None) -> dict:
    return {
        "_id": EXP_ID,
        "parameters": {
            "strategy": "seed_study",
            "random_seed": 7741,
            "problem": PROBLEM,
            "simulation": {"duration": 60, "random_seeds": list(seeds),
                           "incomplete_objective": "accept", **(simulation or {})},
            "objectives": [{"metric_name": "latency", "goal": "min"},
                           {"metric_name": "energy", "goal": "min"}],
            "seed_study": {"panel": {"source": "generated", "size": size, "random_seed": 7741},
                           "chromosomes": chromosomes, "control_replicates": control},
        },
        "source_repository_options": firmware or {"csma": str(CSMA)},
    }


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(batch_module, "create_files", lambda config, fs: {
        "csc_file_id": ObjectId(), "pos_file_id": ObjectId()})
    # No threads: the tests deliver generation events themselves, one at a
    # time, which is the only way to assert the order of what follows.
    monkeypatch.setattr(BatchStrategy, "_start_watcher", lambda self: None)
    monkeypatch.setattr(BatchStrategy, "_start_generation_poll", lambda self: None)
    monkeypatch.setattr(BatchStrategy, "_upload_topology_async", lambda self, *args: None)
    return Store()


def started(store, **kwargs) -> SeedStudyStrategy:
    strategy = SeedStudyStrategy(experiment(**kwargs), store)
    strategy.start()
    return strategy


def finish_main(strategy, store, **kwargs):
    strategy.event_generation_done(store.close(store.generations[0]["_id"], **kwargs))


class TestChoosingThePanel:
    class Adapter:
        """Draws from a fixed list, so distinctness can be asserted."""

        def __init__(self, draws, infeasible=()):
            self.draws = list(draws)
            self.infeasible = set(infeasible)

        def random_individual_generator(self, size):
            return [self.draws.pop(0)]

        def penalty_objectives(self, chromosome, n):
            return [1e9] * n if chromosome.name in self.infeasible else None

    class Chromosome:
        def __init__(self, name):
            self.name = name

        def get_hash(self):
            return self.name

    def _chromosomes(self, *names):
        return [self.Chromosome(name) for name in names]

    def test_infeasible_draws_are_skipped_and_counted(self):
        adapter = self.Adapter(self._chromosomes("a", "x", "b"), infeasible={"x"})

        panel, draws = choose_panel(adapter, 1, 2, lambda c: c)

        assert [c.name for c in panel] == ["a", "b"] and draws == 3

    def test_chromosomes_are_distinct_after_resolving(self):
        """A single firmware rewrites the MAC gene, so two draws can become one."""
        merged = self.Chromosome("same")
        adapter = self.Adapter(self._chromosomes("a", "b", "c"))
        resolve = lambda c: merged if c.name in ("a", "b") else c

        panel, _ = choose_panel(adapter, 1, 2, resolve)

        assert [c.name for c in panel] == ["same", "c"]

    def test_a_space_too_small_stops_and_says_how_far_it_got(self):
        adapter = self.Adapter(self._chromosomes(*["a"] * 10))

        with pytest.raises(ValueError, match="only 1 distinct"):
            choose_panel(adapter, 1, 2, lambda c: c, attempts=5)


class TestRunningTheStudy:
    def test_every_chromosome_runs_every_seed(self, store):
        started(store, size=3)

        assert len(store.simulations) == 9
        assert {s["random_seed"] for s in store.simulations} == set(SEEDS)
        assert len({s["individual_id"] for s in store.simulations}) == 3

    def test_the_plan_is_recorded_for_the_replica_gate(self, store):
        started(store)

        assert store.generations[0]["expected_seeds"] == SEEDS

    def test_the_panel_is_written_on_the_experiment(self, store):
        started(store, size=4)

        panel = store.experiment["seed_study_panel"]
        assert panel["size"] == 4 and panel["source"] == "generated"
        assert panel["random_seed"] == 7741
        assert len(panel["chromosomes"]) == 4
        assert panel["genome_hashes"] == [i["individual_id"] for i in store.individuals]

    def test_what_is_recorded_is_what_runs(self, store):
        """With csma alone, every gene reads csma: the defect the other strategies
        have, where a chromosome records a protocol that did not run."""
        started(store, size=8)

        assert {i["chromosome"]["mac_protocol"] for i in store.individuals} == {0}
        assert {s["source_repository_id"] for s in store.simulations} == {CSMA}

    def test_a_single_firmware_for_tsch_rewrites_to_tsch(self, store):
        started(store, size=8, firmware={"tsch": str(TSCH)})

        assert {i["chromosome"]["mac_protocol"] for i in store.individuals} == {1}
        assert {s["source_repository_id"] for s in store.simulations} == {TSCH}

    def test_the_same_panel_seed_draws_the_same_panel(self, store):
        started(store, size=3)
        first = store.experiment["seed_study_panel"]["genome_hashes"]

        again = Store()
        started(again, size=3)

        assert again.experiment["seed_study_panel"]["genome_hashes"] == first

    def test_a_provided_chromosome_is_what_runs(self, store):
        given = [{"mac_protocol": 1, "relays": [{"x": 1.0, "y": 2.0}] * 3}]

        started(store, chromosomes=given)

        assert store.experiment["seed_study_panel"]["source"] == "provided"
        assert len(store.individuals) == 1
        assert store.individuals[0]["chromosome"]["mac_protocol"] == 0, "resolved to csma"

    def test_a_provided_duplicate_is_refused(self, store):
        """Different genes, the same network once csma is the only firmware."""
        a = {"mac_protocol": 0, "relays": [{"x": 1.0, "y": 2.0}] * 3}
        b = {"mac_protocol": 1, "relays": [{"x": 1.0, "y": 2.0}] * 3}

        started(store, chromosomes=[a, b])

        assert store.experiment["status"] == EnumStatus.ERROR
        assert "same as an earlier one" in store.experiment["system_message"]
        assert store.simulations == []

    def test_a_study_that_cannot_build_its_panel_ends_in_error(self, store, monkeypatch):
        def refuse(*args, **kwargs):
            raise ValueError("no feasible chromosome")
        monkeypatch.setattr("lib.strategy.seed_study.choose_workload", refuse)

        started(store)

        assert store.experiment["status"] == EnumStatus.ERROR
        assert "no feasible chromosome" in store.experiment["system_message"]


class TestTheControl:
    def test_the_main_generation_does_not_end_the_study(self, store):
        strategy = started(store, size=2)

        finish_main(strategy, store)

        assert store.experiment["status"] == "Running"
        assert len(store.generations) == 2

    def test_it_repeats_the_first_seed_on_every_chromosome(self, store):
        strategy = started(store, size=3, control=2)

        finish_main(strategy, store)

        control = store.of(store.generations[1]["_id"])
        assert len(control) == 6
        assert {s["random_seed"] for s in control} == {SEEDS[0]}
        counts = {i["individual_id"]: 0 for i in store.individuals}
        for sim in control:
            counts[sim["individual_id"]] += 1
        assert set(counts.values()) == {2}

    def test_it_has_a_generation_of_its_own_with_no_plan(self, store):
        """In the main one the repeats would be collapsed as one result per seed,
        and a plan counts seeds, which a repetition does not add."""
        strategy = started(store)

        finish_main(strategy, store)

        generation = store.generations[1]
        assert generation["seed_study"] == {"role": "control"}
        assert generation["index"] == 1
        assert "expected_seeds" not in generation

    def test_its_runs_are_the_same_simulation(self, store):
        strategy = started(store, size=1)

        finish_main(strategy, store)

        main = next(s for s in store.of(store.generations[0]["_id"])
                    if s["random_seed"] == SEEDS[0])
        control = store.of(store.generations[1]["_id"])[0]
        assert (control["parameters"]["simulationElements"]
                == main["parameters"]["simulationElements"])
        assert control["parameters"]["randomSeed"] == SEEDS[0]
        assert control["source_repository_id"] == main["source_repository_id"]

    def test_the_study_ends_after_it(self, store):
        strategy = started(store, size=2)
        finish_main(strategy, store)

        strategy.event_generation_done(store.close(store.generations[1]["_id"]))

        assert store.experiment["status"] == EnumStatus.DONE
        assert store.experiment["pareto_front"]
        assert "2 chromosome(s), 3 seed(s)" in store.experiment["system_message"]

    def test_a_failed_control_does_not_fail_the_study(self, store):
        """The measurements are good. The analysis reports what was not compared."""
        strategy = started(store, size=2)
        finish_main(strategy, store)

        strategy.event_generation_done(
            store.close(store.generations[1]["_id"], failed=[SEEDS[0]]))

        assert store.experiment["status"] == EnumStatus.DONE

    def test_the_same_event_delivered_twice_enqueues_once(self, store):
        strategy = started(store, size=2)
        first = store.close(store.generations[0]["_id"])

        strategy.event_generation_done(first)
        strategy.event_generation_done(first)

        assert len(store.generations) == 2
        assert len(store.of(store.generations[1]["_id"])) == 4

    def test_a_control_closed_while_still_running_waits(self, store):
        """The first repeats can finish before the rest exist."""
        strategy = started(store, size=2)
        finish_main(strategy, store)
        control = store.generations[1]
        event = {"fullDocument": {"_id": control["_id"], "status": EnumStatus.DONE}}

        strategy.event_generation_done(event)

        assert store.experiment["status"] == "Running"

    def test_no_control_when_none_was_asked_for(self, store):
        strategy = started(store, control=0)

        finish_main(strategy, store)

        assert len(store.generations) == 1
        assert store.experiment["status"] == EnumStatus.DONE

    def test_a_control_that_cannot_be_enqueued_is_stated(self, store, monkeypatch):
        strategy = started(store)
        monkeypatch.setattr(strategy, "_enqueue_control", lambda: 1 / 0)

        finish_main(strategy, store)

        assert store.experiment["status"] == EnumStatus.DONE
        assert "could not be enqueued" in store.experiment["system_message"]

    def test_cancelling_during_the_control_leaves_it_cancelled(self, store):
        strategy = started(store, size=2)
        finish_main(strategy, store)
        store.cancelled = True

        strategy.event_generation_done(store.close(store.generations[1]["_id"]))

        assert store.experiment["status"] == "Running", "the cancel wrote its own status"


class TestFailures:
    def test_a_failed_replica_is_repeated_with_the_same_seed(self, store):
        strategy = started(store, size=1)
        bad = SEEDS[1]

        finish_main(strategy, store, failed=[bad])

        assert len(store.generations) == 1, "the control waits for the repair"
        clone = store.of(store.generations[0]["_id"])[-1]
        assert clone["random_seed"] == bad and clone["status"] == EnumStatus.WAITING

    def test_a_replica_that_fails_again_is_accepted_and_the_study_goes_on(self, store):
        strategy = started(store, size=1)
        bad = SEEDS[1]
        finish_main(strategy, store, failed=[bad])

        finish_main(strategy, store, failed=[bad])

        assert len(store.generations) == 2, "ended on the seeds that finished"
        assert store.individuals[0]["completeness"]["state"] == "partial"
        assert store.individuals[0]["completeness"]["objective_used"] is True

    def test_nothing_finished_is_an_error_with_a_reason(self, store):
        strategy = started(store, size=1, simulation={"replica_policy": "none"})

        finish_main(strategy, store, failed=SEEDS)

        assert store.experiment["status"] == EnumStatus.ERROR
        assert "no simulation" in store.experiment["system_message"]
        assert len(store.generations) == 1


class TestResuming:
    def test_a_restart_reads_the_panel_back(self, store):
        started(store, size=3)
        recorded = store.experiment["seed_study_panel"]["genome_hashes"]

        resumed = started(store, size=3)

        assert [c.get_hash() for c in resumed._chromosomes] == recorded
        assert len(store.generations) == 1, "nothing was enqueued again"
        assert len(store.simulations) == 9

    def test_generations_with_no_panel_refuse_to_guess(self, store):
        started(store, size=3)
        del store.experiment["seed_study_panel"]

        started(store, size=3)

        assert store.experiment["status"] == EnumStatus.ERROR
        assert "mix two measurements" in store.experiment["system_message"]

    def test_a_main_generation_that_finished_while_down_goes_on_to_the_control(self, store):
        started(store, size=2)
        store.close(store.generations[0]["_id"])

        started(store, size=2)

        assert len(store.generations) == 2
        assert store.generations[1]["seed_study"]["role"] == "control"

    def test_a_control_that_finished_while_down_ends_the_study(self, store):
        strategy = started(store, size=2)
        finish_main(strategy, store)
        store.close(store.generations[1]["_id"])

        started(store, size=2)

        assert store.experiment["status"] == EnumStatus.DONE
        assert store.experiment["pareto_front"]

    def test_a_control_still_running_is_waited_for(self, store):
        strategy = started(store, size=2)
        finish_main(strategy, store)

        resumed = started(store, size=2)

        assert store.experiment["status"] == "Running"
        assert len(store.generations) == 2
        resumed.event_generation_done(store.close(store.generations[1]["_id"]))
        assert store.experiment["status"] == EnumStatus.DONE


class TestSynthetic:
    def test_no_firmware_and_no_files(self, store, monkeypatch):
        """The way the whole path is tested without Cooja."""
        def refuse(*args):
            raise AssertionError("a synthetic run creates no files")
        monkeypatch.setattr(batch_module, "create_files", refuse)
        doc = experiment(size=2, simulation={
            "synthetic": {"enabled": True, "bench": "DTLZ2", "noise_std": 0.1}})
        doc["source_repository_options"] = {}

        SeedStudyStrategy(doc, store).start()

        assert len(store.simulations) == 6
        assert {s["source_repository_id"] for s in store.simulations} == {None}
