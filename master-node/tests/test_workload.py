"""Which runs may share one held-time estimate."""
import pytest
from bson import ObjectId

from lib.scheduler.workload import workload_of

SOURCE = ObjectId("507f1f77bcf86cd799430099")


def sim(*, source=SOURCE, duration=60, fixed=20, mobile=6):
    return {"source_repository_id": source, "parameters": {
        "duration": duration,
        "simulationElements": {"fixedMotes": [{}] * fixed, "mobileMotes": [{}] * mobile},
    }}


def test_the_same_work_shares_an_estimate():
    assert workload_of(sim()) == workload_of(sim())


@pytest.mark.parametrize("other", [
    sim(source=ObjectId()),         # another firmware: TSCH costs ~13x CSMA
    sim(duration=180),              # three times the simulated time
    sim(fixed=100),                 # four times the network
])
def test_work_of_another_cost_does_not(other):
    assert workload_of(other) != workload_of(sim())


def test_small_changes_of_size_share_it():
    # Within a power of two: P2 varies its relays from one individual to the next.
    assert workload_of(sim(fixed=12)) == workload_of(sim(fixed=24))


@pytest.mark.parametrize("document", [
    sim(duration=1.5), sim(source="$bad.id"), {}, {"parameters": None},
    sim(duration="sixty"),
])
def test_the_key_can_always_be_a_field_name(document):
    key = workload_of(document)

    assert key and key[0].isalpha()
    assert "." not in key and "$" not in key
