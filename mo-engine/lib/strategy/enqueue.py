"""Writing a generation, and finishing one that a dead process left half written.

A generation is several documents: itself, its plan, its individuals, their
cache entries and their simulations. The generation goes in first, with
``enqueue_complete: false``, and the flag turns true once all of it is there.
Found false on resume, it is a fragment, and evaluating it would be evaluating
fewer individuals or seeds than configured.

It is finished rather than discarded: its simulations may be running, or done.

1. The fragment is set aside, and the strategy forgets the genomes it
   introduced. What is left is exactly the state right before that enqueue:
   the previous generation, terminal, waiting for its handler.
2. The strategy resumes from there as from any other checkpoint. The handler
   is deterministic, so it reaches the enqueue with the same population and
   the same random state.
3. The enqueue adopts the fragment instead of inserting a generation, and
   skips whatever is already written.

Step 2 is checked, not assumed. A fragment with another index, random state,
plan, or an individual outside the population drawn again means the code or
the experiment changed since it was written; finishing it would mix two runs,
so the experiment ends in ``Error`` and says why.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Iterable, Mapping, Optional

from pylib.db.models.enums import EnumStatus
from pylib.timeutil import utc_now

logger = logging.getLogger(__name__)


class CannotFinishEnqueue(RuntimeError):
    """The fragment is not what this process would have written."""


class ResumableEnqueue:
    """What a strategy writes a generation through. Mixed into ``EngineStrategy``.

    The defaults describe a generation nothing was written of, which is every
    generation except one being finished.
    """

    _cut: Optional[dict] = None
    _adopted: Optional[dict] = None
    _written_individuals: Mapping[str, dict] = {}
    _written_replicas: Mapping[tuple, int] = {}
    _written_cache: frozenset = frozenset()

    # ── on resume ──────────────────────────────────────────────────────────

    def _set_aside_cut(
        self,
        generations: list,
        *,
        inserted: Optional[set] = None,
        caches: Iterable[dict] = (),
    ) -> list:
        """``generations`` without a fragment at their end, which is remembered."""
        if not generations or generations[-1].get("enqueue_complete") is not False:
            return generations
        self._cut = generations[-1]
        for document in self.mongo.individual_repo.find_by_generation(self._cut["_id"]):
            if (document.get("completeness") or {}).get("reused"):
                continue    # known before the fragment
            genome_hash = document["individual_id"]
            if inserted is not None:
                inserted.discard(genome_hash)
            for cache in caches:
                cache.pop(genome_hash, None)
        logger.warning(
            "Experiment %s: generation %s was cut while it was being enqueued; "
            "finishing it.", self._exp_id, self._cut.get("index"),
        )
        return generations[:-1]

    # ── on enqueue ─────────────────────────────────────────────────────────

    def _open_generation(
        self, document: dict, hashes: Iterable[str], seeds: Optional[list] = None
    ) -> Any:
        """Insert the generation, or adopt the fragment it finishes."""
        self._adopted = None
        self._written_individuals, self._written_replicas = {}, {}
        self._written_cache = frozenset()
        cut = self._cut
        if cut is None:
            document["enqueue_complete"] = False
            return self.mongo.generation_repo.insert(document)

        individuals = {
            d["individual_id"]: d
            for d in self.mongo.individual_repo.find_by_generation(cut["_id"])}
        difference = _difference(cut, document, individuals, set(hashes), seeds)
        if difference:
            self._refuse_cut(cut, difference)

        self._written_individuals = individuals
        self._written_replicas = Counter(
            (s.get("individual_id"), s.get("random_seed"))
            for s in self.mongo.simulation_repo.find_replicas(cut["_id"]))
        cache = getattr(self.mongo, "genome_cache_repo", None)
        if cache is not None:
            self._written_cache = frozenset(
                e["genome_hash"] for e in cache.get_all_by_experiment(self._exp_id))
        # The master may have closed what there was of it.
        self.mongo.generation_repo.update(
            cut["_id"], {"status": EnumStatus.WAITING, "end_time": None})
        self._cut, self._adopted = None, cut
        return cut["_id"]

    def _insert_individual(self, document: dict) -> None:
        if document["individual_id"] not in self._written_individuals:
            self.mongo.individual_repo.insert(document)

    def _insert_cache_entry(self, genome_hash: str, chromosome: dict) -> None:
        if genome_hash not in self._written_cache:
            self.mongo.genome_cache_repo.insert(self._exp_id, genome_hash, chromosome)

    def _replica_written(self, genome_hash: str, seed: int) -> bool:
        return (genome_hash, seed) in self._written_replicas

    def _has_picture(self, genome_hash: str) -> bool:
        document = self._written_individuals.get(genome_hash) or {}
        return bool(document.get("topology_picture_id"))

    def _generation_written(self, generation_id: Any) -> None:
        """All of it is there."""
        self.mongo.generation_repo.update(generation_id, {"enqueue_complete": True})
        if self._adopted is not None:
            # Its simulations may all have ended while the engine was down, and
            # then none is left whose end would close it.
            self.mongo.generation_repo.close_if_finished(
                generation_id, require_simulations=True)

    def _refuse_cut(self, cut: dict, difference: str) -> None:
        reason = (
            f"Generation {cut.get('index')} was cut while being enqueued and cannot be "
            f"finished: {difference}, so the code or the experiment changed since. "
            "Relaunch the experiment."
        )
        # Here and not only in the engine: the refusal can also be reached from
        # a watcher thread, where nothing would turn the exception into a status.
        self.mongo.experiment_repo.update_unless_cancelled(str(self._exp_id), {
            "status": EnumStatus.ERROR, "system_message": reason, "end_time": utc_now()})
        self.stop()
        raise CannotFinishEnqueue(reason)


def _difference(
    cut: dict, document: dict, individuals: dict, hashes: set, seeds: Optional[list]
) -> Optional[str]:
    """How the fragment differs from what would be written now, if it does."""
    if cut.get("index") != document.get("index"):
        return f"the next generation would be {document.get('index')}"
    for field, name in (("rng_state", "the random state"), ("benchmark", "the wave")):
        if field in cut and cut[field] != document.get(field):
            return f"{name} differs"
    planned = cut.get("expected_seeds")
    if planned and seeds is not None and sorted(planned) != sorted(seeds):
        return "the simulation seeds differ"
    strangers = set(individuals) - hashes
    if strangers:
        return f"{len(strangers)} of its individuals are not in the population drawn again"
    return None
