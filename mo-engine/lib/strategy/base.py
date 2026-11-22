from abc import ABC, abstractmethod
from pylib.db.models import Experiment
from pylib.db import MongoRepository

from .enqueue import ResumableEnqueue


class EngineStrategy(ResumableEnqueue, ABC):
    def __init__(self, experiment: Experiment, mongo: MongoRepository):
        self.experiment = experiment
        self.mongo = mongo

    def _stop_if_cancelled(self) -> bool:
        """Let go of an experiment that was cancelled or deleted.

        Asked by the polling fallback, so that it does not depend on a
        generation closing: a deleted one never does.
        """
        if not self.mongo.experiment_repo.is_cancelled(str(self._exp_id)):
            return False
        self.stop()
        return True

    @abstractmethod
    def start(self):
        """
        Initializes the execution strategy.
        This involves one or more watching threads.
        """
        pass

    @abstractmethod
    def event_simulation_done(self, sim_doc: dict):
        """
        Called for every simulation that reaches a terminal state (DONE or ERROR).
        Intended for accounting only (e.g. progress counters, logging).
        Must NOT trigger flow-control decisions.
        """
        pass

    @abstractmethod
    def event_generation_done(self, gen_doc: dict):
        """
        Called when a generation reaches a terminal state (DONE or ERROR).
        Controls the algorithm flow: objective extraction, evolution, next generation.
        Replaces the former event_batch_done.
        """
        pass

    @abstractmethod
    def stop(self):
        """Closes threads."""
        pass
