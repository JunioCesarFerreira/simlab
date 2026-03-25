from abc import ABC, abstractmethod
from pylib.db.models import Experiment
from pylib.db import MongoRepository

class EngineStrategy(ABC):
    def __init__(self, experiment: Experiment, mongo: MongoRepository):
        self.experiment = experiment
        self.mongo = mongo

    @abstractmethod
    def start(self):
        """
        Initializes the execution strategy.
        This involves one or more watching threads.
        """
        pass

    @abstractmethod
    def event_batch_done(self, sim_doc: dict):
        """
        Receive a batch result and decide the next step
        """
        pass
    
    @abstractmethod
    def stop(self):
        """Closes threads"""
        pass
