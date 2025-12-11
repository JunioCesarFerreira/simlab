from abc import ABC, abstractmethod
from pylib.dto.experiment import Experiment
from pylib.mongo_db import MongoRepository

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
    def event_simulation_done(self, sim_doc: dict):
        """
        Receive a simulation result and decide the next step
        Should be used to handle the simulation completion event
        """
        pass
    
    @abstractmethod
    def stop(self):
        """Closes threads"""
        pass
