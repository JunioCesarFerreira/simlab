from abc import ABC, abstractmethod
from pylib.dto import Experiment
from pylib.mongo_db import MongoRepository

class EngineStrategy(ABC):
    def __init__(self, experiment: Experiment, mongo: MongoRepository):
        self.experiment = experiment
        self.mongo = mongo

    @abstractmethod
    def start(self):
        """Inicializa a estratégia de execução"""
        pass

    @abstractmethod
    def on_simulation_result(self, result_doc: dict):
        """Recebe um resultado de simulação e decide o próximo passo"""
        pass
