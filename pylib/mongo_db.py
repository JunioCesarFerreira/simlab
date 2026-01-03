from typing import NamedTuple

from .mongo.connection import MongoDBConnection, EnumStatus
from .mongo.experiment import ExperimentRepository
from .mongo.simulation import SimulationRepository
from .mongo.generation import GenerationRepository
from .mongo.source import SourceRepositoryAccess
from .mongo.gridfs_handler import MongoGridFSHandler
from .mongo.analytics import AnalyticsRepository

EnumStatus = EnumStatus # Para uso externo

# Fábrica de componentes

class MongoRepository(NamedTuple):
    experiment_repo: ExperimentRepository
    simulation_repo: SimulationRepository
    generation_repo: GenerationRepository
    source_repo: SourceRepositoryAccess
    fs_handler: MongoGridFSHandler
    analytics_repo: AnalyticsRepository


def create_mongo_repository_factory(mongo_uri: str, db_name: str) -> MongoRepository:
    connection = MongoDBConnection(mongo_uri, db_name)
    fs_handler = MongoGridFSHandler(connection)
    experiment_repo = ExperimentRepository(connection)
    simulation_repo = SimulationRepository(connection)
    simulation_queue_repo = GenerationRepository(connection)
    source_repo = SourceRepositoryAccess(connection)
    analytics_repo = AnalyticsRepository(connection)
    return MongoRepository(
        experiment_repo=experiment_repo,
        simulation_repo=simulation_repo,
        generation_repo=simulation_queue_repo,
        source_repo=source_repo,
        fs_handler=fs_handler,
        analytics_repo=analytics_repo
    )
