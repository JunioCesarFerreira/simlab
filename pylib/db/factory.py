from typing import NamedTuple

from pylib.db.connection import MongoDBConnection
from pylib.db.gridfs import MongoGridFSHandler
from pylib.db.repositories.experiment import ExperimentRepository
from pylib.db.repositories.simulation import SimulationRepository
from pylib.db.repositories.generation import GenerationRepository
from pylib.db.repositories.individual import IndividualRepository
from pylib.db.repositories.source import SourceRepositoryAccess
from pylib.db.repositories.analytics import AnalyticsRepository
from pylib.db.repositories.genome_cache import GenomeCacheRepository
from pylib.db.repositories.campaign import CampaignRepository


class MongoRepository(NamedTuple):
    experiment_repo: ExperimentRepository
    simulation_repo: SimulationRepository
    generation_repo: GenerationRepository
    individual_repo: IndividualRepository
    source_repo: SourceRepositoryAccess
    fs_handler: MongoGridFSHandler
    analytics_repo: AnalyticsRepository
    genome_cache_repo: GenomeCacheRepository
    campaign_repo: CampaignRepository


def create_mongo_repository_factory(mongo_uri: str, db_name: str) -> MongoRepository:
    connection = MongoDBConnection(mongo_uri, db_name)
    fs_handler = MongoGridFSHandler(connection)
    experiment_repo = ExperimentRepository(connection)
    simulation_repo = SimulationRepository(connection)
    generation_repo = GenerationRepository(connection)
    individual_repo = IndividualRepository(connection)
    source_repo = SourceRepositoryAccess(connection)
    analytics_repo = AnalyticsRepository(connection)
    genome_cache_repo = GenomeCacheRepository(connection)
    campaign_repo = CampaignRepository(connection)
    return MongoRepository(
        experiment_repo=experiment_repo,
        simulation_repo=simulation_repo,
        generation_repo=generation_repo,
        individual_repo=individual_repo,
        source_repo=source_repo,
        fs_handler=fs_handler,
        analytics_repo=analytics_repo,
        genome_cache_repo=genome_cache_repo,
        campaign_repo=campaign_repo,
    )
