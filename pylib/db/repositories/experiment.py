import logging
from datetime import datetime
from typing import Optional, Callable, Any, Iterable
import gridfs
from bson import ObjectId, errors

from pylib.db.models.experiment import Experiment, DataConversionConfig
from pylib.db.models.enums import EnumStatus
from pylib.db.connection import MongoDBConnection

log = logging.getLogger(__name__)


class ExperimentRepository:
    def __init__(self, connection: MongoDBConnection):
        self.connection = connection

    def insert(self, experiment: Experiment) -> ObjectId:
        with self.connection.connect() as db:
            return db["experiments"].insert_one(experiment).inserted_id

    def find_by_status(self, status: EnumStatus) -> list[dict[str, Any]]:
        with self.connection.connect() as db:
            return list(db["experiments"].find(
                {"status": status},
                {"_id": 1, "name": 1, "system_message": 1, "start_time": 1, "end_time": 1,
                 "parameters.simulation.synthetic": 1}
            ))

    def find_all_info(self) -> list[dict[str, Any]]:
        """Lightweight listing of every experiment, status included."""
        with self.connection.connect() as db:
            return list(db["experiments"].find(
                {},
                {"_id": 1, "name": 1, "status": 1, "system_message": 1, "start_time": 1,
                 "end_time": 1, "parameters.simulation.synthetic": 1}
            ))

    def find_startable_by_status(self, status: EnumStatus) -> list[dict[str, Any]]:
        with self.connection.connect() as db:
            return list(db["experiments"].find({"status": status}))

    def find_analysis_files(self, experiment_id: str) -> dict[str, Any]:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", experiment_id)
            return None
        with self.connection.connect() as db:
            result = db["experiments"].find_one(
                {"_id": oid},
                {"analysis_files": 1})
            return result["analysis_files"]

    def update(self, experiment_id: str, updates: dict) -> bool:
        updates["id"] = experiment_id
        with self.connection.connect() as db:
            result = db["experiments"].update_one({"_id": ObjectId(experiment_id)}, {"$set": updates})
            return result.modified_count > 0

    def update_status(self, experiment_id: str, status: str) -> bool:
        return self.update(experiment_id, {"status": status})

    def update_starting(self, experiment_id: str) -> bool:
        with self.connection.connect() as db:
            result = db["experiments"].update_one(
                {"_id": ObjectId(experiment_id), "status": EnumStatus.WAITING},
                {"$set": {
                    "id": experiment_id,
                    "status": EnumStatus.RUNNING,
                    "start_time": datetime.now()
                }}
            )
            return result.modified_count > 0

    def get(self, experiment_id: str) -> Experiment:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID: %s", experiment_id)
            return None
        with self.connection.connect() as db:
            return db["experiments"].find_one({"_id": oid})

    def get_metrics_data_conversion(self, experiment_id: str) -> DataConversionConfig:
        try:
            oid = ObjectId(experiment_id)
        except errors.InvalidId:
            raise ValueError(f"Invalid experiment_id: {experiment_id!r}")

        with self.connection.connect() as db:
            doc = db["experiments"].find_one(
                {"_id": oid},
                {
                    "_id": 0,
                    "data_conversion_config.node_col": 1,
                    "data_conversion_config.time_col": 1,
                    "data_conversion_config.metrics": 1,
                }
            )

        if not doc:
            return {}

        return doc.get("data_conversion_config") or {}

    # GridFS artifacts owned exclusively by a single experiment. Source-code
    # repositories (``source_repository_id`` / ``source_repository_options``)
    # are shared across experiments and are intentionally excluded here.
    _SIM_FILE_FIELDS = (
        "pos_file_id", "csc_file_id",
        "log_cooja_id", "runtime_log_id", "csv_log_id",
    )

    def delete(self, experiment_id: str) -> dict[str, int]:
        """
        Delete an experiment by _id and cascade-delete all related documents
        and the GridFS artifacts they own (topology pictures, simulation
        position/config/log/csv files and analysis files).

        Shared artifacts such as source-code repositories are NOT touched.
        Returns counters for each collection / artifact type affected.
        """
        try:
            exp_oid = ObjectId(experiment_id)
        except errors.InvalidId:
            log.error("Invalid ID")
            return {"deleted_experiments": 0, "deleted_generations": 0,
                    "deleted_individuals": 0, "deleted_simulations": 0,
                    "deleted_genome_cache": 0, "deleted_files": 0}

        with self.connection.connect() as db:
            fs = gridfs.GridFS(db)

            # Collect owned GridFS file ids before removing their documents.
            file_ids: list[ObjectId] = []

            for ind in db["individuals"].find(
                {"experiment_id": exp_oid}, {"topology_picture_id": 1}
            ):
                file_ids.append(ind.get("topology_picture_id"))

            sim_projection = {field: 1 for field in self._SIM_FILE_FIELDS}
            for sim in db["simulations"].find({"experiment_id": exp_oid}, sim_projection):
                for field in self._SIM_FILE_FIELDS:
                    file_ids.append(sim.get(field))

            exp_doc = db["experiments"].find_one({"_id": exp_oid}, {"analysis_files": 1})
            if exp_doc:
                file_ids.extend((exp_doc.get("analysis_files") or {}).values())

            files_deleted = self._delete_files(fs, file_ids)

            sims_deleted = db["simulations"].delete_many({"experiment_id": exp_oid}).deleted_count
            inds_deleted = db["individuals"].delete_many({"experiment_id": exp_oid}).deleted_count
            gens_deleted = db["generations"].delete_many({"experiment_id": exp_oid}).deleted_count
            cache_deleted = db["genome_cache"].delete_many({"experiment_id": exp_oid}).deleted_count
            exp_deleted = db["experiments"].delete_one({"_id": exp_oid}).deleted_count

            return {
                "deleted_experiments": int(exp_deleted),
                "deleted_generations": int(gens_deleted),
                "deleted_individuals": int(inds_deleted),
                "deleted_simulations": int(sims_deleted),
                "deleted_genome_cache": int(cache_deleted),
                "deleted_files": files_deleted,
            }

    @staticmethod
    def _delete_files(fs: gridfs.GridFS, file_ids: Iterable[Optional[ObjectId]]) -> int:
        """Delete each valid, unique GridFS id, tolerating missing files."""
        deleted = 0
        seen: set[ObjectId] = set()
        for fid in file_ids:
            if not isinstance(fid, ObjectId) or fid in seen:
                continue
            seen.add(fid)
            try:
                fs.delete(fid)
                deleted += 1
            except Exception as e:  # pragma: no cover - defensive, missing file
                log.warning("Failed to delete GridFS file %s: %s", fid, e)
        return deleted

    def add_analysis_file_to_experiment(self,
            experiment_id: str,
            description: str,
            path: str,
            name: str
    ) -> Optional[ObjectId]:
        from pylib.db.gridfs import MongoGridFSHandler
        with self.connection.connect() as db:
            grid = MongoGridFSHandler(self.connection)
            file_id = grid.upload_file(path, name)

            result = db.experiments.update_one(
                {"_id": ObjectId(experiment_id)},
                {"$set": {f"analysis_files.{description}": file_id}}
            )
            if result.matched_count == 0:
                raise ValueError("Experiment not found")

            return file_id

    def watch_status_waiting(self, on_change: Callable[[dict], None]):
        log.info("[ExperimentRepository] Waiting new experiments...")
        pipeline = [
            {
                "$match": {
                    "operationType": {"$in": ["insert", "update", "replace"]},
                    "fullDocument.status": EnumStatus.WAITING
                }
            }
        ]
        self.connection.watch_collection(
            "experiments",
            pipeline,
            on_change,
            full_document="updateLookup"
        )
