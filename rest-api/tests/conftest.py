import sys
import os
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

# ── path setup ─────────────────────────────────────────────────────────────────
# Allow imports of both `api.*` (rest-api/) and `pylib.*` (simlab root)
_RESTAPI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SIMLAB_ROOT  = os.path.abspath(os.path.join(_RESTAPI_ROOT, ".."))
for _p in (_RESTAPI_ROOT, _SIMLAB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── shared IDs ─────────────────────────────────────────────────────────────────
EXP_ID  = "507f1f77bcf86cd799439011"
GEN_ID  = "507f1f77bcf86cd799439012"
SIM_ID  = "507f1f77bcf86cd799439013"
IND_ID  = "507f1f77bcf86cd799439014"
SRC_ID  = "507f1f77bcf86cd799439015"
FILE_ID = "507f1f77bcf86cd799439016"


# ── sample mongo documents ──────────────────────────────────────────────────────
def sample_experiment():
    return {
        "_id": ObjectId(EXP_ID),
        "name": "Test Experiment",
        "status": "Waiting",
        "system_message": "",
        "created_time": None,
        "start_time": None,
        "end_time": None,
        "parameters": {
            "strategy": "nsga3",
            "algorithm": {},
            "simulation": {},
            "problem": {},
            "objectives": [],
        },
        "source_repository_options": {},
        "data_conversion_config": {"node_col": "node", "time_col": "time", "metrics": []},
        "pareto_front": None,
        "analysis_files": {},
    }


def sample_generation():
    return {
        "_id": ObjectId(GEN_ID),
        "experiment_id": ObjectId(EXP_ID),
        "index": 0,
        "status": "Done",
        "start_time": datetime(2024, 1, 1),
        "end_time": datetime(2024, 1, 2),
    }


def sample_individual():
    return {
        "_id": ObjectId(IND_ID),
        "individual_id": "abc123hash",
        "chromosome": {"positions": [[0.1, 0.2]]},
        "objectives": [0.5, 0.3],
        "topology_picture_id": None,
        "experiment_id": ObjectId(EXP_ID),
        "generation_id": ObjectId(GEN_ID),
    }


def sample_simulation():
    return {
        "_id": ObjectId(SIM_ID),
        "experiment_id": ObjectId(EXP_ID),
        "generation_id": ObjectId(GEN_ID),
        "individual_id": "abc123hash",
        "status": "Waiting",
        "system_message": "",
        "random_seed": 42,
        "start_time": None,
        "end_time": None,
        "parameters": {
            "name": "sim-0",
            "duration": 120,
            "randomSeed": 42,
            "radiusOfReach": 100.0,
            "radiusOfInter": 50.0,
            "region": [0.0, 0.0, 100.0, 100.0],
            "simulationElements": {"fixedMotes": [], "mobileMotes": []},
        },
        "pos_file_id": "",
        "csc_file_id": "",
        "source_repository_id": "",
        "log_cooja_id": "",
        "runtime_log_id": "",
        "csv_log_id": "",
        "network_metrics": {},
    }


def sample_source():
    return {
        "_id": ObjectId(SRC_ID),
        "name": "Test Source",
        "description": "A test source repository",
        "source_files": [{"id": FILE_ID, "file_name": "main.c"}],
    }


# ── fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_factory():
    return MagicMock()


@pytest.fixture
def client(mock_factory):
    from main import app
    from api.dependencies import get_factory
    from api.auth import get_api_key

    app.dependency_overrides[get_factory] = lambda: mock_factory
    app.dependency_overrides[get_api_key] = lambda: "api-password"

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
