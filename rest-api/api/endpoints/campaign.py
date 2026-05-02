from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from bson import errors as bson_errors

from pylib.db import MongoRepository
from api.dependencies import get_factory
from api.domain.campaign import CampaignDto, CampaignFullDto, CampaignInfoDto
from api.mappers.campaign import (
    campaign_from_mongo,
    campaign_full_from_mongo,
    campaign_info_from_mongo,
    campaign_to_mongo,
)
from api.mappers.experiment import experiment_from_mongo

router = APIRouter()


@router.post("/", response_model=str)
def create_campaign(
    campaign: CampaignDto,
    factory: MongoRepository = Depends(get_factory),
) -> str:
    """Create a new campaign. Returns the generated campaign_id."""
    try:
        doc = campaign_to_mongo(campaign)
        doc.setdefault("created_time", datetime.now())
        doc.setdefault("experiment_ids", [])
        return str(factory.campaign_repo.insert(doc))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=list[CampaignInfoDto])
def list_campaigns(
    factory: MongoRepository = Depends(get_factory),
) -> list[CampaignInfoDto]:
    """Retrieve all campaigns."""
    try:
        docs = factory.campaign_repo.find_all()
        return [campaign_info_from_mongo(d) for d in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}/full", response_model=CampaignFullDto)
def get_campaign_full(
    campaign_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> CampaignFullDto:
    """Retrieve a campaign with its experiments embedded."""
    try:
        doc = factory.campaign_repo.get(campaign_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Campaign not found")

        experiment_ids = [str(eid) for eid in doc.get("experiment_ids", [])]
        experiments = []
        for exp_id in experiment_ids:
            exp_doc = factory.experiment_repo.get(exp_id)
            if exp_doc:
                experiments.append(experiment_from_mongo(exp_doc))

        return campaign_full_from_mongo(doc, experiments)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{campaign_id}", response_model=CampaignDto)
def get_campaign(
    campaign_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> CampaignDto:
    """Retrieve a single campaign by its ObjectId."""
    try:
        doc = factory.campaign_repo.get(campaign_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign_from_mongo(doc)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{campaign_id}", response_model=bool)
def update_campaign(
    campaign_id: str,
    updates: dict,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Partially update a campaign using $set semantics."""
    try:
        return factory.campaign_repo.update(campaign_id, updates)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{campaign_id}", response_model=bool)
def delete_campaign(
    campaign_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Delete a campaign. Does not cascade to experiments."""
    try:
        result = factory.campaign_repo.delete(campaign_id)
        if not result:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return True
    except HTTPException:
        raise
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid campaign_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{campaign_id}/experiments/{experiment_id}", response_model=bool)
def add_experiment_to_campaign(
    campaign_id: str,
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Add an experiment to a campaign."""
    try:
        result = factory.campaign_repo.add_experiment(campaign_id, experiment_id)
        if not result:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return True
    except HTTPException:
        raise
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{campaign_id}/experiments/{experiment_id}", response_model=bool)
def remove_experiment_from_campaign(
    campaign_id: str,
    experiment_id: str,
    factory: MongoRepository = Depends(get_factory),
) -> bool:
    """Remove an experiment from a campaign."""
    try:
        result = factory.campaign_repo.remove_experiment(campaign_id, experiment_id)
        if not result:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return True
    except HTTPException:
        raise
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
