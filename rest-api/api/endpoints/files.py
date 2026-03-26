import os
import shutil
import tempfile
import zipfile
import mimetypes

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from bson import ObjectId, errors as bson_errors
from gridfs.errors import NoFile

from pylib.db import MongoRepository
from api.dependencies import get_factory

router = APIRouter()


@router.get("/{file_id}/as/{extension}", response_class=FileResponse)
def download_file(
    file_id: str,
    extension: str,
    background_tasks: BackgroundTasks,
    factory: MongoRepository = Depends(get_factory)
):
    """Download a GridFS file returning it with the given extension and inferred MIME type."""
    try:
        oid = ObjectId(file_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid file_id")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
    tmp_path = tmp.name
    tmp.close()

    try:
        factory.fs_handler.download_file(oid, tmp_path)
    except NoFile:
        os.remove(tmp_path)
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(os.remove, tmp_path)

    ext = extension.lower()
    mime_type, _ = mimetypes.guess_type(f"file.{ext}")
    if mime_type is None:
        mime_type = "text/plain; charset=utf-8" if ext in {"txt", "log", "csv", "xml", "csc", "dat"} \
                    else "application/octet-stream"

    return FileResponse(tmp_path, filename=f"{file_id}.{ext}", media_type=mime_type,
                        background=background_tasks)


@router.get("/simulations/{simulation_id}/topology", response_class=FileResponse)
def download_topology_by_simulation(
    simulation_id: str,
    background_tasks: BackgroundTasks,
    factory: MongoRepository = Depends(get_factory)
):
    """Download the topology image associated with a simulation."""
    try:
        ObjectId(simulation_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid simulation_id")

    file_id = factory.simulation_repo.get_topology_pic_file_id(simulation_id)
    if file_id is None:
        raise HTTPException(status_code=404, detail="Topology picture not found for this simulation")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()

    try:
        factory.fs_handler.download_file(file_id, tmp_path)
    except NoFile:
        os.remove(tmp_path)
        raise HTTPException(status_code=404, detail="File not found in GridFS")
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(os.remove, tmp_path)
    return FileResponse(tmp_path, filename=f"{simulation_id}_topology.png", media_type="image/png",
                        background=background_tasks)


@router.get("/experiments/{experiment_id}/analysis/zip", response_class=FileResponse)
def download_experiment_analysis_zip(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    factory: MongoRepository = Depends(get_factory)
):
    """Download all analysis files of an experiment as a ZIP archive."""
    try:
        ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")

    analysis_files = factory.experiment_repo.find_analysis_files(experiment_id)
    if not analysis_files:
        raise HTTPException(status_code=404, detail="No analysis files found for this experiment")

    work_dir = tempfile.mkdtemp(prefix="simlab_analysis_")
    zip_path = os.path.join(work_dir, f"{experiment_id}_analysis.zip")

    try:
        for description, file_id in analysis_files.items():
            if not file_id:
                continue
            try:
                oid = ObjectId(file_id)
            except bson_errors.InvalidId:
                continue
            safe_name = description.replace(" ", "_") + ".png"
            factory.fs_handler.download_file(oid, os.path.join(work_dir, safe_name))

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(work_dir):
                fpath = os.path.join(work_dir, fname)
                if fpath != zip_path:
                    zipf.write(fpath, arcname=fname)
    except NoFile:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=404, detail="One or more analysis files were not found in GridFS")
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(shutil.rmtree, work_dir, True)
    return FileResponse(zip_path, filename=f"{experiment_id}_analysis.zip", media_type="application/zip",
                        background=background_tasks)


@router.get("/experiments/{experiment_id}/topologies/zip", response_class=FileResponse)
def download_experiment_topologies_zip(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    factory: MongoRepository = Depends(get_factory)
):
    """
    Download all topology images for all generations of an experiment as a ZIP.
    Structure: {gen_index}-{gen_id}/{ind_index}-{individual_id}_topology.png
    """
    try:
        exp_oid = ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")

    if not factory.experiment_repo.get(experiment_id):
        raise HTTPException(status_code=404, detail="Experiment not found")

    generations = factory.generation_repo.find_by_experiment(exp_oid)
    if not generations:
        raise HTTPException(status_code=404, detail="No generations found for this experiment")

    work_dir = tempfile.mkdtemp(prefix="simlab_exp_topologies_")
    files_dir = os.path.join(work_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, f"{experiment_id}_topologies.zip")

    try:
        for generation in generations:
            gen_id = generation["_id"]
            gen_index = generation.get("index", 0)
            individuals = factory.individual_repo.find_by_generation(gen_id)
            if not individuals:
                continue

            gen_subfolder = os.path.join(files_dir, f"{gen_index}-{gen_id}")
            os.makedirs(gen_subfolder, exist_ok=True)

            for ind_index, individual in enumerate(individuals):
                topology_file_id = individual.get("topology_picture_id")
                if not topology_file_id:
                    continue
                try:
                    oid = ObjectId(topology_file_id)
                except bson_errors.InvalidId:
                    continue
                ind_id_str = individual.get("individual_id", str(ind_index))
                file_path = os.path.join(gen_subfolder, f"{ind_index}-{ind_id_str}_topology.png")
                try:
                    factory.fs_handler.download_file(oid, file_path)
                except Exception:
                    continue

        has_files = any(fnames for _, _, fnames in os.walk(files_dir))
        if not has_files:
            raise HTTPException(status_code=404, detail="No topology files were successfully downloaded")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for dirpath, _, filenames in os.walk(files_dir):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    zipf.write(fpath, arcname=os.path.relpath(fpath, files_dir))

    except NoFile:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=404, detail="One or more topology files were not found in GridFS")
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    background_tasks.add_task(shutil.rmtree, work_dir, True)
    return FileResponse(zip_path, filename=f"{experiment_id}_topologies.zip", media_type="application/zip",
                        background=background_tasks)
