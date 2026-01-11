import zipfile
import shutil
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from bson import ObjectId, errors as bson_errors
from gridfs.errors import NoFile
import os, sys, tempfile
import mimetypes

project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from pylib import mongo_db

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/?replicaSet=rs0")
DB_NAME = os.getenv("DB_NAME", "simlab")
factory = mongo_db.create_mongo_repository_factory(MONGO_URI, DB_NAME)

router = APIRouter()

@router.get("/{file_id}/as/{extension}", response_class=FileResponse)
def download_file(file_id: str, extension: str, background_tasks: BackgroundTasks):
    """
    Download a file stored in GridFS and return it with a desired extension.
    
    - file_id must be a valid GridFS ObjectId
    - The file is streamed from GridFS into a temporary file
    - The temporary file is automatically removed after the response is sent
    - The extension parameter affects only the response filename and MIME type
    
    The original file content is preserved exactly as stored.
    """
    try:
        oid = ObjectId(file_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid file_id")

    # cria arquivo temporário .txt
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
    tmp_path = tmp.name
    tmp.close()

    try:
        # baixa o conteúdo do GridFS diretamente para o arquivo temporário
        factory.fs_handler.download_file(oid, tmp_path)
    except NoFile:
        os.remove(tmp_path)
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=500, detail=f"Erro ao baixar arquivo: {e}")

    # remove o temp após enviar a resposta
    background_tasks.add_task(os.remove, tmp_path)

    # garante minúsculas
    ext = extension.lower()

    # tenta adivinhar o tipo MIME
    mime_type, _ = mimetypes.guess_type(f"file.{ext}")

    # fallback
    if mime_type is None:
        if ext in {"txt", "log", "csv", "xml", "csc", "dat"}:
            mime_type = "text/plain; charset=utf-8"
        else:
            mime_type = "application/octet-stream"

    return FileResponse(
        tmp_path,
        filename=f"{file_id}.{ext}",
        media_type=mime_type,
        background=background_tasks,
    )


@router.get("/simulations/{simulation_id}/topology", response_class=FileResponse)
def download_topology_file_by_simulation(
    simulation_id: str,
    background_tasks: BackgroundTasks
):
    """
    Download the topology image associated with a simulation.

    - simulation_id must be a valid Simulation ObjectId
    - The topology image is retrieved via SimulationRepository
    - The file is streamed from GridFS into a temporary PNG file
    - The temporary file is removed after the response is sent
    """

    # ------------------------------------------------------------
    # Validate simulation_id
    # ------------------------------------------------------------
    try:
        _ = ObjectId(simulation_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid simulation_id")

    # ------------------------------------------------------------
    # Get topology_picture_id from simulation
    # ------------------------------------------------------------
    file_id = factory.simulation_repo.get_topology_pic_file_id(simulation_id)

    if file_id is None:
        raise HTTPException(
            status_code=404,
            detail="Topology picture not found for this simulation"
        )

    # ------------------------------------------------------------
    # Create temporary PNG file
    # ------------------------------------------------------------
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()

    # ------------------------------------------------------------
    # Download from GridFS
    # ------------------------------------------------------------
    try:
        factory.fs_handler.download_file(file_id, tmp_path)
    except NoFile:
        os.remove(tmp_path)
        raise HTTPException(status_code=404, detail="File not found in GridFS")
    except Exception as e:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading topology image: {e}"
        )

    # ------------------------------------------------------------
    # Cleanup after response
    # ------------------------------------------------------------
    background_tasks.add_task(os.remove, tmp_path)

    # ------------------------------------------------------------
    # MIME type (explicit for PNG)
    # ------------------------------------------------------------
    mime_type = "image/png"

    return FileResponse(
        tmp_path,
        filename=f"{simulation_id}_topology.png",
        media_type=mime_type,
        background=background_tasks,
    )
    
@router.get("/experiments/{experiment_id}/analysis/zip", response_class=FileResponse)
def download_experiment_analysis_zip(
    experiment_id: str,
    background_tasks: BackgroundTasks
):
    """
    Download all analysis files of an experiment as a ZIP archive.

    - experiment_id must be a valid Experiment ObjectId
    - All files referenced in experiment.analysis_files are fetched from GridFS
    - Files are packed into a ZIP and returned
    - Temporary files are removed after response
    """

    # ------------------------------------------------------------
    # Validate experiment_id
    # ------------------------------------------------------------
    try:
        exp_oid = ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")

    # ------------------------------------------------------------
    # Fetch analysis_files metadata
    # ------------------------------------------------------------
    analysis_files = factory.experiment_repo.find_analysis_files(experiment_id)

    if not analysis_files:
        raise HTTPException(
            status_code=404,
            detail="No analysis files found for this experiment"
        )

    # ------------------------------------------------------------
    # Create temp working directory
    # ------------------------------------------------------------
    work_dir = tempfile.mkdtemp(prefix="simlab_analysis_")
    zip_path = os.path.join(work_dir, f"{experiment_id}_analysis.zip")

    try:
        # --------------------------------------------------------
        # Download each analysis file
        # --------------------------------------------------------
        for description, file_id in analysis_files.items():
            if not file_id:
                continue

            try:
                oid = ObjectId(file_id)
            except bson_errors.InvalidId:
                continue

            # filename inside zip
            safe_name = description.replace(" ", "_") + ".png"
            file_path = os.path.join(work_dir, safe_name)

            factory.fs_handler.download_file(oid, file_path)

        # --------------------------------------------------------
        # Create ZIP
        # --------------------------------------------------------
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for fname in os.listdir(work_dir):
                fpath = os.path.join(work_dir, fname)
                if fpath == zip_path:
                    continue
                zipf.write(fpath, arcname=fname)

    except NoFile:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=404,
            detail="One or more analysis files were not found in GridFS"
        )
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating analysis ZIP: {e}"
        )

    # ------------------------------------------------------------
    # Cleanup after response
    # ------------------------------------------------------------
    background_tasks.add_task(shutil.rmtree, work_dir, True)

    return FileResponse(
        zip_path,
        filename=f"{experiment_id}_analysis.zip",
        media_type="application/zip",
        background=background_tasks,
    )