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
    

@router.get("/experiments/{experiment_id}/topologies/zip", response_class=FileResponse)
def download_experiment_topologies_zip(
    experiment_id: str,
    background_tasks: BackgroundTasks
):
    """
    Download all topology images of all generations in an experiment as a ZIP archive.
    
    Structure:
    - {experiment_id}_topologies.zip
      ├── 0-{generation_id_1}/
      │   ├── 0-{simulation_id_1}_topology.png
      │   ├── 1-{simulation_id_2}_topology.png
      │   └── ...
      ├── 1-{generation_id_2}/
      │   ├── 0-{simulation_id_N}_topology.png
      │   ├── 1-{simulation_id_M}_topology.png
      │   └── ...
      └── ...
      
    - experiment_id must be a valid Experiment ObjectId
    - Retrieves all generations for the experiment
    - For each generation, retrieves all simulations
    - For each simulation, fetches the topology image from GridFS
    - Files are organized in subfolders named "{gen_index}-{generation_id}"
    - Files within folders are named "{sim_index}-{simulation_id}_topology.png"
    - Temporary files are removed after response
    """
    
    # Validate experiment_id
    try:
        exp_oid = ObjectId(experiment_id)
    except bson_errors.InvalidId:
        raise HTTPException(status_code=400, detail="Invalid experiment_id")
    
    # Get experiment document to retrieve generation IDs
    experiment = factory.experiment_repo.get(experiment_id)
    
    if not experiment:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found"
        )
    
    # Get all generation IDs from experiment
    generation_ids = experiment.get("generations", [])
    
    if not generation_ids or len(generation_ids) == 0:
        raise HTTPException(
            status_code=404,
            detail="No generations found for this experiment"
        )
    
    # Create temp working directory for files (separate from ZIP location)
    work_dir = tempfile.mkdtemp(prefix="simlab_exp_topologies_")
    files_dir = os.path.join(work_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, f"{experiment_id}_topologies.zip")
    
    try:
        # Process each generation
        for gen_index, generation_id in enumerate(generation_ids):
            try:
                gen_id_str = str(generation_id) if isinstance(generation_id, ObjectId) else generation_id
                
                # Get generation document
                generation = factory.batch_repo.get(gen_id_str)
                
                if not generation:
                    print(f"Warning: Generation {gen_id_str} not found")
                    continue
                
                # Get simulations for this generation
                simulations_ids = generation.get("simulations_ids", [])
                
                if not simulations_ids:
                    print(f"Warning: No simulations found for generation {gen_id_str}")
                    continue
                
                # Create subfolder for this generation: {gen_index}-{generation_id}
                gen_subfolder = os.path.join(files_dir, f"{gen_index}-{gen_id_str}")
                os.makedirs(gen_subfolder, exist_ok=True)
                
                # Download topologies for each simulation in this generation
                for sim_index, simulation_id in enumerate(simulations_ids):
                    try:
                        sim_id_str = str(simulation_id) if isinstance(simulation_id, ObjectId) else simulation_id
                        
                        # Get simulation document
                        simulation = factory.simulation_repo.get(sim_id_str)
                        
                        if not simulation:
                            print(f"Warning: Simulation {sim_id_str} not found")
                            continue
                        
                        topology_file_id = simulation.get("topology_picture_id")
                        
                        if not topology_file_id:
                            print(f"Warning: No topology found for simulation {sim_id_str}")
                            continue
                        
                        try:
                            oid = ObjectId(topology_file_id)
                        except bson_errors.InvalidId:
                            print(f"Warning: Invalid topology file ID for simulation {sim_id_str}")
                            continue
                        
                        # Create filename inside subfolder: {sim_index}-{simulation_id}_topology.png
                        safe_name = f"{sim_index}-{sim_id_str}_topology.png"
                        file_path = os.path.join(gen_subfolder, safe_name)
                        
                        factory.fs_handler.download_file(oid, file_path)
                        
                    except Exception as e:
                        # Log and continue with next simulation
                        print(f"Warning: Could not download topology for simulation {simulation_id}: {e}")
                        continue
                        
            except Exception as e:
                # Log and continue with next generation
                print(f"Warning: Could not process generation {generation_id}: {e}")
                continue
        
        # Verify that at least one topology was downloaded
        has_files = False
        for item in os.walk(files_dir):
            dirpath, dirnames, filenames = item
            if filenames:
                has_files = True
                break
        
        if not has_files:
            raise HTTPException(
                status_code=404,
                detail="No topology files were successfully downloaded"
            )
        
        # Create ZIP with nested subfolder structure from files_dir only
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for dirpath, dirnames, filenames in os.walk(files_dir):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    
                    # Calculate relative path for archive (relative to files_dir, not work_dir)
                    arcname = os.path.relpath(fpath, files_dir)
                    zipf.write(fpath, arcname=arcname)
    
    except NoFile:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=404,
            detail="One or more topology files were not found in GridFS"
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating experiment topologies ZIP: {e}"
        )
    
    # Cleanup after response
    background_tasks.add_task(shutil.rmtree, work_dir, True)
    
    return FileResponse(
        zip_path,
        filename=f"{experiment_id}_topologies.zip",
        media_type="application/zip",
        background=background_tasks,
    )