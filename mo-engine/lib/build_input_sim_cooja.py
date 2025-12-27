import os, tempfile
from pathlib import Path
from pylib import cooja_files
from pylib.dto.simulator import SimulationConfig
from pylib.mongo_db import MongoGridFSHandler

TEMPLATE_XML = os.getenv("TEMPLATE_XML", "./simulation_template.xml")
IS_DOCKER = os.getenv("IS_DOCKER", False)

def create_files(sim_config: SimulationConfig, grid_fs: MongoGridFSHandler)-> dict[str, str]: 
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_xml = tmp_path / "simulation.xml"
        out_dat = tmp_path / "positions.dat"       

        # Gera os arquivos a partir do template
        cooja_files.convert_simulation_files(sim_config, TEMPLATE_XML, out_xml, out_dat)

        # Envia para o GridFS
        xml_id = str(grid_fs.upload_file(str(out_xml), "simulation.xml"))
        
        if os.path.exists(out_dat):
            dat_id = str(grid_fs.upload_file(str(out_dat), "positions.dat"))
        else:
            dat_id = ""

        if IS_DOCKER:
            os.remove(out_xml)
            if os.path.exists(out_dat):
                os.remove(out_dat)

        return {
            "csc_file_id": xml_id,
            "pos_file_id": dat_id
        }