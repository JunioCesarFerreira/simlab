from dto import SimulationConfig
from parse_json_pos import generate_positions_from_json
from replace_xml import update_simulation_xml 
import os

def convert_simulation_files(
    config: SimulationConfig, 
    template_file: str = "simulation_template.xml",
    outsim: str = "./output/simulation.xml",
    outpos: str = "./output/positions.dat"
    ):
    """Processa a simulação completa a partir dos arquivos de configuração."""
    
    # Gera arquivo de posições e obtém posições iniciais
    fixed_positions, mobile_start_positions = generate_positions_from_json(
        config["simulationElements"], 
        output_filename=outpos
    )
    
    # Se não motes moveis remove o arquivo positions.dat pois este não é necessário
    if mobile_start_positions is None or len(mobile_start_positions) == 0:
        os.remove(outpos)
    
    # Identifica motes servidores (assume que o primeiro fixo é o servidor)
    root_motes = [1]
    
    # Gera arquivo XML de simulação
    update_simulation_xml(
        fixed_positions=fixed_positions,
        mobile_positions=mobile_start_positions,
        root_motes=root_motes,
        simulation_time=config["duration"],
        tx_range=config["radiusOfReach"],
        interference_range=config["radiusOfInter"],
        input_file=template_file,
        output_file=outsim
    )