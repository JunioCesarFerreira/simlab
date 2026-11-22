import os
from pathlib import Path
from pylib.config.simulator import SimulationConfig
from pylib.cooja_builder.parse_json_pos_dat import generate_positions_from_json
from pylib.cooja_builder.replace_xml import update_simulation_xml
from pylib.cooja_builder.csv_converter import cooja_log_to_csv as _cooja_log_to_csv

def convert_simulation_files(
    config: SimulationConfig, 
    template_file: str = "simulation_template.xml",
    outsim: str = "./output/simulation.xml",
    outpos: str = "./output/positions.dat"
    ):
    """Processa a simulação completa a partir dos arquivos de configuração."""
    
    # Gera arquivo de posições e obtém a posição inicial de cada mote, na ordem
    # dos IDs: o sorvedouro (sink.c) primeiro, fixo ou móvel.
    positions, n_mobile = generate_positions_from_json(
        config["simulationElements"], 
        output_filename=outpos,
        # Cada mote móvel repete a própria trajetória até o fim da simulação.
        duration_sec=float(config["duration"]) * 60.0,
    )
    
    # Se não motes moveis remove o arquivo positions.dat pois este não é necessário
    if n_mobile == 0:
        os.remove(outpos)
    
    # O mote 1 roda o firmware da raiz, e o leitor do log a procura como Mote:1.
    root_motes = [1]
    
    # Gera arquivo XML de simulação
    update_simulation_xml(
        positions=positions,
        has_mobile=n_mobile > 0,
        root_motes=root_motes,
        simulation_time=config["duration"],
        tx_range=config["radiusOfReach"],
        interference_range=config["radiusOfInter"],
        random_seed=config.get("randomSeed", 123456),
        input_file=template_file,
        output_file=outsim
    )
    
def convert_cooja_log_to_csv(cooja_log_input: str, csv_output: str) -> None:
    _cooja_log_to_csv(Path(cooja_log_input), Path(csv_output))