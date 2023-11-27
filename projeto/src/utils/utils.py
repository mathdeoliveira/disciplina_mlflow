import os 
import yaml 


def load_config_file():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    
    caminho_relativo = os.path.join('..', '..', 'config', 'config.yaml')
    
    config_file_path = os.path.abspath(os.path.join(diretorio_atual, caminho_relativo))
    
    config_file = yaml.safe_load(open(config_file_path, 'rb'))
    return config_file

def save_model(model):
    pass