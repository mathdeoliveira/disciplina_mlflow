import os 
import sys 
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import structlog
import pandas as pd

logger = structlog.getLogger()
from utils.utils import load_config_file


class TrainModels:
    def __init__(self, dados_X: pd.DataFrame,
                       dados_y: pd.DataFrame):
        self.dados_X = dados_X 
        self.dados_y = dados_y
        self.model_name = load_config_file().get('model_name')
        
    def train(self, model):
        model.fit(self.dados_X, self.dados_y)
        joblib.dump(model, self.model_name)
        return model 
