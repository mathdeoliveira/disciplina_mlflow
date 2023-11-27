import structlog
import pandas as pd 
from sklearn.pipeline import Pipeline

logger = structlog.getLogger()

class DataPreprocess:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe 
        self.trained_pipe = None
     
    def train(self, dataframe: pd.DataFrame):
        logger.info('Preprocessamento iniciou...')
        self.trained_pipe = self.pipe.fit(dataframe)
        
    def transform(self, dataframe: pd.DataFrame):
        if self.trained_pipe is None:
            raise ValueError('Pipeline nao foi treinado.')
        logger.info('Transformacao dos dados com preprocessador iniciou...')
        data_processed = self.trained_pipe.transform(dataframe)
        return data_processed
