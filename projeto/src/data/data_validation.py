import os 
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd 
import pandera 
from pandera import Check, Column, DataFrameSchema
import structlog

logger = structlog.getLogger()

from utils.utils import load_config_file

class DataValidation:
    """Classe de validacao dos dados
    
    """
    def __init__(self) -> None:
        self.columns_to_use = load_config_file().get('columns_to_use')
        
    def check_shape_data(self, dataframe: pd.DataFrame) -> bool:
        try:
            logger.info('Validacao iniciou')
            dataframe.columns = self.columns_to_use
            return True 
        except Exception as e:
            logger.error(f'Validacao errou: {e}')
            return False
        
    def check_columns(self, dataframe: pd.DataFrame) -> bool:
        schema = DataFrameSchema(
                {
                    "target": Column(int, Check.isin([0, 1]), Check(lambda x: x > 0), coerce=True),
                    "TaxaDeUtilizacaoDeLinhasNaoGarantidas": Column(float, nullable=True),
                    "Idade": Column(int, nullable=True),
                    "NumeroDeVezes30-59DiasAtrasoNaoPior": Column(int, nullable=True),
                    "TaxaDeEndividamento": Column(float, nullable=True),
                    "RendaMensal": Column(float, nullable=True),
                    "NumeroDeLinhasDeCreditoEEmprestimosAbertos": Column(int, nullable=True),
                    "NumeroDeVezes90DiasAtraso": Column(int, nullable=True),
                    "NumeroDeEmprestimosOuLinhasImobiliarias": Column(int, nullable=True),
                    "NumeroDeVezes60-89DiasAtrasoNaoPior": Column(int, nullable=True),
                    "NumeroDeDependentes": Column(float, nullable=True)
                }
            )
        try:
            schema.validate(dataframe)
            logger.info("Validation columns passed...")
            return True
        except pandera.errors.SchemaErrors as exc:
            logger.error("Validation columns failed...")
            pandera.display(exc.failure_cases)
        return False
    
    def run(self, dataframe: pd.DataFrame) -> bool:
        if self.check_shape_data(dataframe) and self.check_columns(dataframe):
            logger.info('Validacao com sucesso.')
            return True 
        else:
            logger.error('Validacao falhou.')
            return False
