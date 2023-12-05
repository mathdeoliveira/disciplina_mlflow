import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import pandas as pd
import structlog

logger = structlog.getLogger()
from utils.utils import load_config_file


class DataLoad:
    """Class data load"""

    def __init__(self) -> None:
        pass

    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """Carrega os dados a partir do nome do dataset fornecido

        Args:
            dataset_name (str): O nome do dataset a ser carregado

        Returns:

        Raises:
        """
        logger.info(f"Comecando a carga dos dados com o nome: {dataset_name}")
        dataset = load_config_file().get(dataset_name)

        try:
            dataset = load_config_file().get(dataset_name)
            if dataset is None:
                raise ValueError(
                    f"Erro: O nome do dataset fornecido e incorreto: {dataset}"
                )
            loaded_data = pd.read_csv(f"../data/raw/{dataset}")
            return loaded_data[load_config_file().get("columns_to_use")]
        except ValueError as ve:
            logger.error(str(ve))
        except Exception as e:
            logger.error(f"Erro inesperado: {str(e)}")
