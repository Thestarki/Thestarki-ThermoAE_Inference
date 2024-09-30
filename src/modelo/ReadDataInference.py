"""
Este módulo carrega os dados de treinamento do Autoencoder.

Ele contém duas funções que possibilitam carregar os dados
de duas formas diferentes.
"""

import sys

import pandas as pd
from loguru import logger
from sqlalchemy import select

from configs.config import engine, settings

sys.path.append('src')


def read_db(scheema, key) -> pd.DataFrame:
    """
    Read the data from the database.

    This function returns the reference data from a
    SQLite database and records in the logger whether it was executed.
    The data is read via sqlalchemy by selecting a schema
    and an engine.

    Args:
        scheema (BaseSettings): Database schema that will return the dataframe.
        key (str): Name of the database.

    Returns:
        pd.DataFrame: A dataframe containing the entire dataset
    """
    logger.info(f'Carregando da base de dados {key}')
    query = select(scheema)
    return pd.read_sql(query, engine)


def read_data():
    """_summary_.

    Returns:
        _type_: _description_
    """
    data = []
    for idx in range(len(settings.names_database.keys())):
        data.append(
            read_db(
                list(settings.names_database.values())[idx],
                list(settings.names_database.keys())[idx],
            ),
        )
    return data
