"""This module contains all the settings for the model."""
import os
from typing import Any

import torch
from pydantic import FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from torch import nn

from db.db_config import Mf1, Mf2, Mf3, MfBoi, MfHx, ReferenceData

dotenv = os.path.join(os.path.dirname(__file__), '.env')


class Settings(BaseSettings):
    """
    A class to collect all the adjustable parameters of the autoencoder.

    This class configures the entire model developed.
    parameters in the project.

    Attributes:

        batch_size: batch size used for training
        input_size: number of neurons in the input layer of the network
        hidden_size: size of the network's hidden layer
        out_size: size of the network's output layer
        epochs: number of times the data will pass through the network
    """

    model_config = SettingsConfigDict(env_file=dotenv, env_file_encoding='utf-8')

    # Neural Network configuration
    batch_size: int

    hidden_size: int

    epochs: int

    criterion: Any = nn.MSELoss()

    log_level: str

    variables: list

    time_removed_reference: int

    time_removed_malfunc: int

    interval: int

    names_database: dict = {
        'Referencia': ReferenceData,
        'Mf1': Mf1,
        'Mf2': Mf2,
        'Mf3': Mf3,
        'Mfboi': MfBoi,
        'MfHx': MfHx,
    }

    path_to_model: FilePath
    path_to_db: FilePath
    path_to_val: str
    path_to_predicted: str


def check_gpu() -> torch.device:
    """Function to check if gpu is available.

    Returns:
        _type_: Returns the device used
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device('cuda')
    else:
        return torch.device('cpu')


settings = Settings()
engine = create_engine(os.path.join('sqlite:///', settings.path_to_db))
