"""
Module for creating dataloaders.

This module creates the dataloaders that place the data in the
format acceptable to pytorch's neural network modules.
"""

import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from configs.config import settings

sys.path.append('src')


class ReturnTensor(Dataset):
    """
    Reads data from a csv file and transforms it into a tensor.

    This class reads the data from a csv file and returns the
    data from an existing csv file in the program's
    dependencies of the program and returns the data as tensors. The
    described here override the methods of the Dataset class.

    Arguments:
        Dataset: An abstract pytorch class representing a dataset.
        dataset. This class allows you to overwrite
        some of its internal methods, such as __init__,
        __getitem__ and __len__ by externally defined classes,
        such as this ReturnTensor class

    Methods:
        __init__: Initial constructor of the class, reads.
        of the data.
        __getitem__: function to return the items as tensors.
        __len__: returns the size of the dataframe.

    Returns:
        Dataframe: A dataframe containing the entire dataset
    """

    def __init__(self, csv_path: str) -> None:
        """
        Function to read the data.

        This function reads the data and converts it to numpy.

        Args:
            csv_path (str): Path to the data csv file
        """
        self.data = pd.read_csv(csv_path).to_numpy()
        self.variables = settings.variables

    def __getitem__(self, idx: int) -> torch.Tensor:
        """This function selects the columns of data.

        Args:
            idx (int): an index to be traversed

        Returns:
            torch.Tensor: _description_
        """
        sample = self.data[idx][0:len(self.variables)]

        return torch.from_numpy(sample.astype(np.float32))

    def __len__(self) -> int:
        """Returns the size of the Dataframe.

        Returns:
            int: Dataframe size
        """
        return len(self.data)


def write_df_validation(data) -> None:
    """
    Splits the data between training and testing.

    This function shuffles the indexes of the original dataframe,
    separates the data into 70% for training and 30% for testing and
    writes it to a csv file for training and another for testing.

    Args:
        data (Dataframe): A dataframe containing all the data
    """
    logger.info('Escrevendo os Dataframes de treino e teste')
    # Storing the information on a file
    data.to_csv(settings.path_to_val, index=False)
    logger.info('Dataframes de treino e teste escritos!')


def calling_return_tensor() -> torch.Tensor:
    """
    Calls the class that will transform the data into tensors.

    This function calls the class that transforms the data into
    tensors and returns two dataframes with the data in
    tensors.

    Returns:
        torch.Tensor: Two dataframes, one containing the tensors
        that will be used to build the DataLoader
        for training the neural network and another containing the tensors
        used to build the neural network's test DataLoader.
        for testing the neural network.
    """
    logger.info('Tranformando os data para Tensores')
    return ReturnTensor(settings.path_to_val)


def build_dataloader(data) -> DataLoader:
    """_summary_.

    Args:
        data (_type_): _description_

    Returns:
        DataLoader: _description_
    """
    val_loader = []
    for idx in range(len(data)):

        write_df_validation(data[idx])
        val_set = calling_return_tensor()

        logger.info('Criando os Dataloaders')
        val_loader.append(DataLoader(
            val_set,
            batch_size=settings.batch_size,
            shuffle=True,
        ),
        )

    logger.info('Dataloaders Criados!')

    return val_loader
