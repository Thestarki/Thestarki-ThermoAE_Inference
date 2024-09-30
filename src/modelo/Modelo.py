"""
This module builds the autoencoder and runs its training.

It contains the Autoencoder class which describes the topology
topology of the neural network, as well as two functions for the
training and testing the network.
"""

import sys
from typing import Any

import numpy as np
import pandas as pd
from torch import nn, no_grad

from configs.config import check_gpu, settings
from modelo.ProcessDataInference import mapping_variables

sys.path.append('src')


class Autoencoder(nn.Module):
    """
    A service class for building the autoencoder topology.

    This class creates the input, hidden and output layers of the
    autoencoder and performs the propagation step.

    Args:
        nn (_type_): Pytorch's neural network module.

    Returns:
        _type_: The architecture of the neural network
    """

    def __init__(self) -> None:
        """Topologia do autoencoder."""
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(len(settings.variables), settings.hidden_size),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(settings.hidden_size, len(settings.variables)),
        )

    def forward(self, x_tensor) -> Any:
        """Propagation step.

        Args:
            x_tensor (tensor): Tensor for training.

        Returns:
            Any: Data after passing through the network.
        """
        encoder = self.encoder(x_tensor)
        return self.decoder(encoder)


def predict(model, val_loader):
    """_summary_.

    Args:
        model (_type_): _description_
        val_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    (
        variables_pred,
        map_pred,
        map_variables,
        pred_to_variables,
    ) = mapping_variables()

    device = check_gpu()

    model.eval()
    with no_grad():
        # List to store all the loss of an epoch
        epoch_loss = []
        predicted = pd.DataFrame()
        for batch in val_loader:
            # getting the information of the batch
            data = batch

            # casting na gpu
            data = data.to(device)

            # forward
            pred = model(data)
            loss = settings.criterion(data, pred)

            # Getting the loss
            epoch_loss.append(loss.cpu().data)

            features = pd.DataFrame(data.cpu().numpy())
            features = features.rename(columns=map_variables)

            prediction = pd.DataFrame(pred.cpu().numpy())
            prediction = prediction.rename(columns=map_pred)

            aux1 = pd.concat([features, prediction], axis=1)

            predicted = pd.concat([predicted, aux1], axis=0)

        loss_total = np.asarray(epoch_loss)
        print(
            f'Loss in the dataset: %.4f +/- %.4f'
            % (loss_total.mean(), loss_total.std()),
        )

        return predicted
