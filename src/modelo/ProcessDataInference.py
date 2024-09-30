"""
This module handles the model data.

This module does the initial processing of the data to be
used in the autoencoder. You can filter the initial
data; you can select a range for the data;
and bring the data to the same scale.

"""

import sys

import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from configs.config import settings

sys.path.append('src')


def select_variables(data):
    """_summary_.

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    for idx in range(len(data)):
        logger.info(f'variaveis selecionadas: {settings.variables}')
        data[idx] = data[idx][settings.variables]
    return data


def filter_initial_hours(data) -> list:
    """
    Cuts the number of initial hours for system stabilization.

    Cuts the number of initial hours for stabilization
    system due to the fact that the system's data
    also composed of data collected during plant heating and
    this data does not reflect the regular operating condition.

    Args:
        data (DataFrame): Data to be cut.

    Returns:
        list: Cut data
    """
    for idx in range(len(data)):
        id = list(settings.names_database.keys())[idx]

        if id == 'Referencia':
            logger.info(
                f'{settings.time_removed_reference} hrs iniciais descartadas',
            )
            data[idx] = data[idx][settings.time_removed_reference:]
            data[idx] = data[idx].reset_index(drop=True)
        else:
            logger.info(
                f'{settings.time_removed_malfunc} hrs iniciais descartadas',
            )
            data[idx] = data[idx][settings.time_removed_malfunc:]
            data[idx] = data[idx].reset_index(drop=True)

    return data


def select_interval(data) -> pd.DataFrame:
    """
    It only takes data within a range of minutes.

    The original data is given every minute, this function
    function allows you to return data at a certain interval. For example
    example: if the interval is 5, it returns the data every
    5 minutes.

    Args:
        data (Dataframe): Data every minute.

    Returns:
        DataFrame: Only the data in the selected range.
    """
    data = data[data.index % settings.interval == 0]
    logger.info(f'Selecionando data a cada {settings.interval} min')
    return data.reset_index(drop=True)


def scale_data(data) -> pd.DataFrame:
    """
    This function takes the data to the same scale.

    The original data is on different scales
    and this function applies the StandardScaler from the
    SKlearn package to bring all the data to the same
    scale.

    Args:
        data (DataFrame): Data to be transformed.

    Returns:
        DataFrame: Rescaled data.
    """
    ss = StandardScaler()
    for idx in range(len(data)):
        logger.info('Trazendo os data para a mesma escala')
        data[idx] = pd.DataFrame(
            ss.fit_transform(data[idx]),
            columns=data[idx].columns,
        )
    return data, ss


def mapping_variables():
    """_summary_.

    Returns:
        _type_: _description_
    """
    variables_pred = [
        settings.variables[idx] + '_pred'
        for idx in range(len(settings.variables))
    ]

    values = [idx for idx in range(len(settings.variables))]

    map_variables = dict(zip(values, settings.variables))
    map_pred = dict(zip(values, variables_pred))

    pred_to_variables = dict(zip(variables_pred, settings.variables))

    return variables_pred, map_pred, map_variables, pred_to_variables


def rescale_data(data, ss):
    """_summary_.

    Args:
        data (_type_): _description_
        ss (_type_): _description_

    Returns:
        _type_: _description_
    """
    (
        variables_pred,
        map_pred,
        map_variables,
        pred_to_variables,
    ) = mapping_variables()

    scaled = []

    for idx in range(len(data)):
        aux_scaled1 = data[idx][settings.variables]

        rescaled_in = pd.DataFrame(ss.inverse_transform(aux_scaled1))
        rescaled_in = rescaled_in.rename(columns=map_variables)

        aux_scaled2 = data[idx][variables_pred]

        rescaled_out = pd.DataFrame(ss.inverse_transform(aux_scaled2))
        rescaled_out = rescaled_out.rename(columns=map_pred)

        scaled.append(pd.concat(
            [
                rescaled_in[settings.variables],
                rescaled_out[variables_pred],
            ],
            axis=1,
            ),
            )
    return scaled
