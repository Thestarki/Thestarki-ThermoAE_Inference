"""Modulo para salvar os dados do modelo."""

from configs.config import settings


def save_data(data):
    """_summary_.

    Args:
        data (_type_): _description_
    """
    for idx in range(len(data)):
        data[idx].to_csv(
            settings.path_to_predicted
            + list(settings.names_database.keys())[idx]
            + '.csv',
            index=False,
        )
