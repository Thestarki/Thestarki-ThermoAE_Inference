"""_summary_."""

import torch

from configs.config import check_gpu, settings
from modelo.BuildDataloaderInf import build_dataloader
from modelo.Modelo import predict
from modelo.ProcessDataInference import (filter_initial_hours,
                                         mapping_variables, rescale_data,
                                         scale_data, select_interval,
                                         select_variables)
from modelo.ReadDataInference import read_data
from modelo.SaveDataInference import save_data

data = read_data()

data = select_variables(data)

data = filter_initial_hours(data)

data[0] = select_interval(data[0])

data, ss = scale_data(data)

data = build_dataloader(data)

variables_pred, map_pred, map_variables, pred_to_variables = mapping_variables()

device = check_gpu()

model = torch.load(settings.path_to_model, weights_only=False)

predicted = []
for idx in range(len(data)):
    predicted.append(predict(model, data[idx]))

predicted = rescale_data(predicted, ss)

save_data(predicted)
