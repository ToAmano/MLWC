import numpy as np
import torch

from mlwc.include.mlwc_logger import setup_library_logger, timer_dec

logger = setup_library_logger("MLWC." + __name__)


@timer_dec
def ml_pred(model, dataloader, device):
    # lists for results
    pred_list: list = []
    true_list: list = []

    # * Test models
    with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        for data in dataloader:
            if isinstance(data[0], dict):  # data[0]がdictの場合
                for i in range(len(data[1])):
                    data_1 = [
                        {key: value[i] for key, value in data[0].items()},
                        data[1][i],
                    ]
                    x = data_1[0].to(device)
                    y = data_1[1]
                    y_pred = model(**x)
                    pred_list.append(y_pred.to("cpu").detach().numpy())
                    true_list.append(y.detach().numpy())
            elif (
                data[0].dim() == 3
            ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for x, y in zip(data[0], data[1]):
                    y_pred = model(x.to(device))
                    pred_list.append(y_pred.to("cpu").detach().numpy())
                    true_list.append(y.detach().numpy())
            elif data[0].dim() == 2:  # 2次元の場合はそのまま(batch = 1)
                # self.batch_step(data,validation=True)
                x = data[0]
                y = data[1]
                y_pred = model(x)
                pred_list.append(y_pred.to("cpu").detach().numpy())
                true_list.append(y.detach().numpy())
    #
    pred_list = np.array(pred_list).reshape(-1, 3)
    true_list = np.array(true_list).reshape(-1, 3)
    # calculate RSME
    # rmse = np.sqrt(np.mean((true_list - pred_list) ** 2))
