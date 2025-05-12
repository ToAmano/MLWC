import inspect
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Subset

import mlwc.ml.dataset.mldataset_xyz
import mlwc.ml.loss.ml_loss
from mlwc.include.mlwc_logger import get_default_log_file_name, setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


class Trainer:
    def __init__(
        self,
        model,
        # TODO :: implement for mps (apple silicon)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
        validation_batch_size: int = 32,
        max_epochs: int = 1000000,
        learning_rate: dict = {
            "type": "MultiStepLR",
            "milestones": [1000, 1000],
            "gamma": 0.1,
            "start_lr": 0.01,
        },
        lr_scheduler_name: str = "none",
        lr_scheduler_kwargs: Optional[dict] = None,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        modeldir: str = "./",
        restart: Optional[bool] = False,
    ):

        # import instance variables
        self.model = model
        self.device: str = device
        self.batch_size: int = batch_size
        self.validation_batch_size: int = validation_batch_size
        self.max_epochs: int = max_epochs
        self.learning_rate: dict = learning_rate
        self.lr_scheduler_name: str = lr_scheduler_name
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.optimier_name = optimizer_name
        self.optimier_kwargs = optimizer_kwargs
        self.n_train = n_train
        self.n_val = n_val
        self.modeldir = modeldir  # to which a model is saved
        self.restart: bool = restart  # True for restarting

        # other instance variables
        self.valid_rmse_list = []
        self.train_rmse_list = []
        self.valid_loss_list = []
        self.train_loss_list = []
        self.steps: int = 0  # total steps
        self.iepoch: int = 0  # total epochs
        self.best_epoch = 0

        # related to previous run
        self.previous_maxstep: int = -1

        # batch loss
        # TODO :: ここは完全にクラス化してもっと洗練された実装にしておきたい
        self.epoch_valid_loss = []
        self.epoch_train_loss = []

        # if not exist self.modeldir, mkdir it
        if not os.path.isdir(self.modeldir):
            os.makedirs(self.modeldir)
        self.logger.info(f"model data will be saved to {self.modeldir}")

        # generator
        self.dataset_rng = torch.Generator()

        # initialize training states
        self.best_metrics = float("inf")
        self.best_epoch = 0
        self.iepoch = 0  # -1 if self.report_init_validation else 0

        # print modelinfo
        from torchinfo import summary

        logger.info(summary(model=self.model))

        # set loss function
        self.lossfunction = nn.MSELoss()
        # model initialize (move to device)
        self.init_model()

        # optimizer/scheduler
        self.init_optimizer_scheduler()

        # load loss/RMSE logger
        self.loss_log = mlwc.ml.loss.ml_loss.LossStatistics(self.modeldir)

        # load previous run information
        if self.restart == True:
            self.get_previous_info()
            self.read_from_previous_run()  # 既存ファイルがある場合，前回の結果を読み出し
            # self.iepoch =

    @property
    def logger(self):
        """set logging name to Trainer"""
        return setup_library_logger("MLWC.Trainer")

    @property
    def epoch_logger(self):
        # return logging.getLogger(self.epoch_log)
        return setup_library_logger(self.epoch_log)

    @property
    def init_epoch_logger(self):
        # return logging.getLogger(self.init_epoch_log)
        return setup_library_logger(self.init_epoch_log)

    def init_model(self):
        self.logger.info(f"Torch device (cpu or cuda gpu or m1 mac gpu): {self.device}")
        self.model = self.model.to(self.device)  # move to device

    def init_optimizer_scheduler(self):

        # Setting optimizer
        # TODO :: In nequip, instantiate_from_cls_name function is used in init_objects (trainer.py)
        torch.backends.cudnn.benchmark = True
        # Optimization algorithm:: We recommend adam (adagrad was not good in our experiments.)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), float(self.learning_rate["start_lr"])
        )

        # set scheduler ( for dynamic change of learning rate)
        # see https://take-tech-engineer.com/pytorch-lr-scheduler/
        # get scheduler function name
        scheduler_type = self.learning_rate["type"]
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
        # get parametes of the scheduler
        scheduler_params = inspect.signature(scheduler_class).parameters
        # extract valid parameters from input
        valid_params = {
            k: v
            for k, v in self.learning_rate.items()
            if k in scheduler_params and v != "type" and v != "start_lr"
        }
        print(f" valid_params for scheduler :: {valid_params}")
        # define scheduler
        self.scheduler = scheduler_class(self.optimizer, **valid_params)
        print(f" scheduler :: {self.scheduler}")

    def set_dataset(
        self,
        dataset: mlwc.ml.dataset.mldataset_xyz.DataSet_xyz,
        validation_dataset: Optional[mlwc.ml.dataset.mldataset_xyz.DataSet_xyz] = None,
    ):
        # total length of dataset
        total_n = len(dataset)
        # 元のデータセットから，訓練データ数，validationデータ数に応じたデータを取り出す．
        if self.n_train is None or self.n_val is None:
            self.logger.warning(" n_train or n_val is not set.")
            self.logger.warning(
                " automatically 10% for validation and 90% for training "
            )
            self.n_train = int(0.9 * total_n)
            self.n_val = int(0.1 * total_n)

        if (
            validation_dataset is None
        ):  # val_datasetがない場合はdatasetから両方サンプルする
            if (self.n_train + self.n_val) > total_n:
                raise ValueError(
                    "too little data for training and validation. please reduce n_train and n_val"
                )
            # generate random index for train/val
            idcs = torch.randperm(total_n, generator=self.dataset_rng)

            self.train_idcs = idcs[: self.n_train]
            self.val_idcs = idcs[self.n_train : self.n_train + self.n_val]
        else:  # validation_datasetがあれば別々にsampleする
            if self.n_train > len(dataset):
                raise ValueError("Not enough data in dataset for requested n_train")
            if self.n_val > len(validation_dataset):
                raise ValueError(
                    "Not enough data in validation dataset for requested n_val"
                )

            self.train_idcs = torch.randperm(len(dataset), generator=self.dataset_rng)[
                : self.n_train
            ]
            self.val_idcs = torch.randperm(
                len(validation_dataset), generator=self.dataset_rng
            )[: self.n_val]

        if validation_dataset is None:
            validation_dataset = dataset

        self.logger.info(f" n_traing ( number of training  data): {self.n_train}")
        self.logger.info(f" n_val    ( number of validatin data): {self.n_val}")

        # torch_geometric datasets inherantly support subsets using `index_select`
        # self.dataset_train = dataset.index_select(self.train_idcs)
        # self.dataset_val = validation_dataset.index_select(self.val_idcs)

        self.dataset_train = Subset(dataset, self.train_idcs)
        self.dataset_valid = Subset(validation_dataset, self.val_idcs)

        # dataset_train, dataset_valid = torch.utils.data.random_split(dataset=dataset, lengths=[len(dataset)-10000, 10000], generator=torch.Generator().manual_seed(42))

        # dataloader
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
        )
        self.dataloader_valid = torch.utils.data.DataLoader(
            self.dataset_valid,
            batch_size=self.validation_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=0,
        )

    def read_from_previous_run(self):
        cptfile = f"{self.modeldir}/model_{self.model.modelname}_out_tmp{self.previous_maxstep}.cpt"
        if os.path.isfile(cptfile) == True:
            self.logger.info(" -------------------------------------- ")
            self.logger.info(" cpt file exist :: load previous data !!")
            self.logger.info(" -------------------------------------- ")
            cpt = torch.load(cptfile)
            stdict_m = cpt["model_state_dict"]
            stdict_o = cpt["opt_state_dict"]
            stdict_s = cpt["scheduler_state_dict"]
            self.model.load_state_dict(stdict_m)
            self.optimizer.load_state_dict(stdict_o)
            self.scheduler.load_state_dict(stdict_s)

    def get_previous_info(self):
        filenames = [
            int(
                f.name.removeprefix(
                    f"model_{self.model.modelname}_out_tmp"
                ).removesuffix(".cpt")
            )
            for f in os.scandir(self.modeldir)
            if f.is_file() and f"model_{self.model.modelname}_out_tmp" in f.name
        ]
        self.logger.info(filenames)
        self.previous_maxstep = np.max(np.array(filenames))
        self.logger.info(f"Previous run goes to {self.previous_maxstep} step")
        # !! update iepoch
        self.iepoch = self.previous_maxstep

    def train(self):
        # 実際のtrainingを行う場所
        # 個々の部品は別途定義してある

        # TODO ここも実装
        # if getattr(self, "dl_train", None) is None:
        #     raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        # if not self._initialized:
        #     self.init()

        # self.init_log()
        # self.wall = perf_counter()
        # self.previous_cumulative_wall = self.cumulative_wall
        # self.init_metrics()

        # check initial loss
        self.initial_loss()

        # Perform training
        while not self.stop_condition:
            self.epoch_step()  # epoch_step includes batch_step
            self.end_of_epoch_save()
        # for callback in self._final_callbacks:
        #    callback(self)
        # self.final_log()
        # self.save()
        # finish_all_writes()

        # save all the models
        self.save_model_all()

    def initial_loss(self) -> int:
        # validation
        self.model.eval()
        with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
            for data in self.dataloader_valid:
                self.logger.debug("start batch valid")
                if isinstance(data[0], dict):  # data[0]がdictの場合
                    for i in range(self.validation_batch_size):
                        data_1 = [
                            {key: value[i] for key, value in data[0].items()},
                            data[1][i],
                        ]
                        self.batch_step(data_1, validation=True)
                elif (
                    data[0].dim() == 3
                ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                    # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                    for data_1 in zip(data[0], data[1]):
                        self.logger.debug(
                            f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}"
                        )
                        self.batch_step(data_1, validation=True)
                elif data[0].dim() == 2:  # 2次元の場合はそのまま
                    self.batch_step(data, validation=True)

        # バッチ全体でLoss値(のroot，すなわちRSME)を平均する
        # TODO :: ここはもう少し良い実装を考えたい
        self.logger.debug(
            f" number of n_train/batch size ( iteration number of each step): {int(self.n_train/self.batch_size)} {int(self.n_val/self.validation_batch_size)}"
        )
        ave_loss_valid = np.mean(
            np.array(
                self.valid_loss_list[-int(self.n_val / self.validation_batch_size) :]
            )
        )
        # Average loss in epoch
        self.epoch_valid_loss.append(ave_loss_valid)
        return 0

    def epoch_step(self):
        """
        1 epochのtrain/validationを行う．
        すなわち，dataloaderにあるデータをすべて使って推論する．
        """

        # 時間計測
        start_time = time.time()  # 現在時刻（処理開始前）を取得

        # training
        self.model.train()  # モデルを学習モードに変更
        for data in self.dataloader_train:
            self.logger.debug("start batch train")
            if isinstance(data[0], dict):  # data[0]がdictの場合
                for i in range(len(data[1])):  # FIXME:: ここか？
                    print(f" batch step == {i}")
                    data_1 = [
                        {key: value[i] for key, value in data[0].items()},
                        data[1][i],
                    ]
                    self.batch_step(data_1, validation=False)
            elif (
                data[0].dim() == 3
            ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                for data_1 in zip(data[0], data[1]):
                    self.logger.debug(
                        f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}"
                    )
                    self.batch_step(data_1, validation=False)
            elif data[0].dim() == 2:  # 2次元の場合はそのまま
                # print("start batch train")
                self.batch_step(data, validation=False)

        # validation
        self.model.eval()  # モデルを推論モードに変更 (BN)
        with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
            for data in self.dataloader_valid:
                self.logger.debug("start batch valid")
                if isinstance(data[0], dict):  # data[0]がdictの場合
                    for i in range(self.validation_batch_size):
                        data_1 = [
                            {key: value[i] for key, value in data[0].items()},
                            data[1][i],
                        ]
                        self.batch_step(data_1, validation=True)
                elif (
                    data[0].dim() == 3
                ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                    # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                    for data_1 in zip(data[0], data[1]):
                        self.logger.debug(
                            f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}"
                        )
                        self.batch_step(data_1, validation=True)
                elif data[0].dim() == 2:  # 2次元の場合はそのまま
                    self.batch_step(data, validation=True)

        # バッチ全体でLoss値(のroot，すなわちRSME)を平均する
        # TODO :: ここはもう少し良い実装を考えたい
        self.logger.debug(
            f" number of n_train/batch size ( iteration number of each step): {int(self.n_train/self.batch_size)} {int(self.n_val/self.validation_batch_size)}"
        )
        ave_rmse_train = np.mean(
            np.array(self.train_rmse_list[-int(self.n_train / self.batch_size) :])
        )
        ave_rmse_valid = np.mean(
            np.array(
                self.valid_rmse_list[-int(self.n_val / self.validation_batch_size) :]
            )
        )
        ave_loss_train = np.mean(
            np.array(self.train_loss_list[-int(self.n_train / self.batch_size) :])
        )
        ave_loss_valid = np.mean(
            np.array(
                self.valid_loss_list[-int(self.n_val / self.validation_batch_size) :]
            )
        )
        # Average loss in epoch
        self.epoch_valid_loss.append(ave_loss_valid)
        self.epoch_train_loss.append(ave_loss_train)
        # timer
        end_time = time.time()  # 現在時刻（処理完了後）を取得
        time_diff = (
            end_time - start_time
        )  # 処理完了後の時刻から処理開始前の時刻を減算する
        self.logger.info(
            f"epoch= {self.iepoch+1} : time= {time_diff:.2f} [s] : lr= {self.optimizer.param_groups[0]['lr']:6f} : loss(train)= {ave_loss_train:.5f} : loss(valid)= {ave_loss_valid:.5f} : RMSE[D](train)= {ave_rmse_train:.5f} : RMSE[D](valid)= {ave_rmse_valid:.5f}"
        )

        # update scheduler (learning rate)
        self.scheduler.step()

        # update epoch step
        self.iepoch += 1

        # self.end_of_epoch_log()

    def end_of_epoch_save(self) -> None:
        """
        save model and trainer details at each epoch ( for restarting)
        """
        # モデルの一時保存
        if self.previous_maxstep < 0:  # 前回から読み込まない場合
            torch.save(
                self.model.state_dict(),
                f"{self.modeldir}/model_{self.model.modelname}_weight_tmp_{str(self.iepoch)}.pth",
            )
        else:  # 前回から読み込む場合
            torch.save(
                self.model.state_dict(),
                f"{self.modeldir}/model_{self.model.modelname}_weight_tmp_{str(self.iepoch+self.previous_maxstep)}.pth",
            )

        # 学習状態の一時保存
        torch.save(
            {
                "iter": self.iepoch,
                "model_state_dict": self.model.state_dict(),
                "opt_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss": self.train_loss_list,
            },
            self.modeldir
            + "/model_"
            + self.model.modelname
            + "_out_tmp"
            + str(self.iepoch)
            + ".cpt",
        )
        # print("model is saved !! ", self.modeldir+'/model_'+self.model.modelname+'_out_tmp'+str(self.iepoch)+'.cpt')

        # C++ version model save
        # TODO :: using non-method function
        save_model_cc(
            self.model,
            modeldir=self.modeldir,
            name=self.model.modelname + "_tmp" + str(self.iepoch),
        )
        # >>> end

    def batch_step(self, data, validation: bool = False) -> None:
        """
        data:: これが実際に計算するデータで，dataloaderから引っ張ってきたもの
        """
        if validation:
            self.model.eval()
        else:
            self.model.train()

        # datasetは[x,y]の形で返すようになっている
        x = move_dict_to_device(data[0], self.device)
        # x = data[0].to(self.device)
        y = data[1].to(self.device)
        self.model.to(self.device)
        if not validation:  # training
            # True であればOK
            # 勾配情報を0に初期化, https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
            self.optimizer.zero_grad()
            if isinstance(x, dict):
                y_pred = self.model(**x)  # prediction
            else:
                y_pred = self.model(x)

            # calculate loss (reshape to y)
            loss = self.lossfunction(y_pred.reshape(y.shape), y)
            try:
                loss.backward()  # 勾配の計算
            except:
                print(f"y_pred: {y_pred[0]}")
                print(f"y_true: {y[0]}")
                raise ValueError("FALSEFALSE!!")
            self.optimizer.step()  # 勾配の更新
            # self.optimizer.zero_grad()
            # self.scheduler.step()                        # !! update learning rate (at each batch)
            self.train_rmse_list.append(np.sqrt(loss.item()))
            self.train_loss_list.append(loss.item())
            # logging rmse
            self.loss_log.add_train_batch_loss(loss.item(), self.iepoch)
            del loss  # 誤差逆伝播を実行後、計算グラフを削除

        else:  # validation
            with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
                if isinstance(x, dict):
                    y_pred = self.model(**x)  # prediction
                else:
                    y_pred = self.model(x)
                loss = self.lossfunction(
                    y_pred.reshape(y.shape), y
                )  # 損失を計算(shapeを揃える)
                # np_loss = np.sqrt(np.mean((y_pred.to("cpu").detach().numpy()-y.detach().numpy())**2))  #損失のroot，RSMEと同じ
                # logging rmse
                self.loss_log.add_valid_batch_loss(loss.item(), self.iepoch)
                self.valid_rmse_list.append(np.sqrt(loss.item()))
                self.valid_loss_list.append(loss.item())
        # >>>> FINISH FUNCTION

    def save_model_all(self):
        """
        モデルを全て保存する．
        """
        # モデルの重み保存
        print(
            " model is saved to {} at {}".format(
                "model_" + self.model.modelname + "_weight.pth", self.modeldir
            )
        )
        torch.save(
            self.model.state_dict(),
            self.modeldir + "/model_" + self.model.modelname + "_weight.pth",
        )  # fin
        # モデル全体保存
        # https://take-tech-engineer.com/pytorch-model-save-load/#toc3
        print(
            " model is saved to {} at {}".format(
                "model_" + self.model.modelname + "_all.pth", self.modeldir
            )
        )
        torch.save(
            self.model, self.modeldir + "/model_" + self.model.modelname + "_all.pth"
        )
        # python用のtorch scriptを保存
        torch.jit.script(self.model).save(
            self.modeldir + "/model_" + self.model.modelname + "_torchscript.pt"
        )
        # c++用のtorch scriptを保存
        self.save_model_cc_script()
        return 0

    def save_model_cc(self):
        """
        C++用にモデルを保存する関数
        """
        # 学習時の入力サンプル
        device = "cpu"
        example_input = torch.rand(1, self.model.nfeatures).to(
            device
        )  # model.nfeatures=288

        # 学習済みモデルのトレース
        model_tmp = self.model.to(
            device
        )  # model自体のdeviceを変えないように別変数に格納
        model_tmp.eval()  # evaluation mode
        traced_net = torch.jit.trace(model_tmp, example_input)
        # save the model
        print(
            " model is saved to {} at {}".format(
                "model_" + self.model.modelname + ".pt", self.modeldir
            )
        )
        traced_net.save(self.modeldir + "/model_" + self.model.modelname + ".pt")
        # model move to device (for next step)
        self.model.to(self.device)
        return 0

    def save_model_cc_script(self):
        """save torchscript model to C++ using scripting

        Returns:
            _type_: _description_
        """
        # 学習時の入力サンプル
        device = "cpu"
        example_input = torch.rand(1, self.model.nfeatures).to(
            device
        )  # model.nfeatures=288

        # 学習済みモデルのトレース
        model_tmp = self.model.to(
            device
        )  # model自体のdeviceを変えないように別変数に格納
        model_tmp.eval()  # ちゃんと推論モードにする！！
        traced_net = torch.jit.script(model_tmp)
        # print(traced_net.code)
        # print(traced_net.nfeatures)
        # 変換モデルの出力
        print(
            " model is saved to {} at {}".format(
                "model_" + self.model.modelname + ".pt", self.modeldir
            )
        )
        traced_net.save(self.modeldir + "/model_" + self.model.modelname + ".pt")
        # modelをgpuへ再度戻す
        self.model.to(self.device)
        return 0

    def save_prediction_result():
        print("TEST")

    @property
    def stop_condition(self):
        """
        学習を止めるかどうかの判定に用いる．
        現状はepoch数がmaxに達したら停止する．
        """
        if self.iepoch >= self.max_epochs:
            self.stop_arg = "max epochs"
            return True
        return False

    def validate_model(self):

        # pred, trueのリストを作成
        pred_list = []
        true_list = []

        # * Test by models
        start_time = time.perf_counter()  # start time check
        self.model.eval()  # model to evaluation mode
        with torch.no_grad():  # https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
            for data in self.dataloader_valid:
                # self.logger.debug("start batch valid")
                if (
                    data[0].dim() == 3
                ):  # 3次元の場合[NUM_BATCH,NUM_BOND,288]はデータを整形する
                    # TODO :: torch.reshape(data[0], (-1, 288)) does not work !!
                    for data_1 in zip(data[0], data[1]):
                        # self.logger.debug(f" DEBUG :: data_1[0].shape = {data_1[0].shape} : data_1[1].shape = {data_1[1].shape}")
                        # self.batch_step(data_1,validation=True)
                        # modve descriptor to device
                        x = data_1[0].to(self.device)
                        y = data_1[1]
                        y_pred = self.model(x)
                        pred_list.append(y_pred.to("cpu").detach().numpy())
                        true_list.append(y.detach().numpy())
                if data[0].dim() == 2:  # 2次元の場合はそのまま
                    # self.batch_step(data,validation=True)
                    x = data_1[0]
                    y = data_1[1]
                    y_pred = self.model(x)
                    pred_list.append(y_pred.to("cpu").detach().numpy())
                    true_list.append(y.detach().numpy())
                # lossを計算?
                np_loss = np.sqrt(
                    np.mean(
                        (y_pred.to("cpu").detach().numpy() - y.detach().numpy()) ** 2
                    )
                )  # 損失のroot，RSMEと同じ
        #
        pred_list = np.array(pred_list).reshape(-1, 3)
        true_list = np.array(true_list).reshape(-1, 3)
        end_time = time.perf_counter()  # 計測終了
        # RSMEを計算する
        rmse = np.sqrt(np.mean((true_list - pred_list) ** 2))
        from sklearn.metrics import r2_score

        # save results
        self.logger.info(" ======")
        self.logger.info("  Finish testing.")
        self.logger.info("  Save results as pred_true_list.txt")
        self.logger.info(f" RSME_train = {rmse}")
        self.logger.info(f" r^2        = {r2_score(true_list,pred_list)}")
        self.logger.info(" ")
        self.logger.info(" ELAPSED TIME  {:.2f}".format((end_time - start_time)))
        self.logger.info(np.shape(pred_list))
        self.logger.info(np.shape(true_list))
        np.savetxt("pred_list.txt", pred_list)
        np.savetxt("true_list.txt", true_list)
        # make figures
        make_figure(pred_list, true_list)
        plot_residure_density(pred_list, true_list)


def move_dict_to_device(data, device):
    if isinstance(data, dict):
        return {key: move_dict_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):  # リスト内のTensorも移動
        return [move_dict_to_device(item, device) for item in data]
    elif isinstance(data, tuple):  # タプルの場合
        return tuple(move_dict_to_device(item, device) for item in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data  # Tensor 以外はそのまま


def make_figure(pred_list: np.array, true_list: np.array) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    # calculate RSME
    rmse = np.sqrt(np.mean((true_list - pred_list) ** 2))
    print(" RSME = {0}".format(rmse))
    # plot figure
    # figure, axesオブジェクトを作成
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    scatter1 = ax.scatter(
        np.linalg.norm(pred_list, axis=1),
        np.linalg.norm(true_list, axis=1),
        alpha=0.2,
        s=5,
        label="RMSE={}".format(rmse),
    )
    # 各要素で設定したい文字列の取得
    xticklabels = ax.get_xticklabels()
    yticklabels = ax.get_yticklabels()
    xlabel = "ML predicted dipole [D]"
    ylabel = "DFT simulated dipole [D]"
    # 各要素の設定を行うsetコマンド
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    # ax.set_xlim(0,3)
    # ax.set_ylim(0,3)
    ax.grid()
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    # ax.legend = ax.legend(*scatter.legend_elements(prop="colors"),loc="upper left", title="Ranking")
    lgnd = ax.legend(loc="upper left", fontsize=20)
    for handle in lgnd.legendHandles:
        handle.set_sizes([30])
        handle.set_alpha([1.0])
    fig.savefig("pred_true_norm.png")
    # FINISH FUNCTION


def calculate_gaussian_kde(
    data_x: np.array, data_y: np.array
) -> Tuple[np.array, np.array, np.array]:
    """calculate gaussian kde using scipy.stats.gaussian_kde

    Args:
        data_x (np.array): _description_
        data_y (np.array): _description_

    Returns:
        np.array, np.array, np.array: _description_
    """

    # https://runtascience.hatenablog.com/entry/2021/05/06/%E3%80%90Matplotlib%E3%80%91python%E3%81%A7%E5%AF%86%E5%BA%A6%E3%83%97%E3%83%AD%E3%83%83%E3%83%88%28Density_plot%29
    from scipy.stats import gaussian_kde

    # KDE probability
    x = data_x
    y = data_y
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    # zの値で並び替え→x,yも並び替える
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    return x, y, z


def plot_residure_density(pred_list: np.array, true_list: np.array, limit: bool = True):
    """
    学習結果をplotする関数．
    こちらではtrain/validの区別なく，全てのデータをまとめて，代わりにdensity mapで表示する
    """
    import matplotlib.pyplot as plt
    import numpy as np

    print(" ========= ")
    print(" calculate density map (takes a few minutes)")
    print(" ")
    print(" ")

    # calculate RMSE
    rmse = np.sqrt(np.mean((true_list - pred_list) ** 2))
    print(" RSME_train = {0}".format(rmse))

    # if the number of data is too large, limit the number of data
    if len(pred_list) > 10000:
        random_index = np.random.choice(len(pred_list), size=10000, replace=False)
        pred_list = np.array(pred_list)[random_index]
        true_list = np.array(true_list)[random_index]

    # matplotlibで複数のプロットをまとめる．
    # https://python-academia.com/matplotlib-multiplegraphs/
    # グラフを表示する領域を，figオブジェクトとして作成。
    fig = plt.figure(figsize=(15, 5), facecolor="lightblue")

    # グラフを描画するsubplot領域を作成。
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # 各subplot領域にデータを渡す
    # KDE probability
    x, y, z = calculate_gaussian_kde(pred_list[:, 0], true_list[:, 0])
    im = ax1.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    x, y, z = calculate_gaussian_kde(pred_list[:, 1], true_list[:, 1])
    im = ax2.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    x, y, z = calculate_gaussian_kde(pred_list[:, 2], true_list[:, 2])
    im = ax3.scatter(x, y, c=z, s=50, cmap="jet")
    fig.colorbar(im)

    # タイトル
    ax1.set_title("Dipole_x")
    ax2.set_title("Dipole_y")
    ax3.set_title("Dipole_z")

    # 各subplotにxラベルを追加
    ax1.set_xlabel("ML dipole [D]")
    ax2.set_xlabel("ML dipole [D]")
    ax3.set_xlabel("ML dipole [D]")

    ax1.set_ylabel("DFT dipole [D]")
    ax2.set_ylabel("DFT dipole [D]")
    ax3.set_ylabel("DFT dipole [D]")

    # 凡例表示
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    ax3.legend(loc="upper left")

    # grid表示
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    fig.savefig("pred_true_density.png")
    return 0


def save_model_cc(model, modeldir="./", name="cc"):
    """
    C++用にモデルを保存する関数
    """
    # 学習時の入力サンプル
    device = "cpu"
    example_input = torch.rand(1, model.nfeatures).to(device)  # model.nfeatures=288

    # 学習済みモデルのトレース
    model_tmp = model.to(device)  # model自体のdeviceを変えないように別変数に格納
    model_tmp.eval()  # ちゃんと推論モードにする！！
    # traced_net = torch.jit.trace(model_tmp, example_input)
    traced_net = torch.jit.script(model_tmp)
    # 変換モデルの出力
    print(" model is saved to {} at {}".format("model_" + name + ".pt", modeldir))
    traced_net.save(modeldir + "/model_" + name + ".pt")
    return 0


def save_model_all(model, modeldir: str, name: str = "ch"):
    """
    モデルを全て保存する．
    """
    # モデルの重み保存
    print(
        " model is saved to {} at {}".format("model_" + name + "_weight.pth", modeldir)
    )
    torch.save(model.state_dict(), modeldir + "/model_" + name + "_weight.pth")  # fin
    # モデル全体保存
    # https://take-tech-engineer.com/pytorch-model-save-load/#toc3
    print(" model is saved to {} at {}".format("model_" + name + "_all.pth", modeldir))
    torch.save(model, modeldir + "/model_" + name + "_all.pth")
    # python用のtorch scriptを保存
    torch.jit.script(model).save(modeldir + "/model_" + name + "_torchscript.pt")
    # c++用のtorch scriptを保存
    save_model_cc(model, modeldir, name=name)
    return 0
