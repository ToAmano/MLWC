# flake8: noqa

""" """

import torch
from torch import nn

from mlwc.include.mlwc_logger import setup_cmdline_logger
from mlwc.ml.model.mlmodel_abstract import AbstractModel

logger = setup_cmdline_logger("MLWC." + __name__)


class NetWithoutBatchNormalization(AbstractModel):
    """
    specify modelname !!
    """

    def __init__(
        self,
        modelname: str,
        nfeatures: int = 288,
        M: int = 20,
        Mb: int = 6,
        Rcs: float = 4.0,
        Rc: float = 6.0,
        bondtype: str = "CH",
        hidden_layers_enet: list[int] = [50, 50],
        hidden_layers_fnet: list[int] = [50, 50],
        list_atomim_number: list[int] = [6, 1, 8],
        list_descriptor_length: list[int] = [24, 24, 24],
    ):
        # !! caution !!
        # parameters below are used in cpp/predict.cpp (dipole_frame::predict_bond_dipole_at_frame)
        # to automatically construct desctiptors.
        super().__init__()
        self.modeltype: str = "NET_withoutBN"  # save class name
        self.modelname: str = modelname
        # Start parameters
        self.M: int = M
        self.Mb: int = Mb  # <= M
        # TODO :: hard code 4*24*3=288 # len(train_X_ch[0][0])
        self.nfeatures: int = nfeatures
        self.Rcs: float = Rcs  # inner cutoff radius
        self.Rc: float = Rc  # outer cutoff radius
        self.bondtype: str = bondtype  # "CH" or "HH"
        self.list_atomim_number: list[int] = list_atomim_number  # [C, H, O]
        # [C, H, O]
        self.list_descriptor_length: list[int] = list_descriptor_length
        self.len_descriptor: int = 4 * sum(list_descriptor_length)  # 288
        # End parameters

        self.hidden_layers_enet: list[int] = hidden_layers_enet
        self.hidden_layers_fnet: list[int] = hidden_layers_fnet

        # Embedding Net
        self.nfeatures_enet = int(self.len_descriptor / 4)  # 72
        # 定数（モデル定義時に必要となるもの）
        self.INPUT_FEATURES_enet = self.nfeatures_enet  # 入力（特徴）の数： 記述子の数
        # self.LAYER1_NEURONS_enet = 50             # ニューロンの数
        # self.LAYER2_NEURONS_enet = 50             # ニューロンの数
        self.OUTPUT_RESULTS_enet = self.M * self.nfeatures_enet  # 出力結果の数：

        # Fitting Net
        self.nfeatures_fnet = int(self.M * self.Mb)
        # 定数（モデル定義時に必要となるもの）
        self.INPUT_FEATURES_fnet = self.nfeatures_fnet  # 入力（特徴）の数： 記述子の数
        # self.LAYER1_NEURONS_fnet = 50     # ニューロンの数
        # self.LAYER2_NEURONS_fnet = 50     # ニューロンの数
        self.OUTPUT_RESULTS_fnet = self.M  # 出力結果の数：

        # Dynamically create the embedding layers
        # linear -> leakyReLUの順番で隠れ層を重ねていく．
        enet_layers = []
        input_size = self.INPUT_FEATURES_enet  # input size of input-layer
        for neurons in hidden_layers_enet:
            enet_layers.append(nn.Linear(input_size, neurons))
            enet_layers.append(nn.LeakyReLU())
            input_size = neurons
        enet_layers.append(nn.Linear(input_size, self.OUTPUT_RESULTS_enet))
        self.enet = nn.Sequential(*enet_layers)

        # Dynamically create the fitting layers
        fnet_layers = []
        input_size = self.INPUT_FEATURES_fnet
        for neurons in hidden_layers_fnet:
            fnet_layers.append(nn.Linear(input_size, neurons))
            fnet_layers.append(nn.LeakyReLU())
            input_size = neurons
        fnet_layers.append(nn.Linear(input_size, self.OUTPUT_RESULTS_fnet))
        self.fnet = nn.Sequential(*fnet_layers)

        # バッチ規格化層
        # self.bn2 = nn.BatchNorm1d(LAYER1_NEURONS) #バッチ正規化

        # # 隠れ層：1つ目のレイヤー（layer）
        # self.Enet_layer1 = nn.Linear(
        #     self.INPUT_FEATURES_enet,                # 入力ユニット数（＝入力層）
        #     self.LAYER1_NEURONS_enet)                # 次のレイヤーの出力ユニット数

        # # 隠れ層：2つ目のレイヤー（layer）
        # self.Enet_layer2 = nn.Linear(
        #     self.LAYER1_NEURONS_enet,                # 入力ユニット数
        #     self.LAYER2_NEURONS_enet)                # 次のレイヤーの出力ユニット数

        # # 出力層
        # self.Enet_layer_out = nn.Linear(
        #     self.LAYER2_NEURONS_enet,                # 入力ユニット数
        #     self.OUTPUT_RESULTS_enet)                # 出力結果への出力ユニット数

        # ##### Fitting net #####
        # # 隠れ層：1つ目のレイヤー（layer）
        # self.Fnet_layer1 = nn.Linear(
        #     self.INPUT_FEATURES_fnet,                # 入力ユニット数（＝入力層）
        #     self.LAYER1_NEURONS_fnet)                # 次のレイヤーの出力ユニット数

        # # 隠れ層：2つ目のレイヤー（layer）
        # self.Fnet_layer2 = nn.Linear(
        #     self.LAYER1_NEURONS_fnet,                # 入力ユニット数
        #     self.LAYER2_NEURONS_fnet)                # 次のレイヤーの出力ユニット数

        # # 出力層
        # self.Fnet_layer_out = nn.Linear(
        #     self.LAYER2_NEURONS_fnet,                # 入力ユニット数
        #     self.OUTPUT_RESULTS_fnet)                # 出力結果への出力ユニット数

    def forward(self, x):

        # Si(1/Rをカットオフ関数で処理した値）のみを抽出する
        Q1 = x[:, ::4]
        NB = Q1.size()[0]
        N = Q1.size()[1]
        # Embedding Netに代入する
        # embedded_x = nn.functional.leaky_relu(self.Enet_layer1(Q1))
        # embedded_x = nn.functional.leaky_relu(self.Enet_layer2(embedded_x))
        # embedded_x = self.Enet_layer_out(embedded_x)  # ※最終層は線形

        embedded_x = self.enet(Q1)
        # embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x, (NB, self.M, N))
        # 入力データをNB x N x 4 の行列に変形
        matrix_q = torch.reshape(x, (NB, N, 4))
        # Enetの出力との掛け算
        matrix_t = torch.matmul(embedded_x, matrix_q)
        # matTの次元はNB x M x 4 となっている
        # matSを作る(ハイパーパラメータMbで切り詰める)
        matrix_s = matrix_t[:, : self.Mb, :]
        # matSの転置行列を作る　→　NB x 4 x Mb となる
        matrix_st = torch.transpose(matrix_s, 1, 2)
        # matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる
        matrix_d = torch.matmul(matrix_t, matrix_st)
        # matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。
        matrix_d_1d = torch.reshape(matrix_d, (NB, self.M * self.Mb))
        # fitting Net に代入する
        # fitD = nn.functional.leaky_relu(self.Fnet_layer1(matD1))
        # fitD = nn.functional.leaky_relu(self.Fnet_layer2(fitD))
        # fitD = self.Fnet_layer_out(fitD)  # ※最終層は線形
        fitting_d = self.fnet(matrix_d_1d)
        # fitDの次元はNB x M となる。これをNB x 1 x Mの行列にする
        fitting_d_3d = torch.reshape(fitting_d, (NB, 1, self.M))
        # fttD3とmatTの掛け算
        matrix_w = torch.matmul(fitting_d_3d, matrix_t)
        # matWはNb x 1 x  4 になっている。これをNB x 4 の2次元にする
        matrix_w_2d = torch.reshape(matrix_w, (NB, 4))
        # はじめの要素はいらないので、切り詰めてx,y,z にする
        matrix_output_w = matrix_w_2d[:, 1:]

        return matrix_output_w

    @torch.jit.export
    def embedded(self, x):
        # calculate embedded matrix E
        # see Eq. 11 in Phys. Rev. B 110, 165159
        Q1 = x[:, ::4]
        NB = Q1.size()[0]  # batch size (dynamical value)
        N = Q1.size()[
            1
        ]  # !! TODO : Nは取り入れる原子の数だが，これはself.nfeatures/4と同じでは？

        embedded_x = self.enet(Q1)
        # embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x, (NB, self.M, N))
        return embedded_x

    @torch.jit.export
    def feature_matrix(self, x):
        # calculate feature matrix D
        # see Eq. 11 in Phys. Rev. B 110, 165159
        Q1 = x[:, ::4]
        NB = Q1.size()[0]  # batch size (dynamical value)
        N = Q1.size()[
            1
        ]  # !! TODO : Nは取り入れる原子の数だが，これはself.nfeatures/4と同じでは？

        embedded_x = self.enet(Q1)
        # embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x, (NB, self.M, N))
        # 入力データをNB x N x 4 の行列に変形
        matrix_q = torch.reshape(x, (NB, N, 4))
        # Enetの出力との掛け算
        matrix_t = torch.matmul(embedded_x, matrix_q)
        # matTの次元はNB x M x 4 となっている
        # matSを作る(ハイパーパラメータMbで切り詰める)
        matrix_s = matrix_t[:, : self.Mb, :]
        # matSの転置行列を作る　→　NB x 4 x Mb となる
        matrix_st = torch.transpose(matrix_s, 1, 2)
        # matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる
        matrix_d = torch.matmul(matrix_t, matrix_st)
        # matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。
        matrix_d_1d = torch.reshape(matrix_d, (NB, self.M * self.Mb))
        return matrix_d_1d

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""
        return self.Rc

    @torch.jit.export
    def get_rscut(self) -> float:
        """Get inner cutoff radius of the model."""
        return self.Rcs

    @torch.jit.export
    def get_modelname(self) -> str:
        """Get the model name."""
        return self.modelname

    def save_torchscript_py(self, directory: str) -> None:
        # python用のtorch scriptを保存
        torch.jit.script(self).save(
            directory + "/model_" + self.modelname + "_torchscript.pt"
        )

    def save_torchscript_cpp(self, directory: str) -> None:
        """Save torchscript model"""
        torch.jit.script(self).save(directory + "/model_" + self.modelname + ".pt")

    def save_weight(self, directory: str) -> None:
        torch.save(
            self.state_dict(), directory + "/model_" + self.modelname + "_weight.pth"
        )  # fin

    def print_parameters(self) -> None:
        print(f" model NET :: nfeatures      :: {self.nfeatures}")
        print(f" model NET :: len_descriptor :: {self.len_descriptor}")
        print(f" nfeatures_enet              :: {format(self.nfeatures_enet)}")
        print(f" nfeatures_fnet              :: {format(self.nfeatures_fnet)}")
