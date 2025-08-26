# flake8: noqa
"""
- 入力として座標をとる．
- これによって，forward関数の中で記述子の計算からモデルへの入力，最終的な
- 必要なもの
    - 座標リスト
    - 原子種
    - BC座標: list_bond_centers
    - ボンドのindex: self.bond_index
        - これはBC座標からbond indexを取得するのが目的
"""

import torch
from torch import nn

from mlwc.include.mlwc_logger import setup_library_logger
from mlwc.ml.descriptor.descriptor_torch import DescriptorTorchBondcenter
from mlwc.ml.model.mlmodel_abstract import AbstractModel, BaseModelWrapper

logger = setup_library_logger("MLWC." + __name__)


class NetWithoutBatchNormalizationDescs(AbstractModel):
    """
    Taking original structure (atomic coordinates and bond centers) as input
    specify modelname !!
    """

    def __init__(
        self,
        modelname: str,
        nfeatures: int = 288,
        m: int = 20,
        mb: int = 6,
        rcs: float = 4.0,
        rc: float = 6.0,
        bondtype: str = "CH",
        hidden_layers_enet: list[int] = [50, 50],
        hidden_layers_fnet: list[int] = [50, 50],
        list_atomim_number: list[int] = [6, 1, 8],
        list_maxat: list[int] = [24, 24, 24],
    ):
        # !! caution !!
        # parameters below are used in cpp/predict.cpp (dipole_frame::predict_bond_dipole_at_frame)
        # to automatically construct desctiptors.
        super().__init__()
        self.modeltype: str = "NET_withoutBN_descs"  # save class name for prediction
        # self.input_features: list[str] =
        self.modelname: str = modelname
        self.m: int = m
        self.mb: int = mb  # <= M
        self.nfeatures: int = nfeatures
        self.rcs: float = rcs  # inner cutoff radius
        self.rc: float = rc  # outer cutoff radius
        self.bondtype: str = bondtype  # "CH" or "HH"

        self.list_atomic_number: torch.Tensor = torch.tensor(
            list_atomim_number, dtype=torch.int
        )
        self.list_maxat: torch.Tensor = torch.tensor(list_maxat, dtype=torch.int)

        self.len_descriptor: int = 4 * sum(list_maxat)  # 288

        self.hidden_layers_enet: list[int] = hidden_layers_enet
        self.hidden_layers_fnet: list[int] = hidden_layers_fnet

        # Embedding Net
        self.nfeatures_enet = int(self.len_descriptor / 4)  # 72
        self.unput_features_enet = self.nfeatures_enet  # 入力（特徴）の数： 記述子の数
        self.output_results_enet = self.m * self.nfeatures_enet  # 出力結果の数：

        # Fitting Net
        self.nfeatures_fnet = int(self.m * self.mb)
        self.input_features_fnet = self.nfeatures_fnet  # 入力（特徴）の数： 記述子の数
        self.output_results_fnet = self.m  # 出力結果の数：

        # Dynamically create the embedding layers
        # linear -> leakyReLUの順番で隠れ層を重ねていく．
        enet_layers = []
        input_size = self.unput_features_enet  # input size of input-layer
        for neurons in hidden_layers_enet:
            enet_layers.append(nn.Linear(input_size, neurons))
            enet_layers.append(nn.LeakyReLU())
            input_size = neurons
        enet_layers.append(nn.Linear(input_size, self.output_results_enet))
        self.enet = nn.Sequential(*enet_layers)

        # Dynamically create the fitting layers
        fnet_layers = []
        input_size = self.input_features_fnet
        for neurons in hidden_layers_fnet:
            fnet_layers.append(nn.Linear(input_size, neurons))
            fnet_layers.append(nn.LeakyReLU())
            input_size = neurons
        fnet_layers.append(nn.Linear(input_size, self.output_results_fnet))
        self.fnet = nn.Sequential(*fnet_layers)

        # set descriptor
        # TODO :: 任意のdescriptorを設定できるようにする．
        self.descriptor = DescriptorTorchBondcenter()

    def forward(
        self,
        atomic_coordinate: torch.Tensor,
        atomic_numbers: torch.Tensor,
        bond_centers: torch.Tensor,
        unitcell_vector: torch.Tensor,
        device: str,  # TODO:: remove this variable
    ):
        # if device not in ["cpu", "cuda", "mps"]:
        #     raise ValueError(
        #         f"device should be one of cpu, cuda, and mps :: got {device}"
        #     )
        atomic_coordinate = atomic_coordinate.to(device)
        atomic_numbers = atomic_numbers.to(device)
        bond_centers = bond_centers.to(device)
        unitcell_vector = unitcell_vector.to(device)
        # descriptor
        x: torch.Tensor = self.descriptor.forward(
            atomic_coordinate,
            atomic_numbers,
            bond_centers,
            unitcell_vector,
            self.list_atomic_number,
            self.list_maxat,
            self.rcs,
            self.rc,
            device,
        )
        # Si(1/Rをカットオフ関数で処理した値）のみを抽出する
        q1 = x[:, ::4]
        nb: int = q1.size()[0]  # num_batch
        natoms: int = q1.size()[1]  # MaxAt*atomic_species (len(descs)/4)
        # print(
        #     f"Q1 = {Q1.device}, x = {x.device}, atomic_coordinate = {atomic_coordinate.device}"
        # )
        embedded_x = self.enet(q1)
        # embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x, (nb, self.m, natoms))
        # 入力データをNB x N x 4 の行列に変形
        matrix_q = torch.reshape(x, (nb, natoms, 4))
        # Enetの出力との掛け算
        matrix_t = torch.matmul(embedded_x, matrix_q)
        # matTの次元はNB x M x 4 となっている
        # matSを作る(ハイパーパラメータMbで切り詰める)
        matrix_s = matrix_t[:, : self.mb, :]
        # matSの転置行列を作る　→　NB x 4 x Mb となる
        matrix_st = torch.transpose(matrix_s, 1, 2)
        # matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる
        matrix_d = torch.matmul(matrix_t, matrix_st)
        # matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。
        matrix_d_1d = torch.reshape(matrix_d, (nb, self.m * self.mb))
        # fitting Net に代入する
        # fitD = nn.functional.leaky_relu(self.Fnet_layer1(matD1))
        # fitD = nn.functional.leaky_relu(self.Fnet_layer2(fitD))
        # fitD = self.Fnet_layer_out(fitD)  # ※最終層は線形
        fitting_d = self.fnet(matrix_d_1d)
        # fitDの次元はNB x M となる。これをNB x 1 x Mの行列にする
        fitting_d_3d = torch.reshape(fitting_d, (nb, 1, self.m))
        # fttD3とmatTの掛け算
        matrix_w = torch.matmul(fitting_d_3d, matrix_t)
        # matWはNb x 1 x  4 になっている。これをNB x 4 の2次元にする
        matrix_w_2d = torch.reshape(matrix_w, (nb, 4))
        # はじめの要素はいらないので、切り詰めてx,y,z にする
        matrix_outout_w = matrix_w_2d[:, 1:]
        return matrix_outout_w

    @torch.jit.export
    def embedding_network(self, x: torch.Tensor):
        """calculate embedded matrix E
        see Eq. 11 in Phys. Rev. B 110, 165159
        """
        q1 = x[:, ::4]
        nb = q1.size()[0]  # batch size (dynamical value)
        # !! TODO : Nは取り入れる原子の数だが，これはself.nfeatures/4と同じでは？ (同じになってなかったらerrorになる設計が良い)
        natoms = q1.size()[1]

        embedded_x = self.enet(q1)
        # embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x, (nb, self.m, natoms))
        return embedded_x

    @torch.jit.export
    def feature_matrix(self, x: torch.Tensor):
        """calculate feature matrix D
        see Eq. 11 in Phys. Rev. B 110, 165159
        """
        q1 = x[:, ::4]
        nb = q1.size()[0]  # batch size (dynamical value)
        natoms = q1.size()[
            1
        ]  # !! TODO : Nは取り入れる原子の数だが，これはself.nfeatures/4と同じでは？

        embedded_x = self.enet(q1)
        # embedded_xを(ミニバッチデータ数)xMxN (N=MaxAt*原子種数)に変換
        embedded_x = torch.reshape(embedded_x, (nb, self.m, natoms))
        # 入力データをNB x N x 4 の行列に変形
        matrix_q = torch.reshape(x, (nb, natoms, 4))
        # Enetの出力との掛け算
        matrix_t = torch.matmul(embedded_x, matrix_q)
        # matTの次元はNB x M x 4 となっている
        # matSを作る(ハイパーパラメータMbで切り詰める)
        matrix_s = matrix_t[:, : self.mb, :]
        # matSの転置行列を作る　→　NB x 4 x Mb となる
        matrix_st = torch.transpose(matrix_s, 1, 2)
        # matDを作る( matTとmatStの掛け算) →　NB x M x Mb となる
        matrix_d = torch.matmul(matrix_t, matrix_st)
        # matDを１次元化する。matD全体をニューラルネットに入力したいので、ベクトル化する。
        matrix_d_1d = torch.reshape(matrix_d, (nb, self.m * self.mb))
        return matrix_d_1d

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get cutoff radius of the model."""
        return self.rc

    @torch.jit.export
    def get_rscut(self) -> float:
        """Get inner cutoff radius of the model."""
        return self.rcs

    @torch.jit.export
    def get_modelname(self) -> str:
        """Get the model name."""
        return self.modelname

    def save_torchscript_py(self, directory: str) -> None:
        """save torch script for python"""
        torch.jit.script(self).save(
            directory + "/model_" + self.modelname + "_torchscript.pt"
        )

    def save_torchscript_cpp(self, directory: str) -> None:
        """save torch script for cpp"""
        torch.jit.script(self).save(f"{directory}/model_{self.modelname}.pt")

    def save_weight(self, directory: str) -> None:
        """only save weight"""
        torch.save(self.state_dict(), f"{directory}/model_{self.modelname}_weight.pth")

    def print_parameters(self) -> None:
        """output model parameters"""
        logger.info(" model NET :: nfeatures      :: %s", self.nfeatures)
        logger.info(" model NET :: len_descriptor :: %s", self.len_descriptor)
        logger.info(" nfeatures_enet              :: %s", format(self.nfeatures_enet))
        logger.info(" nfeatures_fnet              :: %s", format(self.nfeatures_fnet))


class ModelAHandler(BaseModelWrapper):
    def __init__(
        self, model: NetWithoutBatchNormalizationDescs, input_features: list[str]
    ):
        self.model = model
        self.input_features = input_features

    def preprocess(self, raw_input: dict) -> torch.Tensor:
        values = [raw_input[k] for k in self.input_features]
        return torch.tensor(values).float().unsqueeze(0)

    def predict(self, tensor_input: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            return self.model(tensor_input).item()

    @classmethod
    def load_from_file(cls, model_path: str, device: str) -> "ModelAHandler":
        model = torch.jit.load(model_path)
        model = model.to(device)
        logger.info("%s :: %s", model_path, model)
        model.share_memory()  # https://knto-h.hatenablog.com/entry/2018/05/22/130745
        model.eval()
        return cls(model=model, input_features=model.input_features)
