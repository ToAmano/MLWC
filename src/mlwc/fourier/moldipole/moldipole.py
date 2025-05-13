# 発想を転換して，total_dipole.txtに対するクラスをここで実装する．
# pandas dataframeを継承するかどうかは難しいところ．

# 設計として，インスタンスがtimestep, temperature, unitcell, dataを持つ．

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# for calculation of dielectric constant
from mlwc.fourier.acf_fourier import (
    calc_total_mol_acf_cross,
    calc_total_mol_acf_self,
    dielec,
    raw_calc_gfactor,
)
from mlwc.fourier.dipole_core import diel_function
from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger("MLWC." + __name__)


class moldipole:
    """plot time vs dipole figure for total_dipole

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self.timestep: float = None
        self.temperature: float = None
        self.unitcell: np.ndarray = None
        self.data: np.ndarray = None

    def set_params(
        self,
        data: np.ndarray,
        unitcell: np.ndarray,
        timestep: float,
        temperature: float,
    ):
        if not isinstance(data, np.ndarray):
            raise ValueError(" ERROR :: data is not numpy array")
        if not isinstance(unitcell, np.ndarray):
            raise ValueError(" ERROR :: unitcell is not numpy array")
        if not isinstance(timestep, float):
            raise ValueError(" ERROR :: timestep is not float")
        if not isinstance(temperature, float):
            raise ValueError(" ERROR :: temperature is not float")
        if len(np.shape(data)) != 3:
            raise ValueError(
                f" ERROR :: data shape is not correct :: len(np.shape(data)) == {len(np.shape(data))}"
            )
        if np.shape(data)[2] != 3:
            raise ValueError(
                f" ERROR :: data shape is not correct :: np.shape(data)[2] == {np.shape(data)[2]}"
            )
        self.data = data
        self.unitcell = unitcell
        self.timestep = timestep
        self.temperature = temperature
        return 0

    def print_info(self) -> int:
        """print parameters"""
        logger.info(" ============================ ")
        logger.info(" number of data  :: %s", np.shape(self.data))
        logger.info(" timestep [fs]   :: %s", self.timestep)
        logger.info(" temperature [K] :: %s", self.temperature)
        logger.info(" unitcell [Ang]  :: %s", self.unitcell)
        logger.info(" ============================ ")
        return 0

    def get_volume(self):
        """get the volume of the unitcell in m^3"""
        A3 = 1.0e-30
        return (
            np.abs(
                np.dot(
                    np.cross(self.unitcell[:, 0], self.unitcell[:, 1]),
                    self.unitcell[:, 2],
                )
            )
            * A3
        )

    def get_num_mol(self):
        return np.shape(self.data)[1]

    def get_totaldipole(self, max_length: int = -1):
        return np.sum(self.data[:max_length], axis=1)

    def get_mean_moldipole(self, max_length: int = -1):
        # absoulte value of moldipole
        abs_moldipole = np.linalg.norm(self.data[:max_length], axis=2)
        # reshape to 2D ( reshape try to return "view")
        abs_moldipole = abs_moldipole.reshape(-1)
        return np.mean(abs_moldipole, axis=0)

    def get_mean_dipolesquare(self, max_length: int = -1):
        """_summary_

        Args:
            max_length (int, optional): use up to max_length data of the first row. Defaults to -1.

        Returns:
            _type_: _description_
        """
        data = self.data[:max_length]
        dMx = data[:, 0] - np.mean(data[:, 0])
        dMy = data[:, 1] - np.mean(data[:, 1])
        dMz = data[:, 2] - np.mean(data[:, 2])
        mean_M2 = np.mean(dMx**2) + np.mean(dMy**2) + np.mean(dMz**2)  # <M^2>
        return mean_M2

    def calc_gfactor(self, max_length: int = -1) -> float:
        """calculate kirkwood g-factor

        The definition of the Kirkwood g-factor is given by the following equation:
        g = <M^2> / (N * <mu>^2)
        where M is the the total dipole moment, N is the number of molecules, and <mu> is the mean molecular dipole moment.
        For detail, see the following references:
        - handgraaf2004Densityfunctional
        - pagliai2003Hydrogen
        - sharma2007Dipolar


        Returns:
            float: g_factor
        """

        total_dipole = self.get_totaldipole(max_length)
        dMx = total_dipole[:, 0] - np.mean(total_dipole[:, 0])
        dMy = total_dipole[:, 1] - np.mean(total_dipole[:, 1])
        dMz = total_dipole[:, 2] - np.mean(total_dipole[:, 2])
        mean_M2 = np.mean(dMx**2) + np.mean(dMy**2) + np.mean(dMz**2)  # <M^2>
        mean_moldipole = self.get_mean_moldipole(max_length)
        num_mol = self.get_num_mol()
        g_factor = mean_M2 / num_mol / (mean_moldipole**2)
        return g_factor

    def calc_dielectric_spectrum(self, eps_n2: float, start: int, end: int, step: int):
        logger.info(" ==================== ")
        logger.info(f"  start index :: {start}")
        logger.info(f"  end   index :: {end}")
        logger.info(f" moving average step :: {step}")
        logger.info(" ==================== ")
        process = dielec(self.unitcell, self.temperature, self.timestep)
        if end == -1:
            calc_data = self.data[start:, :, :]
        else:
            calc_data = self.data[start:end, :, :]
        logger.info(" ====================== ")
        logger.info(f"  len(data)    :: {len(calc_data)}")
        logger.info(" ====================== ")
        # まずはACFの計算
        self_data = calc_total_mol_acf_self(calc_data, engine="tsa")
        cross_data = calc_total_mol_acf_cross(calc_data, engine="tsa")
        # rfreq_self = rfreq_cross
        rfreq_self, ffteps1_self, ffteps2_self = process.calc_fourier_only_with_window(
            self_data, eps_n2, window="hann"
        )
        rfreq_cross, ffteps1_cross, ffteps2_cross = (
            process.calc_fourier_only_with_window(cross_data, eps_n2, window="hann")
        )
        rfreq_total, ffteps1_total, ffteps2_total = (
            process.calc_fourier_only_with_window(
                self_data + cross_data, eps_n2, window="hann"
            )
        )

        # here, we introduce moving-average for both dielectric-function and refractive-index
        diel_self = diel_function(rfreq_self, ffteps1_self, ffteps2_self, step)
        diel_self.diel_df.to_csv(self._filename + "_self_diel.csv", index=False)
        diel_self.refractive_df.to_csv(
            self._filename + "_self_refractive.csv", index=False
        )
        # cross
        diel_cross = diel_function(rfreq_cross, ffteps1_cross, ffteps2_cross, step)
        diel_cross.diel_df.to_csv(self._filename + "_cross_diel.csv", index=False)
        diel_cross.refractive_df.to_csv(
            self._filename + "_cross_refractive.csv", index=False
        )
        # total
        diel_total = diel_function(rfreq_total, ffteps1_total, ffteps2_total, step)
        diel_total.diel_df.to_csv(self._filename + "_total_diel.csv", index=False)
        diel_total.refractive_df.to_csv(
            self._filename + "_total_refractive.csv", index=False
        )
        return 0

    def calc_time_vs_gfactor(self, start: int, end: int) -> pd.DataFrame:
        """時間 vs 誘電定数の計算を行う

        Returns:
            _type_: _description_
        """
        gfactor_list = []
        time_list = []

        if start > len(self.data):
            raise ValueError(" ERROR :: start is larger than data length")
        if end > len(self.data):
            raise ValueError(" ERROR :: end is larger than data length")

        if end == -1:
            calc_data = self.data[start:, 1:]
        else:
            calc_data = self.data[start:end, 1:]
        logger.info("length calc_data :: %d", len(calc_data))

        SAMPLE = 100  # !! hard code
        for index in range(len(calc_data)):
            if index == 0:
                continue
            if index % SAMPLE == 0:
                logger.debug(index)
                g_factor = raw_calc_gfactor(calc_data[:index])
                gfactor_list.append(g_factor)
                time_list.append(index * self.timestep)
        # データの保存
        df = pd.DataFrame()
        df["time_fs"] = time_list  # in fs
        df["gfactor"] = gfactor_list
        df.to_csv("gfactor_vs_time.csv", index=False)
        return df

    @classmethod
    def plot_time_vs_gfactor(cls, df: pd.DataFrame):
        if "time_fs" not in df.columns:
            raise ValueError(" ERROR :: time column is not found in DataFrame")
        if "gfactor" not in df.columns:
            raise ValueError(" ERROR :: gfactor column is not found in DataFrame")
        # 時間 vs 誘電定数のプロットを行う
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(
            df["time_fs"] / 1000 / 1000, df["gfactor"], label="gfactor"
        )  # time in ns

        # 各要素で設定したい文字列の取得
        xlabel = "Time [ns]"  # "Time $\mathrm{ps}$"
        ylabel = "G-factor"
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc="upper right", fontsize=15)
        fig.savefig("time_gfactor.pdf")
        fig.delaxes(ax)
        return 0

    def get_dipole_histgram(self) -> pd.DataFrame:
        """
        Calculates and saves the histogram data of the dipole magnitudes.

        This method computes the magnitude of the dipole moments from the input
        data, generates a histogram, and saves the histogram data to a CSV file.
        The histogram is generated with a fixed number of bins (1000), which should be modified.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the dipole values and their corresponding densities.

        Examples
        --------
        >>> hist_plotter = Plot_histgram("dipole.txt", max=5.0)
        >>> hist_data = hist_plotter.get_histgram()
        >>> print(hist_data.head())
               dipole   density
        0  0.0025  0.0001
        1  0.0075  0.0002
        2  0.0125  0.0003
        3  0.0175  0.0004
        4  0.0225  0.0005
        """

        # 先にデータの数と値域から最適なヒストグラム構成を考える
        # TODO :: bins = 1000で固定しているので修正
        _length = len(self.data)

        # 最大値を計算する
        plot_data = np.linalg.norm(self.data.reshape(-1, 3), axis=1)
        _max_val = np.max(plot_data)
        # 最大値が4以下なら5で固定する
        if _max_val < 4:
            _hist_max_val = 5
        else:
            _hist_max_val = _max_val + 2

        # https://qiita.com/nkay/items/56bda7143981e3d5303f
        df = pd.DataFrame()
        hist = np.histogram(
            plot_data, bins=1000, range=[0, _hist_max_val], density=True
        )
        df["dipole"] = (hist[1][1:] + hist[1][:-1]) / 2
        df["density"] = hist[0]
        return df

    def plot_dipole_histgram(self):
        """make histgram & plot histgram

        Returns:
            _type_: _description_
        """
        logger.info(" ---------- ")
        logger.info(" dipole histgram plot ")
        logger.info(" ---------- ")

        # 最大値を計算する
        plot_data = np.linalg.norm(self.data.reshape(-1, 3), axis=1)
        _max_val = np.max(plot_data)
        # 最大値が4以下なら5で固定する
        if _max_val < 4:
            _hist_max_val = 5
        else:
            _hist_max_val = _max_val + 2

        plot_data = np.linalg.norm(self.data[:, 2:].reshape(-1, 3), axis=1)
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.hist(plot_data, bins=1000, range=[0, _hist_max_val], density=True)  # 描画

        # 各要素で設定したい文字列の取得
        xlabel = "Dipole [D]"  # "Time $\mathrm{ps}$"
        ylabel = "Density"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc="upper right", fontsize=15)
        fig.savefig("dipole_histgram.pdf")
        fig.delaxes(ax)
        return 0
