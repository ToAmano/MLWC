import os

import matplotlib.pyplot as plt
import numpy as np

import mlwc.cpmd.converter_cpmd
from mlwc.dataio.cpmd.read_traj_cpmd import (
    raw_cpmd_get_timestep,
    raw_cpmd_read_unitcell_vector,
)
from mlwc.fourier.totaldipole.totaldipole import TotalDipole
from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger(__name__)


class Plot_dipole:
    """Plots dipole moment and dielectric function data from a DIPOLE file.

    This class reads dipole moment data from a DIPOLE file and plots it, along with the
    corresponding dielectric function data.  The timestep dt is required to calculate
    the dielectric function, which may require inspecting the output file.

    According to the CPMD manual P192:
    Columns 2 to 4 in the DIPOLE file are the electronic contribution to the dipole moment,
    columns 5 to 7 are the total (electronic + ionic) dipole moment.
    All dipole moments are divided by the volume of the box.

    Therefore, columns 5-7 should be plotted.
    """

    def __init__(self, evp_filename, stdout):
        """Initializes the Plot_dipole class.

        Reads the DIPOLE file and extracts the timestep from the CPMD output file (stdout).

        Parameters
        ----------
        evp_filename : str
            The name of the EVP file (used for naming output plots).
        stdout : str
            The name of the CPMD output file (used to extract the timestep).

        Raises
        ------
        FileNotFoundError
            If the DIPOLE file does not exist.

        """
        self.__filename = evp_filename
        self.data = np.loadtxt("DIPOLE")  # 読み込むのはdipoleファイル
        self.timestep_fs = raw_cpmd_get_timestep(stdout)
        self.unitcell = raw_cpmd_read_unitcell_vector(stdout)

    def plot_dipole(self):
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(
            self.data[:, 0] * self.timestep_fs,
            self.data[:, 4],
            label=self.__filename + "_x",
            lw=3,
        )  # 描画
        ax.plot(
            self.data[:, 0] * self.timestep_fs,
            self.data[:, 5],
            label=self.__filename + "_y",
            lw=3,
        )  # 描画
        ax.plot(
            self.data[:, 0] * self.timestep_fs,
            self.data[:, 6],
            label=self.__filename + "_z",
            lw=3,
        )  # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Time $\mathrm{ps}$"
        ylabel = "Dipole [D/Volume?]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename + "_Dipole.pdf")
        fig.delaxes(ax)
        return 0

    def plot_dielec(self):
        """
        誘電関数の計算，及びそのプロットを行う．
        体積による規格化や，前にかかる係数などは何も処理しない．

        ---------
        TODO :: ちゃんとDIPOLEファイルでの係数の定義を突き止める．
        """

        totaldipole = TotalDipole()
        totaldipole.set_params(
            self.data, self.unitcell, self.timestep_fs, 300
        )  # FIXME ::hard code

        df_fft = totaldipole.calculate_dielfunction()
        kayser = df_fft["freq_kayser"]
        ffteps2 = df_fft["diel_imag"]

        # make a plot
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(kayser, ffteps2, label="DIPOLES", lw=3)  # 描画
        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Frequency $\mathrm{cm}^{-1}$"
        ylabel = "Dielec [arb.-unit]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        # 描画するのは0以上でok!
        ax.set_xlim([0, max(kayser)])

        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename + "_Dielec.pdf")
        fig.delaxes(ax)

        return 0

    def process(self):
        logger.info(" ==========================")
        logger.info(" Reading {:<20}   :: making Dipole plots ".format(self.__filename))
        logger.info("")
        self.plot_dipole()
        self.plot_dielec()


def command_cpmd_dipole(args):
    """
    plot DIPOLE file
    """
    Dipole = Plot_dipole(args.Filename, args.stdout)
    Dipole.process()
    return 0
