import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import mlwc.fourier.fit_diel
from mlwc.fourier.acf_fourier import (
    calc_total_mol_acf_cross,
    calc_total_mol_acf_self,
    dielec,
)
from mlwc.fourier.dipole_core import diel_function
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


@DeprecationWarning
class Plot_totaldipole:
    """plot time vs dipole figure for total_dipole

    Returns:
        _type_: _description_
    """

    def __init__(self, dipole_filename):
        self._filename = dipole_filename

        if not os.path.isfile(self._filename):
            logger.error(
                " ERROR (Plot_totaldipole) :: "
                + str(self._filename)
                + " does not exist !!"
            )
            raise FileNotFoundError(
                " ERROR (Plot_totaldipole) :: "
                + str(self._filename)
                + " does not exist !!"
            )
        logger.info(" ============================ ")
        logger.info(f" filename  :: {self._filename}")
        # load txt in numpy ndarray
        self.data = np.loadtxt(self._filename, comments="#")
        logger.info(f" number of data :: {np.shape(self.data)}")
        self.__get_timestep()
        logger.info(f" timestep [fs] :: {self.timestep}")
        self.__get_temperature()
        logger.info(f" temperature [K] :: {self.temperature}")
        self.__get_unitcell()
        logger.info(f" unitcell [Ang] :: {self.unitcell}")
        logger.info(" ============================ ")

    def __get_timestep(self) -> int:
        """extract timestep [fs] from total_dipole.txt"""
        with open(self._filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TIMESTEP"):
                    time = float(line.split(" ")[1])
                    break
        self.timestep = time
        return 0

    def __get_unitcell(self):
        """extract unitcell from total_dipole.txt"""
        with open(self._filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#UNITCELL"):
                    unitcell = line.strip("\n").strip().split(" ")[1:]
                    break
        self.unitcell = np.array([float(i) for i in unitcell]).reshape([3, 3])
        return 0

    def __get_temperature(self):
        """extract unitcell from total_dipole.txt"""
        with open(self._filename) as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.startswith("#TEMPERATURE"):
                    temp = float(line.split(" ")[1])
                    break
        self.temperature = temp
        return 0


class Plot_moleculedipole(Plot_totaldipole):
    """plot time vs dipole figure for total_dipole

    Returns:
        _type_: _description_
    """

    def __init__(self, dipole_filename):
        # 継承元から初期化
        super().__init__(dipole_filename)
        self.__get_num_mol()
        logger.info(" --------- ")
        logger.info(f" number of mol :: {self.__NUM_MOL}")
        logger.info(" --------- ")
        # データ形状を変更[frame,mol_id,3dvector]
        self.data = self.data[:, 2:].reshape(-1, self.__NUM_MOL, 3)

    def __get_num_mol(self):
        """extract num_mol from molecule_dipole.txt"""
        # 1行目の最大値が分子数
        self.__NUM_MOL = int(np.max(self.data[:, 1])) + 1
        return 0

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
        # self term ACF
        self_data = calc_total_mol_acf_self(calc_data, engine="tsa")
        rfreq_self, ffteps1_self, ffteps2_self = process.calc_fourier_only_with_window(
            self_data, eps_n2, window="hann"
        )
        # here, we introduce moving-average for both dielectric-function and refractive-index
        diel_self = diel_function(rfreq_self, ffteps1_self, ffteps2_self, step)
        diel_self.diel_df.to_csv(self._filename + "_self_diel.csv")
        diel_self.refractive_df.to_csv(self._filename + "_self_refractive.csv")
        logger.info(" finish self terms")
        # cross term ACF
        cross_data = calc_total_mol_acf_cross(calc_data, engine="tsa")
        logger.info(" finish cross terms")
        # rfreq_self = rfreq_cross
        rfreq_cross, ffteps1_cross, ffteps2_cross = (
            process.calc_fourier_only_with_window(cross_data, eps_n2, window="hann")
        )
        rfreq_total, ffteps1_total, ffteps2_total = (
            process.calc_fourier_only_with_window(
                self_data + cross_data, eps_n2, window="hann"
            )
        )
        # cross
        diel_cross = diel_function(rfreq_cross, ffteps1_cross, ffteps2_cross, step)
        diel_cross.diel_df.to_csv(self._filename + "_cross_diel.csv")
        diel_cross.refractive_df.to_csv(self._filename + "_cross_refractive.csv")
        # total
        diel_total = diel_function(rfreq_total, ffteps1_total, ffteps2_total, step)
        diel_total.diel_df.to_csv(self._filename + "_total_diel.csv")
        diel_total.refractive_df.to_csv(self._filename + "_total_refractive.csv")
        return 0


def command_diel_mol(args):
    EVP = Plot_moleculedipole(args.Filename)
    EVP.calc_dielectric_spectrum(
        float(args.eps), int(args.start), int(args.end), int(args.step)
    )
    return 0
