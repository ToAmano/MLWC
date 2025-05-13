"""
This module provides a class for analyzing total dipole moment data from MD simulations.

It includes functionalities for:
- Setting simulation parameters (timestep, temperature, unit cell, dipole data).
- Calculating dielectric constant.
- Calculating time-dependent dielectric constant.
- Plotting time vs. dipole moment.
- Calculating dipole autocorrelation function.
- Calculating dielectric function.
- Calculating absorption coefficient.
- Calculating refractive index.

The class uses numpy for numerical calculations, pandas for data handling,
and matplotlib for plotting.
"""

# 発想を転換して，total_dipole.txtに対するクラスをここで実装する．
# pandas dataframeを継承するかどうかは難しいところ．

# 設計として，インスタンスがtimestep, temperature, unitcell, dataを持つ．

import cmath
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# for calculation of dielectric constant
from mlwc.fourier.acf_fourier import raw_calc_eps0_dielconst
from mlwc.fourier.autocorrelation import (
    autocorr,
    autocorr_numpy,
    autocorr_scipy,
    autocorr_statsmodels,
)
from mlwc.fourier.fouriertransform import fft
from mlwc.fourier.windowfunction import (
    apply_windowfunction_oneside,
    apply_windowfunction_twoside,
)
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


class TotalDipole:
    """
    A class for analyzing total dipole moment data.

    This class provides methods for setting parameters, calculating dielectric constants,
    plotting time-dependent properties, and performing Fourier transforms on dipole data.

    Attributes:
        timestep (float): Time step of the MD simulation in femtoseconds (fs).
        temperature (float): Temperature of the MD simulation in Kelvin (K).
        unitcell (np.ndarray): Unit cell of the MD simulation.
        data (np.ndarray): Dipole data of the MD simulation.
    """

    def __init__(self, data=None, unitcell=None, timestep=None, temperature=None):
        """
        Initializes the totaldipole class.

        Attributes:
            timestep (float): time step of MD simulation.
            temperature (float): temperature of MD simulation.
            unitcell (np.ndarray): unit cell of MD simulation.
            data (np.ndarray): dipole data of MD simulation.
        """
        self.timestep: float = timestep
        self.temperature: float = temperature
        self.unitcell: np.ndarray = unitcell
        self.data: np.ndarray = data

    def set_params(
        self,
        data: np.ndarray,
        unitcell: np.ndarray,
        timestep: float,
        temperature: float,
    ):
        """Set parameters for totaldipole class.

        Args:
            data (np.ndarray): dipole data of MD simulation. shape=(time, 4).
            unitcell (np.ndarray): unit cell of MD simulation. shape=(3, 3).
            timestep (float): time step of MD simulation [fs].
            temperature (float): temperature of MD simulation [K].

        Returns:
            int: 0 if successful.

        Raises:
            ValueError: if data, unitcell, timestep, or temperature is not the correct type.
            ValueError: if data shape is not correct.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> unitcell = np.random.rand(3, 3)
            >>> timestep = 1.0
            >>> temperature = 300.0
            >>> total_dipole = totaldipole()
            >>> total_dipole.set_params(data, unitcell, timestep, temperature)
            0
        """
        if not isinstance(data, np.ndarray):
            raise ValueError(" ERROR :: data is not numpy array")
        if not isinstance(unitcell, np.ndarray):
            raise ValueError(" ERROR :: unitcell is not numpy array")
        if not isinstance(timestep, float):
            raise ValueError(" ERROR :: timestep is not float")
        if not isinstance(temperature, float):
            raise ValueError(" ERROR :: temperature is not float")
        if np.shape(data)[1] != 3:
            raise ValueError(
                f" ERROR :: data shape is not correct :: got {np.shape(data)}"
            )
        self.data = data
        self.unitcell = unitcell
        self.timestep = timestep
        self.temperature = temperature
        return 0

    @classmethod
    def from_array(cls, data, unitcell, timestep, temperature):
        """既存のNumPy配列から初期化"""
        if not isinstance(data, np.ndarray):
            raise ValueError(" ERROR :: data is not numpy array")
        if not isinstance(unitcell, np.ndarray):
            raise ValueError(" ERROR :: unitcell is not numpy array")
        if not isinstance(timestep, float):
            raise ValueError(" ERROR :: timestep is not float")
        if not isinstance(temperature, float):
            raise ValueError(" ERROR :: temperature is not float")
        if np.shape(data)[1] != 3:
            raise ValueError(" ERROR :: data shape is not correct")
        return cls(data, unitcell, timestep, temperature)

    def print_info(self) -> int:
        """Print information of totaldipole class.

        Returns:
            int: 0 if successful.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> unitcell = np.random.rand(3, 3)
            >>> timestep = 1.0
            >>> temperature = 300.0
            >>> total_dipole = totaldipole()
            >>> total_dipole.set_params(data, unitcell, timestep, temperature)
            0
            >>> total_dipole.print_info()
            0
        """
        logger.info(" ============================ ")
        logger.info(f" number of data :: {np.shape(self.data)}")
        logger.info(f" timestep [fs] :: {self.timestep}")
        logger.info(f" temperature [K] :: {self.temperature}")
        logger.info(f" unitcell [Ang] :: {self.unitcell}")
        logger.info(" ============================ ")
        return 0

    def get_volume(self) -> float:
        """Calculate the volume of the unit cell.

        Returns:
            float: Volume of the unit cell in m^3.

        Examples:
            >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> total_dipole = totaldipole()
            >>> total_dipole.unitcell = unitcell
            >>> total_dipole.get_volume()
            1e-30
        """
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

    def get_mean_dipole(self) -> np.ndarray:
        """Calculate the mean dipole moment.

        Returns:
            np.ndarray: Mean dipole moment in x, y, and z directions.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> total_dipole = totaldipole()
            >>> total_dipole.data = data
            >>> total_dipole.get_mean_dipole()
            array([..., ..., ...])
        """
        dMx = self.data[:, 0]  # -np.mean(self.data[:,0])
        dMy = self.data[:, 1]  # -np.mean(self.data[:,1])
        dMz = self.data[:, 2]  # -np.mean(self.data[:,2])
        # mean_M=np.mean(dMx)**2+np.mean(dMy)**2+np.mean(dMz)**2    # <M>^2
        return np.array([np.mean(dMx), np.mean(dMy), np.mean(dMz)])

    def get_mean_dipolesquare(self) -> float:
        """Calculate the mean square dipole moment.

        Returns:
            float: Mean square dipole moment.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> total_dipole = totaldipole()
            >>> total_dipole.data = data
            >>> total_dipole.get_mean_dipolesquare()
            ...
        """
        dMx = self.data[:, 0]  # -np.mean(self.data[:,0])
        dMy = self.data[:, 1]  # -np.mean(self.data[:,1])
        dMz = self.data[:, 2]  # -np.mean(self.data[:,2])
        mean_M2 = np.mean(dMx**2) + np.mean(dMy**2) + np.mean(dMz**2)  # <M^2>
        return mean_M2

    def calculate_dielconst(self, eps_inf: float = 1.0) -> float:
        """Calculate the dielectric constant.

        Calculates only eps0.

        Args:
            eps_inf (float, optional): Dielectric constant at infinite frequency. Defaults to 1.0.

        Returns:
            list[float]: A list containing the dielectric constant, mean square dipole moment, and mean dipole moment.

        Examples:
            >>> data = np.random.rand(100, 4)
            >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            >>> timestep = 1.0
            >>> temperature = 300.0
            >>> total_dipole = totaldipole()
            >>> total_dipole.set_params(data, unitcell, timestep, temperature)
            0
            >>> total_dipole.calc_dielconst()
            [..., ..., ...]
        """

        eps0 = 8.8541878128e-12
        debye = 3.33564e-30
        kb = 1.38064852e-23
        # volume of the cell in m^3
        volume: float = self.get_volume()
        # <M^2> = (np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2))
        mean_M2: float = self.get_mean_dipolesquare()
        # <M>^2 = np.mean(dMx)**2+np.mean(dMy)**2+np.mean(dMz)**2
        mean_M: float = np.sum(np.square(self.get_mean_dipole()))
        logger.info(f" <M^2>-<M>^2 = {mean_M2-mean_M}")
        logger.info(f" Volume = {volume}")

        # dielconst
        eps_0: float = eps_inf + ((mean_M2 - mean_M) * (debye**2)) / (
            3.0 * volume * kb * self.temperature * eps0
        )
        return [eps_0, mean_M2, mean_M]

    def calc_time_vs_dielconst(
        self, start: int, end: int, eps_inf: float = 1
    ) -> pd.DataFrame:
        """
        Calculate the dielectric constant as a function of time.

        This method calculates the dielectric constant at various time steps
        within a specified range, providing insights into the time evolution
        of the dielectric properties of the simulated system.

        Parameters
        ----------
        start : int
            The starting index for the time range.
        end : int
            The ending index for the time range. Use -1 to include all data points to the end.
        eps_inf : float, optional
            The dielectric constant at infinite frequency. Defaults to 1.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the time, dielectric constant,
            mean square dipole moment, and mean dipole moment.
            The DataFrame is also saved to "eps0_vs_time.csv".

        Raises
        ------
        ValueError
            If `start` or `end` is larger than the data length.

        Examples
        --------
        >>> data = np.random.rand(100, 4)
        >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> timestep = 1.0
        >>> temperature = 300.0
        >>> total_dipole = totaldipole()
        >>> total_dipole.set_params(data, unitcell, timestep, temperature)
        0
        >>> df = total_dipole.calc_time_vs_dielconst(start=10, end=50, eps_inf=1.0)
        >>> print(df.head())
           time_fs      eps0   mean_M2    mean_M
        0     10.0  1.000000  0.083333  0.000000
        1     20.0  1.000000  0.083333  0.000000
        2     30.0  1.000000  0.083333  0.000000
        3     40.0  1.000000  0.083333  0.000000
        """
        eps0_list = []
        mean_M2_list = []
        mean_M_list = []
        time_list = []

        if start > len(self.data):
            raise ValueError(" ERROR :: start is larger than data length")
        if end > len(self.data):
            raise ValueError(" ERROR :: end is larger than data length")

        if end == -1:
            calc_data = self.data[start:, 1:]
        else:
            calc_data = self.data[start:end, 1:]
        logger.info(f"length calc_data :: {len(calc_data)}")

        SAMPLE = 100  # !! hard code
        # インデックスのリストを numpy で作成（0 を除外）
        indices = np.arange(1, len(calc_data))
        indices = indices[indices % SAMPLE == 0]
        logger.debug(f"indices = {indices}")
        # 各インデックスに対して関数を適用（numpyの配列処理）
        results = np.array(
            [
                raw_calc_eps0_dielconst(
                    calc_data[:idx, :], self.unitcell, self.temperature, eps_inf
                )
                for idx in indices
            ]
        )
        # 結果を numpy 配列から個別のリストに変換
        eps0_list, mean_M2_list, mean_M_list = results.T  # 転置して各列を取得
        time_list = indices * self.timestep  # 時間リストを作成
        # DataFrame の作成と CSV 出力
        df = pd.DataFrame(
            {
                "time_fs": time_list,  # in fs
                "eps0": eps0_list,
                "mean_M2": mean_M2_list,
                "mean_M": mean_M_list,
            }
        )
        df.to_csv("eps0_vs_time.csv", index=False)
        return df

    @classmethod
    def plot_time_vs_dielconst(cls, df: pd.DataFrame):
        """Plot time vs dielectric constant.

        This method plots the dielectric constant as a function of time,
        visualizing the time evolution of the dielectric properties.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing the time and dielectric constant data.
            It must have columns named "time_fs" and "eps0".

        Raises
        ------
        ValueError
            If "time_fs" or "eps0" column is not found in DataFrame.

        Returns
        -------
        int
            0 if successful. The plot is saved to "time_dielconst.pdf".

        Examples
        --------
        >>> data = {'time_fs': [10, 20, 30, 40], 'eps0': [1.0, 1.1, 1.2, 1.3]}
        >>> df = pd.DataFrame(data)
        >>> totaldipole.plot_time_vs_dielconst(df)
        0
        """
        if "time_fs" not in df.columns:
            raise ValueError(" ERROR :: time column is not found in DataFrame")
        if "eps0" not in df.columns:
            raise ValueError(" ERROR :: eps0 column is not found in DataFrame")
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(
            df["time_fs"] / 1000 / 1000, df["eps0"], label="dielconst"
        )  # time in ns
        xlabel = "Time [ns]"  # "Time $\mathrm{ps}$"
        ylabel = "Dielconst"
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc="upper right", fontsize=15)
        fig.savefig("time_dielconst.pdf")
        fig.delaxes(ax)
        return 0

    def plot_total_dipole(self):
        """Plot total dipole moment as a function of time.

        This method plots the total dipole moment in x, y, and z directions
        as a function of time, providing insights into the dipole moment
        behavior of the simulated system.

        Returns
        -------
        int
            0 if successful. The plot is saved to "time_totaldipole.pdf".

        Examples
        --------
        >>> data = np.random.rand(100, 4)
        >>> unitcell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        >>> timestep = 1.0
        >>> temperature = 300.0
        >>> total_dipole = totaldipole()
        >>> total_dipole.set_params(data, unitcell, timestep, temperature)
        0
        >>> total_dipole.plot_total_dipole()
        0
        """
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(self.data[:, 0] * self.timestep / 1000, self.data[:, 1], label="x")
        ax.plot(self.data[:, 0] * self.timestep / 1000, self.data[:, 2], label="y")
        ax.plot(self.data[:, 0] * self.timestep / 1000, self.data[:, 3], label="z")
        xlabel = "Time [ps]"  # "Time $\mathrm{ps}$"
        ylabel = "Dipole [D]"
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc="upper right", fontsize=15)
        fig.savefig("time_totaldipole.pdf")
        fig.delaxes(ax)
        return 0

    def _calculate_coefficient_dielectricfunction(self):
        """calculate coeff 1/3kTV

        この比例係数は，無次元量になる．


        Args:
            UNITCELL_VECTORS (_type_): _description_
            TEMPERATURE (float, optional): _description_. Defaults to 300.

        Returns:
            _type_: _description_
        """
        # >>>>>>>>>>>
        eps0 = 8.8541878128e-12
        debye = 3.33564e-30
        kb = 1.38064852e-23

        kbT = kb * self.temperature

        # 比誘電率
        # !! 3で割らないようになっているのは，autocorrのところで平均を取るようにしているから．
        eps_0_coeff: float = (debye**2) / (self.get_volume() * kbT * eps0)

        return eps_0_coeff

    def calculate_dipoleautocorrelation(self) -> pd.DataFrame:
        acf_mean = autocorr(autocorr_scipy).compute_autocorr2d(self.data)
        return acf_mean

    def calculate_fft_from_dipole(
        self, window_type: str, method_dipole: Literal["direct", "derivative"]
    ):
        if method_dipole == "direct":
            dipole_array = self.data - self.get_mean_dipole()  # calculate M-<M>
            logger.info(f"mean_dipole = {self.get_mean_dipole()}")
        elif method_dipole == "derivative":
            # dM/dt : with time of ps = 1/THz
            dipole_array = np.diff(self.data, axis=0) / (self.timestep / 1000)
            logger.info(f"mean_derivative_dipole = {np.mean(dipole_array,axis=0)}")
            dipole_array = dipole_array - np.mean(dipole_array, axis=0)
        # calculate acf
        acf_mean_array: np.ndarray = autocorr(autocorr_numpy).compute_autocorr2d(
            x=dipole_array, ifmean=True
        )
        logger.info(
            f"len(dipole)={len(dipole_array)} :: len(acf)={len(acf_mean_array)}"
        )
        # apply window function to acf
        acf_with_window_array = apply_windowfunction_oneside(
            acf_mean_array, window_type
        )
        # calculate FFT
        df_fft = fft.calculate_fft_dielfunction(acf_with_window_array, self.timestep)
        return df_fft

    def calculate_dielfunction(
        self,
        eps_inf: float = 1.0,
        method_dipole: Literal["direct", "derivative"] = "direct",
        window_type: str = "hann",
    ) -> pd.DataFrame:
        """calculate real & imaginary part of dielectric function"""
        # static dielectric constant
        [eps_0, mean_M2, mean_M] = self.calculate_dielconst(eps_inf)
        logger.info(f"coeff = {self._calculate_coefficient_dielectricfunction()}")
        if method_dipole == "direct":
            df_fft = self.calculate_fft_from_dipole(window_type, method_dipole="direct")
            # calculate system specific coeficient
            df_fft["diel_real"] = (
                df_fft["imag"]
                * (2 * np.pi * df_fft["freq_thz"])
                * self._calculate_coefficient_dielectricfunction()
                + eps_0
            )
            df_fft["diel_imag"] = (
                df_fft["real"]
                * (2 * np.pi * df_fft["freq_thz"])
                * self._calculate_coefficient_dielectricfunction()
            )
        elif method_dipole == "derivative":  # use dipole derivative
            df_fft = self.calculate_fft_from_dipole(
                window_type, method_dipole="derivative"
            )
            # calculate system specific coeficient
            df_fft["diel_real"] = (
                df_fft["imag"]
                / (2 * np.pi * df_fft["freq_thz"])
                * self._calculate_coefficient_dielectricfunction()
                + eps_0
            )
            df_fft["diel_imag"] = (
                df_fft["real"]
                / (2 * np.pi * df_fft["freq_thz"])
                * self._calculate_coefficient_dielectricfunction()
            )
        df_fft = df_fft[["freq_thz", "freq_kayser", "diel_real", "diel_imag"]]
        return df_fft

    def calculate_dielfunction_imag(
        self,
        method_fft: Literal["direct", "wk"] = "direct",
        method_dipole: Literal["direct", "derivative"] = "direct",
    ) -> pd.DataFrame:
        if method_fft == "direct":
            if method_dipole == "direct":
                df_fft = self.calculate_fft_from_dipole("hann", method_dipole="direct")
                # calculate system specific coeficient
                diel_imag = (
                    df_fft["real"]
                    * (2 * np.pi * df_fft["freq_thz"])
                    * self._calculate_coefficient_dielectricfunction()
                )
            elif method_dipole == "derivative":
                df_fft = self.calculate_fft_from_dipole(
                    "hann", method_dipole="derivative"
                )
                # calculate system specific coeficient
                diel_imag = (
                    df_fft["real"]
                    / (2 * np.pi * df_fft["freq_thz"])
                    * self._calculate_coefficient_dielectricfunction()
                )
            else:
                raise ValueError("method_dipole should be direct or derivative")
            df = pd.DataFrame()
            df["freq_thz"] = df_fft["freq_thz"]
            df["freq_kayser"] = df_fft["freq_kayser"]
            df["diel_imag"] = diel_imag
        elif method_fft == "wk":
            # apply window function to dipole
            dipole_with_window_array = apply_windowfunction_twoside(self.data, "hann")
            # calculate dipole FFT
            dipole_fft = fft.calculate_fft_vdos(dipole_with_window_array, self.timestep)
            df = pd.DataFrame()
            df["freq_thz"] = dipole_fft["freq_thz"]
            df["freq_kayser"] = dipole_fft["freq_kayser"]
            df["diel_imag"] = (
                dipole_fft["vdos"] ** 2
                * (2 * np.pi * df["freq_thz"])
                * self._calculate_coefficient_dielectricfunction()
                / 2.0
            )  # Winner-Khinchin
            # TODO :: abs(fft)**2 is the reference expression
            # TODO :: which is correct, two sided FFT (now) or one sided FFT and abs(fft)**2?
        return df

    def calculate_absorption(
        self, method_dipole: Literal["direct", "derivative"]
    ) -> pd.DataFrame:
        """Calculate absorption (alpha*n) in cm-1

        # 光速は[cm*THz]に変換して3e-2になっている．
        # 2piはomega = 2pi*fであることから．

        Args:
            df (pd.DataFrame): dataframe of dielectric function

        Returns:
            _type_: _description_
        """
        speedoflight = 0.03  # in cm*THz
        if method_dipole == "direct":
            df_fft = self.calculate_dielfunction_imag("direct")
            alphan = (
                df_fft["diel_imag"] * (2 * np.pi * df_fft["freq_thz"]) / speedoflight
            )
        elif method_dipole == "derivative":
            df_fft = self.calculate_fft_from_dipole("hann", method_dipole="derivative")
            # calculate system specific coeficient
            alphan = (
                df_fft["real"]
                * self._calculate_coefficient_dielectricfunction()
                / speedoflight
            )
        else:
            raise ValueError("method should be direct or derivative")

        df = pd.DataFrame()
        df["freq_kayser"] = df_fft["freq_kayser"]
        df["freq_thz"] = df_fft["freq_thz"]
        df["alphan"] = alphan
        return df

    def calculate_refractiveindex(self, **kwargs):  # include epsilon/alpha
        speedoflight = 0.03  # in cm-1THz
        df = self.calculate_dielfunction(**kwargs)
        # 本来はここはマイナスだが，プラスで計算しておくと（kappaもマイナスで定義されているので）全体として辻褄が合うようになっている．
        epsilon = df["diel_real"] + 1j * df["diel_imag"]
        refractive_index = []
        re_refractive_index = []
        im_refractive_index = []

        for i in epsilon:
            a, b = cmath.polar(i)
            refractive_index.append(cmath.rect(np.sqrt(a), b / 2))

        re_refractive_index = [a.real for a in refractive_index]
        im_refractive_index = [a.imag for a in refractive_index]

        df["real_ref_index"] = re_refractive_index
        df["imag_ref_index"] = im_refractive_index
        df["alpha"] = (
            df["imag_ref_index"] * (2 * np.pi * df["freq_thz"]) * 2 / speedoflight
        )
        return df


# functions
def calculate_dielconst(totaldipole: TotalDipole, eps_inf: float = 1.0) -> pd.DataFrame:
    return totaldipole.calculate_dielconst(eps_inf)


def calculate_dielfunction(
    totaldipole: TotalDipole, eps_inf: float = 1.0
) -> pd.DataFrame:
    return totaldipole.calculate_dielfunction(eps_inf)


def calculate_absorption(
    totaldipole: TotalDipole, method_dipole: Literal["direct", "derivative"]
) -> pd.DataFrame:
    return totaldipole.calculate_absorption(method_dipole)


def calculate_refractiveindex(
    totaldipole: TotalDipole, eps_inf: float = 1.0
) -> pd.DataFrame:
    return totaldipole.calculate_refractiveindex(eps_inf)
