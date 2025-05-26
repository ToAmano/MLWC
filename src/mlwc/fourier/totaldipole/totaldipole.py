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
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlwc.fourier.acf_fourier import raw_calc_eps0_dielconst
from mlwc.fourier.autocorrelation import (
    autocorr,
    autocorr_numpy,
    autocorr_scipy,
    autocorr_statsmodels,
)
from mlwc.fourier.fouriertransform import fft

# for calculation of dielectric constant
from mlwc.fourier.windowfunction import (
    apply_windowfunc_conv,
    apply_windowfunc_oneside,
    apply_windowfunc_twoside,
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
        unitcell (NDArray[np.float64]): Unit cell of the MD simulation.
        data (NDArray[np.float64]): Dipole data of the MD simulation.
    """

    def __init__(
        self,
        # data: NDArray[np.float64] | None = None,
        # unitcell: NDArray[np.float64] | None = None,
        # timestep: float | None = None,
        # temperature: float | None = None,
        data: Optional[NDArray[np.float64]] = None,
        unitcell: Optional[NDArray[np.float64]] = None,
        timestep: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initializes the totaldipole class.

        Attributes:
            timestep (float): time step of MD simulation.
            temperature (float): temperature of MD simulation.
            unitcell (NDArray[np.float64]): unit cell of MD simulation.
            data (NDArray[np.float64]): dipole data of MD simulation.
        """
        # self.timestep: float = timestep
        # self.temperature: float = temperature
        # self.unitcell: NDArray[np.float64] = unitcell
        # self.data: NDArray[np.float64] = data
        if data is not None:
            self.data: NDArray[np.float64] = data
        if unitcell is not None:
            self.unitcell: NDArray[np.float64] = unitcell
        if timestep is not None:
            self.timestep: float = timestep
        if temperature is not None:
            self.temperature: float = temperature

    def set_params(
        self,
        data: NDArray[np.float64],
        unitcell: NDArray[np.float64],
        timestep: float,
        temperature: float,
    ) -> int:
        """Set parameters for totaldipole class.

        Args:
            data (NDArray[np.float64]): dipole data of MD simulation. shape=(time, 4).
            unitcell (NDArray[np.float64]): unit cell of MD simulation. shape=(3, 3).
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
    def from_array(
        cls,
        data: NDArray[np.float64],
        unitcell: NDArray[np.float64],
        timestep: float,
        temperature: float,
    ):
        """Initialize from numpy array"""
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
        """Print information of totaldipole class."""
        logger.info(" ============================ ")
        logger.info(" number of data :: %s", np.shape(self.data))
        logger.info(" timestep [fs] :: %s", self.timestep)
        logger.info(" temperature [K] :: %s", self.temperature)
        logger.info(" unitcell [Ang] :: %s", self.unitcell)
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

    def get_mean_dipole(self) -> NDArray[np.float64]:
        """Calculate the mean dipole moment.

        Returns:
            NDArray[np.float64]: Mean dipole moment in x, y, and z directions.

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
        mean_M2: float = np.mean(dMx**2) + np.mean(dMy**2) + np.mean(dMz**2)  # <M^2>
        return mean_M2

    def calculate_dielconst(self, eps_inf: float = 1.0) -> list[float]:
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
        logger.info(" <M^2>-<M>^2 = %s", mean_M2 - mean_M)
        logger.info(" Volume = %s", volume)

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

        if start > len(self.data):
            raise ValueError(" ERROR :: start is larger than data length")
        if end > len(self.data):
            raise ValueError(" ERROR :: end is larger than data length")

        if end == -1:
            calc_data = self.data[start:, 1:]
        else:
            calc_data = self.data[start:end, 1:]
        logger.info("length calc_data :: %s", len(calc_data))

        SAMPLE = 100  # !! hard code
        # インデックスのリストを numpy で作成（0 を除外）
        indices = np.arange(1, len(calc_data))
        indices = indices[indices % SAMPLE == 0]
        logger.debug("indices = %s", indices)
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
    def plot_time_vs_dielconst(cls, df: pd.DataFrame) -> int:
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
        ax.set_xlabel("Time [ns]", fontsize=22)
        ax.set_ylabel("Dielconst", fontsize=22)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc="upper right", fontsize=15)
        fig.savefig("time_dielconst.pdf")
        fig.delaxes(ax)
        return 0

    def plot_total_dipole_vs_time(self) -> int:
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
        ax.set_xlabel("Time [ps]", fontsize=22)
        ax.set_ylabel("Dipole [D]", fontsize=22)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.legend(loc="upper right", fontsize=15)
        fig.savefig("time_totaldipole.pdf")
        fig.delaxes(ax)
        return 0

    def _calculate_coefficient_dielectricfunction(self) -> float:
        """calculate coeff 1/epsilon_0kTV (unitless)"""
        eps0 = 8.8541878128e-12
        debye = 3.33564e-30
        kb = 1.38064852e-23
        kbT = kb * self.temperature
        # !! 3で割らないようになっているのは，autocorrのところで平均を取るようにしているから．
        eps_0_coeff: float = (debye**2) / (self.get_volume() * kbT * eps0)
        return eps_0_coeff

    @DeprecationWarning
    def calculate_dipoleautocorrelation(self) -> NDArray[np.float64]:
        acf_mean = autocorr(autocorr_scipy).compute_autocorr2d(self.data)
        return acf_mean

    def calculate_fft_from_dipole(
        self,
        window_type: Literal["hann", "hamming", "blackman", "gaussian", "ma"] | None,
        method_dipole: Literal["direct", "derivative"],
    ) -> pd.DataFrame:
        """Calculate Fourier transform of dipole auto-correlation function"""
        if method_dipole == "direct":
            dipole_array = self.data - self.get_mean_dipole()  # calculate M-<M>
            logger.info("mean_dipole = %s", self.get_mean_dipole())
        elif method_dipole == "derivative":
            # dM/dt : with time of ps = 1/THz
            dipole_array = np.diff(self.data, axis=0) / (self.timestep / 1000)
            logger.info("mean_derivative_dipole = %s", np.mean(dipole_array, axis=0))
            dipole_array = dipole_array - np.mean(
                dipole_array, axis=0
            )  # calculate dotM-<dotM>
        # calculate acf
        acf_mean_array: NDArray[np.float64] = autocorr(
            autocorr_numpy
        ).compute_autocorr2d(x=dipole_array, ifmean=True)
        logger.info(
            "len(dipole)=%d :: len(acf)=%d", len(dipole_array), len(acf_mean_array)
        )
        # apply window function to acf
        acf_with_window_array: NDArray[np.float64] = apply_windowfunc_oneside(
            acf_mean_array, window_type
        )
        # calculate FFT
        df_fft: pd.DataFrame = fft.calculate_fft_dielfunction(
            acf_with_window_array, self.timestep
        )
        return df_fft

    def _apply_dielectric_response(
        self,
        df: pd.DataFrame,
        response_col: str,
        method_dipole: Literal["direct", "derivative"],
        scale: float = 1.0,
        const: float = 0,
    ) -> pd.Series:
        """
        Apply system-specific dielectric coefficient to real or imaginary part of FFT result.

        Args:
            df: DataFrame containing FFT results.
            freq_col: Name of the frequency column (usually "freq_thz").
            response_col: Column name of the real or imag part to scale.
            method_dipole: Whether to scale by angular frequency or inverse.
            scale: Optional additional scaling factor.
            const: Optional additional constant.

        Returns:
            pd.Series: Scaled dielectric response.
        """
        coeff = self._calculate_coefficient_dielectricfunction()
        angular_freq = 2 * np.pi * df["freq_thz"]

        if method_dipole == "direct":
            return df[response_col] * angular_freq * coeff * scale + const
        if method_dipole == "derivative":
            return df[response_col] / angular_freq * coeff * scale + const
        raise ValueError("method_dipole must be 'direct' or 'derivative'")

    def calculate_dielfunction(
        self,
        eps_inf: float = 1.0,
        method_dipole: Literal["direct", "derivative"] = "direct",
        window_type: (
            Literal["hann", "hamming", "blackman", "gaussian", "ma"] | None
        ) = "hann",
    ) -> pd.DataFrame:
        """calculate real & imaginary part of dielectric function

        For derivative, see zhang2020Deep
        """
        [eps_0, _, _] = self.calculate_dielconst(eps_inf)
        logger.info("coeff = %s", self._calculate_coefficient_dielectricfunction())
        df_fft = self.calculate_fft_from_dipole(
            window_type, method_dipole=method_dipole
        )
        df_fft["diel_real"] = self._apply_dielectric_response(
            df_fft, "imag", method_dipole, const=eps_0
        )
        df_fft["diel_imag"] = self._apply_dielectric_response(
            df_fft, "real", method_dipole
        )
        return df_fft[["freq_thz", "freq_kayser", "diel_real", "diel_imag"]]

    def calculate_dielfunction_imag(
        self,
        method_fft: Literal["direct", "wk"] = "direct",
        method_dipole: Literal["direct", "derivative"] = "direct",
    ) -> pd.DataFrame:
        """Calculate imaginary part of the dielectroc function"""
        if method_fft == "direct":
            df_fft = self.calculate_fft_from_dipole("hann", method_dipole=method_dipole)
            df_fft["diel_imag"] = self._apply_dielectric_response(
                df_fft, "real", method_dipole
            )
            return df_fft[["freq_thz", "freq_kayser", "diel_imag"]]
        if method_fft == "wk":
            # apply window function to dipole
            dipole_with_window_array = apply_windowfunc_twoside(self.data, "hann")
            # calculate dipole FFT
            dipole_fft = fft.calculate_fft_vdos(dipole_with_window_array, self.timestep)
            # TODO :: abs(fft)**2 is the reference expression
            # TODO :: which is correct, two sided FFT (now) or one sided FFT and abs(fft)**2?
            return pd.DataFrame(
                {
                    "freq_thz": dipole_fft["freq_thz"],
                    "freq_kayser": dipole_fft["freq_kayser"],
                    "diel_imag": (
                        dipole_fft["vdos"] ** 2
                        * (2 * np.pi * dipole_fft["freq_thz"])
                        * self._calculate_coefficient_dielectricfunction()
                        / 2.0
                    ),  # Winner-Khinchin
                }
            )
        raise ValueError("method_fft should be 'direct' or 'wk'.")

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
            df_fft["alphan"] = (
                df_fft["diel_imag"] * (2 * np.pi * df_fft["freq_thz"]) / speedoflight
            )
        elif method_dipole == "derivative":
            df_fft = self.calculate_fft_from_dipole("hann", method_dipole="derivative")
            df_fft["alphan"] = (
                df_fft["real"]
                * self._calculate_coefficient_dielectricfunction()
                / speedoflight
            )
        else:
            raise ValueError("method should be direct or derivative")
        return df_fft[["freq_kayser", "freq_thz", "alphan"]]

    def calculate_refractiveindex(
        self, **kwargs: Any
    ) -> pd.DataFrame:  # include epsilon/alpha
        """Calculate refractive index (& dielectric function)"""
        speedoflight = 0.03  # in cm-1THz
        df: pd.DataFrame = self.calculate_dielfunction(**kwargs)
        # 本来はここはマイナスだが，プラスで計算しておくと（kappaもマイナスで定義されているので）全体として辻褄が合うようになっている．
        epsilon = df["diel_real"] + 1j * df["diel_imag"]
        refractive_index = []

        for i in epsilon:
            a, b = cmath.polar(i)
            refractive_index.append(cmath.rect(np.sqrt(a), b / 2))

        df["real_ref_index"] = [a.real for a in refractive_index]
        df["imag_ref_index"] = [a.imag for a in refractive_index]
        df["alpha"] = (
            df["imag_ref_index"] * (2 * np.pi * df["freq_thz"]) * 2 / speedoflight
        )
        return apply_windowfunc_conv_df(df)

    def calculate_refindex_and_absorption(self, **kwargs: Any) -> int:
        """wrapper for calculate_refractiveindex and calculate_absorption"""
        df_refindex: pd.DataFrame = self.calculate_refractiveindex(**kwargs)
        df_absorption: pd.DataFrame = self.calculate_absorption("direct")
        return 0


def apply_windowfunc_conv_df(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if column not in {"freq_kayser", "freq_thz"}:
            df[column] = apply_windowfunc_conv(
                np.ndarray(df[column]), 10
            )  # FIXME: hard code
    return df


# functions
def calculate_dielconst(totaldipole: TotalDipole, eps_inf: float = 1.0) -> list[float]:
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
