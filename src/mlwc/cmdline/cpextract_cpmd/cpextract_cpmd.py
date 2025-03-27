#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides a set of tools to extract and analyze data from CP.x (Car-Parrinello molecular dynamics) output files.
It defines subcommands for CPextract.py to process energy, force, dipole moment, and trajectory data.

This module contains classes and functions for plotting energies, forces, and dipole moments from CP.x output files.
It also includes functions for manipulating trajectory files and calculating classical charges.

"""
import numpy as np
import ase.units
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mlwc.dataio.cpmd.read_traj_cpmd import raw_cpmd_read_unitcell_vector
from quadrupole.calc_fourier import calc_fourier
from mlwc.dataio.cpmd.read_traj_cpmd import CPMD_ReadPOS
from mlwc.include.mlwc_logger import setup_cmdline_logger
logger = setup_cmdline_logger(__name__)


class Plot_energies:
    """
    Plots energies from a CP.x output file.

    This class reads an energy file from a CP.x (Car-Parrinello molecular dynamics) simulation
    and generates plots of various energy components as a function of time step.
    It can plot the total energy, temperature, and energy histogram.

    Attributes
    ----------
    __filename : str
        The name of the energy file to read.
    data : numpy.ndarray
        The energy data read from the file.

    Notes
    -----
    Short Legend and Physical Units in the Output:
    NFI    [int]          - step index
    EKINC  [HARTREE A.U.] - kinetic energy of the fictitious electronic dynamics
    TEMPH  [K]            - Temperature of the fictitious cell dynamics
    TEMP   [K]            - Ionic temperature
    ETOT   [HARTREE A.U.] - Scf total energy (Kohn-Sham hamiltonian)
    ENTHAL [HARTREE A.U.] - Enthalpy ( ETOT + P * V )
    ECONS  [HARTREE A.U.] - Enthalpy + kinetic energy of ions and cell
    ECONT  [HARTREE A.U.] - Constant of motion for the CP lagrangian

    Examples
    --------
    >>> energies_plotter = Plot_energies("EVP.dat")
    >>> energies_plotter.process()
    0
    """

    def __init__(self, energies_filename):
        """
        Initializes the Plot_energies class.

        Reads the energy file and initializes the data attribute.

        Parameters
        ----------
        energies_filename : str
            The name of the energy file to read.

        Raises
        ------
        ValueError
            If the energy file does not exist.

        Returns
        -------
        None

        Examples
        --------
        >>> energies_plotter = Plot_energies("EVP.dat")
        >>> print(energies_plotter.data)
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
        """
        self.__filename = energies_filename
        self.data = np.loadtxt(self.__filename)
        self.data = np.loadtxt(self.__filename)

        import os
        if not os.path.isfile(self.__filename):
            raise ValueError(
                " ERROR :: "+str(self.__filename)+" does not exist !!")

    def plot_Energy(self):
        """Plots the total energy as a function of time step.

        Generates a plot of the total energy (column 4 of the energy file)
        as a function of time step (column 0 of the energy file).

        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if the plot is generated successfully.

        Examples
        --------
        >>> energies_plotter = Plot_energies("EVP.dat")
        >>> result = energies_plotter.plot_Energy()
        >>> print(result)
        0
        """
        logger.info(" ---------- ")
        logger.info(" energy plot :: column 0 & 4(ECLASSICAL) ")
        logger.info(" ---------- ")
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(self.data[:, 0], self.data[:, 4] /
                ase.units.Hartree, label=self.__filename, lw=3)  # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Timestep"  # "Time $\mathrm{ps}$"
        ylabel = "Energy[eV]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename+"_E.pdf")
        fig.delaxes(ax)
        return 0

    def plot_energy_histgram(self):
        """Plots a histogram of the total energy.

        Generates a histogram of the total energy (column 4 of the energy file)
        to visualize the distribution of energy values.

        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if the plot is generated successfully.

        Examples
        --------
        >>> energies_plotter = Plot_energies("EVP.dat")
        >>> result = energies_plotter.plot_energy_histgram()
        >>> print(result)
        0
        """
        logger.info(" ---------- ")
        logger.info(" energy plot of histgram :: column 0 & 4(ECLASSICAL) ")
        logger.info(" ---------- ")
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.hist((self.data[:, 4]-np.average(self.data[:, 4]))/ase.units.Hartree*1000, bins=100,
                label=self.__filename+"average={}".format(np.average(self.data[:, 4]))+"eV")  # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Energy[meV]"  # "Time $\mathrm{ps}$"
        ylabel = "number"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename+"_Ehist.pdf")
        fig.delaxes(ax)
        return 0

    def plot_Temperature(self):
        """Plots the temperature as a function of time step.

        Generates a plot of the temperature (column 2 of the energy file)
        as a function of time step (column 0 of the energy file).

        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if the plot is generated successfully.

        Examples
        --------
        >>> energies_plotter = Plot_energies("EVP.dat")
        >>> result = energies_plotter.plot_Temperature()
        >>> print(result)
        0
        """
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(self.data[:, 0], self.data[:, 2],
                label=self.__filename, lw=3)  # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Timesteps"  # "Time $\mathrm{ps}$"
        ylabel = "Temperature [K]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        fig.savefig(self.__filename+"_T.pdf")
        fig.delaxes(ax)
        return 0

    def process(self):
        """Processes the energy data and generates plots.

        Calls the methods to plot the total energy, temperature, and energy histogram.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> energies_plotter = Plot_energies("EVP.dat")
        >>> energies_plotter.process()
        """
        logger.info(" ==========================")
        logger.info(" Reading {:<20}   :: making Temperature & Energy plots ".format(
            self.__filename))
        logger.info("")
        self.plot_Energy()
        self.plot_Temperature()
        self.plot_energy_histgram()


class Plot_forces:
    """Plots forces from a CP.x output file.

    This class reads a force file from a CP.x (Car-Parrinello molecular dynamics) simulation
    and generates plots of force components as a function of time step.

    Attributes
    ----------
    __filename : str
        The name of the force file to read.
    data : numpy.ndarray
        The force data read from the file.

    Notes
    -----
    Short Legend and Physical Units in the Output:
    NFI    [int]          - step index
    EKINC  [HARTREE A.U.] - kinetic energy of the fictitious electronic dynamics
    TEMPH  [K]            - Temperature of the fictitious cell dynamics
    TEMP   [K]            - Ionic temperature
    ETOT   [HARTREE A.U.] - Scf total energy (Kohn-Sham hamiltonian)
    ENTHAL [HARTREE A.U.] - Enthalpy ( ETOT + P * V )
    ECONS  [HARTREE A.U.] - Enthalpy + kinetic energy of ions and cell
    ECONT  [HARTREE A.U.] - Constant of motion for the CP lagrangian

    Examples
    --------
    >>> forces_plotter = Plot_forces("FORCES.dat")
    >>> forces_plotter.process()
    """

    def __init__(self, ftrajectory_filename):
        """Initializes the Plot_forces class.

        Reads the force trajectory file and initializes the data attribute.

        Parameters
        ----------
        ftrajectory_filename : str
            The name of the force trajectory file to read.

        Raises
        ------
        FileNotFoundError
            If the force trajectory file does not exist.

        Returns
        -------
        None

        Examples
        --------
        >>> forces_plotter = Plot_forces("FORCES.dat")
        >>> print(forces_plotter.data)
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]]
        """
        self.__filename = ftrajectory_filename
        self.data = np.loadtxt(self.__filename)

        import os
        if not os.path.isfile(self.__filename):
            raise FileNotFoundError(
                " ERROR :: "+str(self.__filename)+" does not exist !!")

    def plot_Force(self):
        """Plots a histogram of the force components.

        Generates a histogram of the x, y, and z components of the force (columns 7, 8, and 9 of the force file)
        to visualize the distribution of force values.

        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if the plot is generated successfully.

        Examples
        --------
        >>> forces_plotter = Plot_forces("FORCES.dat")
        >>> result = forces_plotter.plot_Force()
        >>> print(result)
        0
        """
        logger.info(" ---------- ")
        logger.info(" Force histgram plot :: column 7-9 ")
        logger.info(" ---------- ")
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        HaBohr_to_eV_Ang = 51.42208619083232
        ax.hist(self.data[:, 7]*HaBohr_to_eV_Ang, bins=100,
                label=self.__filename+"_x", alpha=0.5)
        ax.hist(self.data[:, 8]*HaBohr_to_eV_Ang, bins=100,
                label=self.__filename+"_y",  alpha=0.5)
        ax.hist(self.data[:, 9]*HaBohr_to_eV_Ang, bins=100,
                label=self.__filename+"_z", alpha=0.5)

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Force [eV/Ang]"  # "Time $\mathrm{ps}$"
        ylabel = "number"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename+"_F.pdf")
        fig.delaxes(ax)
        return 0

    def process(self):
        """Processes the force data and generates plots.

        Calls the method to plot the force histogram.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> forces_plotter = Plot_forces("FORCES.dat")
        >>> forces_plotter.process()
        """
        logger.info(" ==========================")
        logger.info(" Reading {:<20}   :: making Temperature & Energy plots ".format(
            self.__filename))
        logger.info("")
        self.plot_Force()


def dfset(filename, cpmdout, interval_step: int, start_step: int = 0):
    """Reads trajectory and force data to create a DFSET_export file for CPMD.x.

    Parameters
    ----------
    filename : str
        The name of the trajectory file.
    cpmdout : str
        The name of the CPMD output file.
    interval_step : int
        The interval step for exporting data.
    start_step : int, optional
        The starting step for exporting data, by default 0.

    Returns
    -------
    int
        0 if the DFSET_export file is created successfully.

    Examples
    --------
    >>> dfset("TRAJECTORY", "CPMD.OUT", 100, 0)
    0
    """
    traj = cpmd.read_traj_cpmd.CPMD_ReadPOS(filename=filename, cpmdout=cpmdout)
    # import forces
    traj.set_force_from_file(filename)
    traj.export_dfset_pwin(interval_step, start_step)
    logger.info(" ")
    logger.info(" make DFSET_export...")
    logger.info(" ")
    return 0


class DIPOLE:
    """Calculates the total dipole moment using classical charges.

    This class calculates the total dipole moment of a system using classical charges
    obtained from a charge file and atomic positions from a trajectory file.

    Attributes
    ----------
    _filename : str
        The name of the XYZ trajectory file.
    _charge_filename : str
        The name of the file containing the classical charges.
    _traj : list of ase.Atoms
        The trajectory data read from the XYZ file.
    _charge : numpy.ndarray
        The classical charges read from the charge file.
    _NUM_ATOM_PER_MOL : int
        The number of atoms per molecule.
    _NUM_MOL : int
        The number of molecules in the system.
    _charge_system : numpy.ndarray
        The classical charges tiled to match the number of molecules in the system.

    Examples
    --------
    >>> dipole = DIPOLE("IONS+CENTERS.xyz", "CHARGE")
    >>> dipole_list = dipole.calc_dipole()
    >>> print(dipole_list)
    [[0, 1.0, 2.0, 3.0], [1, 4.0, 5.0, 6.0]]
    """

    def __init__(self, filename: str, charge_filename: str):
        """Initializes the DIPOLE class.

        Reads the XYZ trajectory file and the charge file, and initializes the attributes.

        Parameters
        ----------
        filename : str
            The name of the XYZ trajectory file.
        charge_filename : str
            The name of the file containing the classical charges.

        Raises
        ------
        FileNotFoundError
            If the XYZ trajectory file does not exist.

        Returns
        -------
        None

        """
        self._filename = filename  # xyz
        self._charge_filename = charge_filename  # charge
        import os
        if not os.path.isfile(self._filename):
            print(" ERROR :: "+str(self._filename)+" does not exist !!")
            print(" ")
            return 1

        # read xyz
        import ase
        import ase.io
        import dataio.cpmd.read_traj_cpmd
        print(" READING TRAJECTORY... This may take a while, be patient.")
        self._traj, wannier_list = io.cpmd.read_traj_cpmd.raw_xyz_divide_aseatoms_list(
            self._filename)
        print(f"FINISH READING TRAJECTORY... {len(self._traj)} steps")

        # read charge
        self._charge = np.loadtxt(self._charge_filename)
        print(f"FINISH READING CHARGE... {len(self._charge)} atoms")
        print(self._charge)
        print(" ==========================")

        self._NUM_ATOM_PER_MOL: int = len(self._charge)
        if len(self._traj[0]) % self._NUM_ATOM_PER_MOL != 0:
            print(
                "ERROR: Number of atoms in the first step is not divisible by the number of atoms per molecule")
            return 1
        self._NUM_MOL: int = int(
            self._traj[0].get_number_of_atoms()/self._NUM_ATOM_PER_MOL)
        print(f"NUM_MOL :: {self._NUM_MOL}")
        self._charge_system = np.tile(
            self._charge, self._NUM_MOL)  # NUM_MOL回繰り返し

    def calc_dipole(self):
        """Calculates the total dipole moment for each frame in the trajectory.

        Calculates the total dipole moment of the system for each frame in the trajectory
        using the classical charges and atomic positions. The dipole moment is calculated in Debye.

        Returns
        -------
        list of list
            A list of dipole moments for each frame in the trajectory. Each element in the list
            is a list containing the frame index and the x, y, and z components of the dipole moment.

        Examples
        --------
        >>> dipole = DIPOLE("IONS+CENTERS.xyz", "CHARGE")
        >>> dipole_list = dipole.calc_dipole()
        >>> print(dipole_list)
        [[0, 1.0, 2.0, 3.0], [1, 4.0, 5.0, 6.0]]
        """
        # 単位をe*AngからDebyeに変換
        from mlwc.include.constants import constant
        # Debye   = 3.33564e-30
        # charge  = 1.602176634e-019
        # ang      = 1.0e-10
        coef = constant.Ang*constant.Charge/constant.Debye
        import numpy as np
        dipole_list = []
        for counter, atoms in enumerate(self._traj):  # loop over MD step
            # self._charge_systemからsystem dipoleを計算
            tmp_dipole = coef * \
                np.einsum("i,ij->j", self._charge_system,
                          atoms.get_positions())
            dipole_list.append(
                [counter, tmp_dipole[0], tmp_dipole[1], tmp_dipole[2]])
        # 計算されたdipoleを保存する．
        np.savetxt("classical_dipole.txt", np.array(dipole_list),
                   header=" index dipole_x dipole_y dipole_z")
        return dipole_list


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

    def plot_dipole(self):
        """Plots the dipole moment data.

        Generates a plot of the x, y, and z components of the total dipole moment as a function of time.

        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if the plot is generated successfully.

        """
        import os
        if not os.path.isfile("DIPOLE"):
            raise FileNotFoundError(
                " ERROR :: "+str("DIPOLE")+" does not exist !!")
        if stdout != "":
            # from ase.io import read
            from dataio.cpmd.read_traj_cpmd import raw_cpmd_get_timestep
            self.timestep = raw_cpmd_get_timestep(
                stdout)/1000  # fs単位で読み込むので，psへ変換
            logger.info(" timestep [ps] :: {}".format(self.timestep))
        else:
            self.timestep = 0.001  # ps単位で，defaultを1fs=0.001psにしておく

    def plot_dipole(self):
        # figure, axesオブジェクトを作成
        fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
        ax.plot(self.data[:, 0]*self.timestep, self.data[:, 4],
                label=self.__filename+"_x", lw=3)  # 描画
        ax.plot(self.data[:, 0]*self.timestep, self.data[:, 5],
                label=self.__filename+"_y", lw=3)  # 描画
        ax.plot(self.data[:, 0]*self.timestep, self.data[:, 6],
                label=self.__filename+"_z", lw=3)  # 描画

        # 各要素で設定したい文字列の取得
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        xlabel = "Time $\mathrm{ps}$"
        ylabel = "Dipole [D/Volume?]"

        # 各要素の設定を行うsetコマンド
        ax.set_xlabel(xlabel, fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename+"_Dipole.pdf")
        fig.delaxes(ax)
        return 0

    def plot_dielec(self):
        '''
        誘電関数の計算，及びそのプロットを行う．
        体積による規格化や，前にかかる係数などは何も処理しない．

        ---------
        TODO :: ちゃんとDIPOLEファイルでの係数の定義を突き止める．
        '''

        N = int(np.shape(self.data[:, 0])[0]/2)
        print("nlag :: ", N)

        # 自己相関関数を求める
        acf_x = sm.tsa.stattools.acf(self.data[:, 4], nlags=N, fft=False)
        acf_y = sm.tsa.stattools.acf(self.data[:, 5], nlags=N, fft=False)
        acf_z = sm.tsa.stattools.acf(self.data[:, 6], nlags=N, fft=False)

        # time in ps
        time = self.data[:, 0]*self.timestep  # (in ps)

        # eps_n2 = 1.333**2
        eps_0 = 1.0269255134097743
        eps_n2 = 3.1**2   # eps_n2=eps_inf^2 ?
        eps_inf = 1.0     # should be fixed
        # eps_0 = pred_eps
        # data=acfs["acf"].to_numpy()
        fft_data = (acf_x+acf_y+acf_z)/3

        TIMESTEP = (time[1]-time[0])  # psec.
        logger.info("TIMESTEP [fs] :: ", TIMESTEP*1000)

        rfreq, ffteps1, ffteps2 = calc_fourier(
            fft_data, eps_0, eps_n2, TIMESTEP)

        # convert THz to cm-1
        kayser = rfreq * 33.3

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

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        # https://www.delftstack.com/ja/howto/matplotlib/how-to-set-tick-labels-font-size-in-matplotlib/#ax.tick_paramsaxis-xlabelsize-%25E3%2581%25A7%25E7%259B%25AE%25E7%259B%259B%25E3%2582%258A%25E3%2583%25A9%25E3%2583%2599%25E3%2583%25AB%25E3%2581%25AE%25E3%2583%2595%25E3%2582%25A9%25E3%2583%25B3%25E3%2583%2588%25E3%2582%25B5%25E3%2582%25A4%25E3%2582%25BA%25E3%2582%2592%25E8%25A8%25AD%25E5%25AE%259A%25E3%2581%2599%25E3%2582%258B
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper right", fontsize=15)

        # pyplot.savefig("eps_real2.pdf",transparent=True)
        # plt.show()
        fig.savefig(self.__filename+"_Dielec.pdf")
        fig.delaxes(ax)

        return 0

    def process(self):
        logger.info(" ==========================")
        logger.info(
            " Reading {:<20}   :: making Dipole plots ".format(self.__filename))
        logger.info("")
        self.plot_dipole()
        self.plot_dielec()


def delete_wfcs_from_ionscenter(filename: str = "IONS+CENTERS.xyz", stdout: str = "bomd-wan.out", output: str = "IONS_only.xyz"):
    '''
    XYZからions_centers.xyzを削除して，さらにsupercell情報を付与する．
    '''

    # トラジェクトリを読み込む
    list_atoms = ase.io.read(filename, index=":")

    # もしsupercell情報を持っていればそれを採用する．
    if list_atoms[0].get_cell() != "":
        UNITCELL_VECTORS = list_atoms[0].get_cell()
    else:
        # supercellを読み込み
        UNITCELL_VECTORS = raw_cpmd_read_unitcell_vector(
            stdout)

    # 出力するase.atomsのリスト
    list_atoms_withoutX = []

    # ワニエの座標を廃棄する．
    for config_num, atom in enumerate(list_atoms):
        # for debug
        # 配列の原子種&座標を取得
        atom_list = list_atoms[config_num].get_chemical_symbols()
        coord_list = list_atoms[config_num].get_positions()

        atom_list_tmp = []
        coord_list_tmp = []
        for i, j in enumerate(atom_list):
            if j != "X":  # 原子がXだったらappendしない
                atom_list_tmp.append(atom_list[i])
                coord_list_tmp.append(coord_list[i])

        CM = ase.Atoms(atom_list_tmp,
                       positions=coord_list_tmp,
                       cell=UNITCELL_VECTORS,
                       pbc=[1, 1, 1])
        list_atoms_withoutX.append(CM)

    # 保存
    ase.io.write(output, list_atoms_withoutX)
    logger.info("==========")
    logger.info(" a trajectory is saved to IONS_only.xyz")
    logger.info(" ")

    return 0


def add_supercellinfo(filename: str = "IONS+CENTERS.xyz", stdout: str = "bomd-wan.out", output: str = "IONS+CENTERS+cell.xyz"):
    '''
    XYZにstdoutから読み込んだsupercell情報を付与する．

    notes
    --------
    XYZではなく，場合によってはTRAJECTORYを読み込みたい場合があるのでその場合に対応している．
    '''

    if filename == "TRAJECTORY":
        logger.warning(" warning :: file name is TRAJECTORY. ")
        answer_atomslist = CPMD_ReadPOS(filename, cpmdout)

    else:
        # トラジェクトリを読み込む
        test_read_trajecxyz = ase.io.read(filename, index=":")

        # supercellを読み込み
        # TODO :: stdout以外からも読み込めると良い．
        UNITCELL_VECTORS = raw_cpmd_read_unitcell_vector(stdout)

        # 出力するase.atomsのリスト
        answer_atomslist = []

        # trajectoryを読み込んでaseへ変換
        for config_num, atom in enumerate(test_read_trajecxyz):
            # for debug
            # 配列の原子種&座標を取得
            atom_list = test_read_trajecxyz[config_num].get_chemical_symbols()
            coord_list = test_read_trajecxyz[config_num].get_positions()

            CM = ase.Atoms(atom_list,
                           positions=coord_list,
                           cell=UNITCELL_VECTORS,
                           pbc=[1, 1, 1])
            answer_atomslist.append(CM)

    # 保存
    ase.io.write(output, answer_atomslist)
    logger.info("==========")
    logger.info(" a trajectory is saved to ", output)
    logger.info(" ")

    return 0


# --------------------------------
# 以下CPextract.pyからロードする関数たち
# --------------------------------


def command_cpmd_energy(args):
    EVP = Plot_energies(args.Filename)
    EVP.process()
    return 0


def command_cpmd_force(args):
    EVP = Plot_forces(args.Filename)
    EVP.process()
    return 0


def command_cpmd_dfset(args):
    dfset(args.Filename, args.cpmdout, args.interval, args.start)
    return 0


def command_cpmd_dipole(args):
    '''
    plot DIPOLE file
    '''
    Dipole = Plot_dipole(args.Filename, args.stdout)
    Dipole.process()
    return 0


def command_cpmd_xyz(args):
    '''
    make IONS_only.xyz from IONS+CENTERS.xyz
    '''
    delete_wfcs_from_ionscenter(args.Filename, args.stdout, args.output)
    return 0


def command_cpmd_xyzsort(args):
    """ cpmdのsortされたIONS+CENTERS.xyzを処理する．


    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    import cpmd.converter_cpmd
    cpmd.converter_cpmd.back_convert_cpmd(
        args.input, args.output, args.sortfile)
    return 0


def command_cpmd_addlattice(args):
    """cpmdで得られたxyzにstdoutの格子定数情報を付加する．


    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    add_supercellinfo(args.input, args.stdout, args.output)
    return 0


def command_cpmd_charge(args):  # 古典電荷
    dipole = DIPOLE(args.Filename, args.charge)
    dipole.calc_dipole()
    return 0
