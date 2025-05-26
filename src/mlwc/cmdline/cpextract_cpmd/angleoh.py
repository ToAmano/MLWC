"""
This script calculates the angle between O-H bonds in a molecular dynamics trajectory.
It computes the auto-correlation function (ACF) and its Fourier transform (FT)
to analyze the vibrational properties of the O-H bonds.
"""

import os

import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import __version__
import mlwc.bond.atomtype
import mlwc.fourier.hydrogenbond
from mlwc.include.file_io import to_csv_with_comment
from mlwc.include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger(__name__)


class ANGLEOH:
    """
    Calculates the angle between O-H bonds in a molecular dynamics trajectory.

    This class reads a trajectory file and a molecule file, and calculates the
    angle between O-H bonds for each frame in the trajectory. It then computes
    the auto-correlation function (ACF) and its Fourier transform (FT) to
    analyze the vibrational properties of the O-H bonds.

    Attributes
    ----------
    __filename : str
        The name of the trajectory file (e.g., xyz file).
    __initial_step : int
        The initial step to start calculating the mean-square displacement.
    __molfile : str
        The name of the molecule file (e.g., .mol or .itp file).
    _timestep : float
        The time step of the trajectory in femtoseconds (fs).
    _NUM_ATOM_PER_MOL : int
        The number of atoms per molecule, including water centers (WC) and
        boundary centers (BC).
    itp_data : ml.atomtype.ITPData
        The data read from the molecule file.
    _traj : ase.Atoms
        The molecular dynamics trajectory read using ASE.
    NUM_MOL : int
        The number of molecules in the trajectory.

    Examples
    --------
    >>> angleoh = ANGLEOH("traj.xyz", "mol.mol", 1.0, 3)
    >>> angleoh.calc_angleoh()
    0
    """

    def __init__(
        self,
        filename: str,
        molfile: str,
        timestep: float,
        NUM_ATOM_PER_MOL: int,
        initial_step: int = 1,
    ):
        """
        Initializes the ANGLEOH class.

        Parameters
        ----------
        filename : str
            The name of the trajectory file (e.g., xyz file).
        molfile : str
            The name of the molecule file (e.g., .mol or .itp file).
        timestep : float
            The time step of the trajectory in femtoseconds (fs).
        NUM_ATOM_PER_MOL : int
            The number of atoms per molecule, including water centers (WC) and
            boundary centers (BC).
        initial_step : int, optional
            The initial step to start calculating the mean-square displacement, by default 1.

        Raises
        ------
        FileNotFoundError
            If the trajectory file or the molecule file does not exist.
        ValueError
            If the initial_step is less than 1.
        ValueError
            If the molecule file does not end with .itp or .mol.
        AssertionError
            If the number of atoms in the first step is not divisible by the
            number of atoms per molecule.

        Returns
        -------
        None
        """
        # TODO:: remove input files and get xyz, itp_data instead
        self.__filename = filename  # xyz
        self.__initial_step = initial_step  # initial step to calculate msd
        self.__molfile = molfile  # .mol
        # timestep in [fs]
        self._timestep = timestep  # timestep in [fs]
        self._NUM_ATOM_PER_MOL = NUM_ATOM_PER_MOL  # including WCs & BCs

        if self.__initial_step < 1:
            raise ValueError("ERROR: initial_step must be larger than 1")

        # * read itp/mol
        # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
        if self.__molfile.endswith(".itp"):
            self.itp_data = mlwc.bond.atomtype.read_itp(self.__molfile)
        elif self.__molfile.endswith(".mol"):
            self.itp_data = mlwc.bond.atomtype.ReadMolFile(self.__molfile)
        else:
            raise ValueError("ERROR :: itp_filename should end with .itp or .mol")

        # read xyz
        logger.info(" READING TRAJECTORY... This may take a while, be patient.")
        self._traj = ase.io.read(self.__filename, index=":")
        logger.info(" FINISH READING TRAJECTORY... :: len(traj) = %d", len(self._traj))
        #
        self.NUM_MOL = len(self._traj[0]) // self._NUM_ATOM_PER_MOL
        assert (
            len(self._traj[0]) % self._NUM_ATOM_PER_MOL == 0
        ), "ERROR: Number of atoms in the first step is not divisible by the number of atoms per molecule"
        logger.info(" NUM_MOL == %d", self.NUM_MOL)

    def calc_angleoh(self) -> list[pd.DataFrame]:
        """
        Calculates the angle between O-H bonds and computes the ACF and FT.

        This method calculates the angle between O-H bonds for each frame in
        the trajectory. It then computes the auto-correlation function (ACF)
        and its Fourier transform (FT) to analyze the vibrational properties
        of the O-H bonds.

        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if the calculation is successful.

        Examples
        --------
        >>> angleoh = ANGLEOH("traj.xyz", "mol.mol", 1.0, 3)
        >>> angleoh.calc_angleoh()
        0
        """
        # O/H atomic index
        o_list = []
        h_list = []
        for [a, b] in self.itp_data._bonds["OH_1_bond"]:
            if a in self.itp_data.o_list:
                o_list.append(a)
                h_list.append(b)
            elif a in self.itp_data.h_list:
                h_list.append(a)
                o_list.append(b)
        logger.info(o_list)
        logger.info(h_list)

        # H/O atoms list in OH bonds
        hydrogen_list: list = [
            self._NUM_ATOM_PER_MOL * mol_id + atom_id
            for mol_id in range(self.NUM_MOL)
            for atom_id in h_list
        ]
        oxygen_list: list = [
            self._NUM_ATOM_PER_MOL * mol_id + atom_id
            for mol_id in range(self.NUM_MOL)
            for atom_id in o_list
        ]
        # calculate OH vector
        bond_vectors = mlwc.fourier.hydrogenbond.calc_oh(
            self._traj, oxygen_list, hydrogen_list
        )
        np.save(self.__filename + "_oh_angle_list.npy", bond_vectors)
        # 全ての時系列に対して自己相関を計算 (axis=1で各行に対して自己相関を計算)
        # 'same' モードで時系列の長さを維持
        # !! numpy correlate does not support FFT
        correlations = np.apply_along_axis(
            lambda x: scipy.signal.correlate(x, x, mode="full"),
            axis=0,
            arr=bond_vectors,
        )
        correlations = np.sum(correlations, axis=2)  # inner dot

        # average ACF  (axis=1で全ての時系列に対する平均を取る)
        mean_correlation = np.mean(correlations, axis=1)[len(bond_vectors) - 1 :]  # acf
        df_acf: pd.DataFrame = mlwc.fourier.hydrogenbond.make_df_acf(
            mean_correlation, self._timestep
        )

        # Fourier Transform
        df_roo: pd.DataFrame = mlwc.fourier.hydrogenbond.calc_lengthcorr(
            mean_correlation, self._timestep
        )

        return df_acf, df_roo

    def save_files(self, df_acf: pd.DataFrame, df_roo: pd.DataFrame) -> None:
        comment: str = f"""
        # File generated by CPextract.py cpmd angleoh version {__version__.__version__}.
        # Parameters: filename={self.__filename}, molfilename={self.__molfile}
        # Parameters: initial_step={self.__initial_step}, timestep={self._timestep}
        # Data below:\n
        """
        to_csv_with_comment(df_acf, comment, self.__filename + "_oh_acf.csv")
        logger.info(" acf is saved as " + self.__filename + "_oh_acf.csv")
        to_csv_with_comment(df_roo, comment, self.__filename + "_oh_ft.csv")
        logger.info(" ft is saved as " + self.__filename + "_oh_ft.csv")

    def visualize_roo(self, df_roo: pd.DataFrame) -> None:
        """visualize data"""
        plt.plot(df_roo["freq_kayser"], df_roo["roo"], label="data")
        plt.legend(fontsize=15)
        plt.xlabel("cm-1", fontsize=15)
        plt.ylabel("ROO", fontsize=15)
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(self.__filename + "_oh_ft.png")


def command_cpmd_angleoh(args):
    """
    Calculates the angle between O-H bonds using command line arguments.

    This function takes command line arguments, initializes the ANGLEOH class,
    and calls the calc_angleoh method to calculate the angle between O-H bonds.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.

    Returns
    -------
    int
        0 if the calculation is successful.

    Examples
    --------
    >>> python angleoh.py --filename traj.xyz --molfile mol.mol --timestep 1.0 --numatom 3 --initial 1
    0
    """
    if not os.path.isfile(args.filename):
        raise FileNotFoundError(
            " ERROR :: " + str(args.filename) + " does not exist !!"
        )

    if not os.path.isfile(args.molfile):
        raise FileNotFoundError(" ERROR :: " + str(args.molfile) + " does not exist !!")

    angleoh = ANGLEOH(
        args.filename,
        args.molfile,
        float(args.timestep),
        int(args.numatom),
        int(args.initial),
    )
    df_acf, df_roo = angleoh.calc_angleoh()
    angleoh.save_files(df_acf, df_roo)
    angleoh.visualize_roo(df_roo)
    return 0
