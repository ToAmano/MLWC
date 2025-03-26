"""
This module calculates the distance, vector, and angle correlation functions
from CPMD trajectory data. It utilizes the ASE library to read trajectory files
and calculates the specified correlation function based on user-defined atom indices.
The results are saved as CSV files and visualized as plots.

Example:
    To use this module, you need to have a CPMD trajectory file (e.g., XYZ format),
    a molecule definition file (e.g., MOL format), and specify the atom indices
    for which the correlation function should be calculated.

    >>> python cpextract_cpmd.py --filename traj.xyz --molfile molecule.mol --index 1 2 --timestep 0.5 --numatom 3 --initial 1 --strategy distance

"""
import numpy as np
import pandas as pd
import ase
import ase.io
import ase.units
import scipy
import matplotlib.pyplot as plt
import bond.atomtype
import fourier.hydrogenbond
import __version__
from include.file_io import to_csv_with_comment
from include.mlwc_logger import setup_cmdline_logger
logger = setup_cmdline_logger(__name__)


class distance_vector_autocorrelation:
    """
    Calculates the distance, vector, or angle auto-correlation function of given atom indices in a molecular dynamics trajectory.

    This class reads a trajectory file, calculates the specified auto-correlation function based on the provided atom indices,
    and saves the results to CSV files. It supports distance, vector, and angle correlation calculations.

    Attributes:
        __filename (str): The path to the trajectory file (e.g., XYZ format).
        __initial_step (int): The initial step to start calculating the auto-correlation function.
        __molfile (str): The path to the molecule definition file (e.g., MOL format).
        _index (list[int]): A list of atom indices to calculate the auto-correlation function.
        _timestep (float): The time step between frames in the trajectory, in femtoseconds (fs).
        _NUM_ATOM_PER_MOL (int): The number of atoms per molecule in the system, including any dummy atoms.
        NUM_MOL (int): The number of molecules in the system.
        comment (str): A comment string that will be added to the output CSV files, containing information about the calculation parameters.

    Parameters
    ----------
    filename : str
        Path to the trajectory file (XYZ format).
    molfile : str
        Path to the molecule definition file (MOL format).
    index : list[int]
        List of atom indices to calculate the distance auto-correlation.
    timestep : float
        Timestep in femtoseconds (fs).
    NUM_ATOM_PER_MOL : int
        Number of atoms per molecule (including WCs & BCs).
    initial_step : int, optional
        Initial step to calculate MSD, by default 1

    Raises
    ------
    FileNotFoundError
        If the trajectory file or molecule definition file does not exist.
    ValueError
        If the initial step is less than 1, or if the molecule definition file does not end with '.itp' or '.mol'.
    AssertionError
        If the number of atoms in the first step is not divisible by the number of atoms per molecule.

    Examples
    --------
    >>> dac = distance_vector_autocorrelation("traj.xyz", "molecule.mol", [1, 2], 0.5, 3)
    >>> dac.calc_distanceft()
    ... # doctest: +SKIP
    """

    def __init__(self, filename: str, molfile: str, index: list[int], timestep: float, NUM_ATOM_PER_MOL: int, initial_step: int = 1):
        """
        Initializes the distance_vector_autocorrelation class.

        Parameters
        ----------
        filename : str
            Path to the trajectory file (XYZ format).
        molfile : str
            Path to the molecule definition file (MOL format).
        index : list[int]
            List of atom indices to calculate the distance auto-correlation.
        timestep : float
            Timestep in femtoseconds (fs).
        NUM_ATOM_PER_MOL : int
            Number of atoms per molecule (including WCs & BCs).
        initial_step : int, optional
            Initial step to calculate MSD, by default 1

        Raises
        ------
        FileNotFoundError
            If the trajectory file or molecule definition file does not exist.
        ValueError
            If the initial step is less than 1, or if the molecule definition file does not end with '.itp' or '.mol'.

        Examples
        --------
        >>> dac = distance_vector_autocorrelation("traj.xyz", "molecule.mol", [1, 2], 0.5, 3)
        """
        self.__filename = filename  # xyz
        self.__initial_step = initial_step  # initial step to calculate msd
        self.__molfile = molfile  # .mol
        self._index = index  # index of atoms to calculate distance auto-correlation
        # timestep in [fs]
        self._timestep = timestep  # timestep in [fs]
        self._NUM_ATOM_PER_MOL = NUM_ATOM_PER_MOL  # including WCs & BCs

        import os
        if not os.path.isfile(self.__filename):
            raise FileNotFoundError(
                " ERROR :: "+str(self.__filename)+" does not exist !!")

        if self.__initial_step < 1:
            raise ValueError("ERROR: initial_step must be larger than 1")

        if not os.path.isfile(self.__molfile):
            raise FileNotFoundError(
                " ERROR :: "+str(self.__molfile)+" does not exist !!")

        # * itpデータの読み込み
        # note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
        # 実際の読み込み
        if self.__molfile.endswith(".itp"):
            self.itp_data = bond.atomtype.read_itp(self.__molfile)
        elif self.__molfile.endswith(".mol"):
            self.itp_data = bond.atomtype.read_mol(self.__molfile)
        else:
            raise ValueError(
                "ERROR :: itp_filename should end with .itp or .mol")
        # bonds_list=itp_data.bonds_list
        # NUM_MOL_ATOMS=self.itp_data.num_atoms_per_mol

        # read xyz
        logger.info(" READING TRAJECTORY... This may take a while, be patient.")
        self._traj = ase.io.read(self.__filename, index=":")
        logger.info(
            f" FINISH READING TRAJECTORY... :: len(traj) = {len(self._traj)}")
        #
        # number of molecules
        self.NUM_MOL = len(self._traj[0])//self._NUM_ATOM_PER_MOL
        assert len(
            self._traj[0]) % self._NUM_ATOM_PER_MOL == 0, "ERROR: Number of atoms in the first step is not divisible by the number of atoms per molecule"
        logger.info(f" NUM_MOL == {self.NUM_MOL}")

        self.comment: str = f'''
        # File generated by CPextract.py cpmd distanceft version {__version__.__version__}.\n
        # Parameters: filename={self.__filename}, molfilename={self.__molfile}\n
        # Parameters: index={self._index}\n
        # Parameters: initial_step={self.__initial_step}, timestep={self._timestep}\n
        # Data below:\n
        '''

    def calc_distanceft(self) -> np.ndarray:
        """
        Calculates the distance auto-correlation function.

        This method calculates the distance auto-correlation function between two atoms specified by their indices.
        It uses the atomic trajectory data to compute the distances between the atoms over time and then calculates the
        auto-correlation function of these distances.

        Returns
        -------
        np.ndarray
            The mean distance auto-correlation function as a NumPy array.

        Raises
        ------
        ValueError
            If the length of the index is not equal to 2.

        Examples
        --------
        >>> dac = distance_vector_autocorrelation("traj.xyz", "molecule.mol", [1, 2], 0.5, 3)
        >>> mean_correlation = dac.calc_distanceft()
        >>> print(mean_correlation)
        ... # doctest: +SKIP
        """
        if len(self._index) != 2:
            raise ValueError(
                "ERROR :: index should have two elements for distance calculation")
        # OHボンドのO原子のリスト
        atom1_list: list[int] = [self._NUM_ATOM_PER_MOL*mol_id +
                                 self._index[0] for mol_id in range(self.NUM_MOL)]
        atom2_list: list[int] = [self._NUM_ATOM_PER_MOL*mol_id +
                                 self._index[1] for mol_id in range(self.NUM_MOL)]
        # calculate atomic distances
        hydrogen_bond_list: np.ndarray = fourier.hydrogenbond.calc_distance(
            self._traj, atom1_list, atom2_list)
        np.save(self.__filename +
                f"_atomindex_{self._index[0]}_{self._index[1]}_list.npy", hydrogen_bond_list)
        # 全ての時系列に対して自己相関を計算 (axis=1で各行に対して自己相関を計算)
        # 'same' モードで時系列の長さを維持
        # !! numpy correlate does not support FFT
        correlations = np.apply_along_axis(lambda x: scipy.signal.correlate(
            x, x, mode='full'), axis=0, arr=hydrogen_bond_list)
        # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
        mean_correlation = np.mean(correlations, axis=1)[
            len(hydrogen_bond_list)-1:]  # acf
        return mean_correlation

    def calc_vectorft(self) -> np.ndarray:
        """
        Calculates the vector auto-correlation function.

        This method calculates the vector auto-correlation function between two atoms specified by their indices.
        It uses the atomic trajectory data to compute the vectors between the atoms over time and then calculates the
        auto-correlation function of these vectors.

        Returns
        -------
        np.ndarray
            The mean vector auto-correlation function as a NumPy array.

        Raises
        ------
        ValueError
            If the length of the index is not equal to 2.

        Examples
        --------
        >>> dac = distance_vector_autocorrelation("traj.xyz", "molecule.mol", [1, 2], 0.5, 3)
        >>> mean_correlation = dac.calc_vectorft()
        >>> print(mean_correlation)
        ... # doctest: +SKIP
        """
        import fourier.hydrogenbond
        if len(self._index) != 2:
            raise ValueError(
                "ERROR :: index should have two elements for vector calculation")
        # OHボンドのO原子のリスト
        atom1_list: list[int] = [self._NUM_ATOM_PER_MOL*mol_id +
                                 self._index[0] for mol_id in range(self.NUM_MOL)]
        atom2_list: list[int] = [self._NUM_ATOM_PER_MOL*mol_id +
                                 self._index[1] for mol_id in range(self.NUM_MOL)]
        # calculate OH vector
        bond_vectors: np.array = fourier.hydrogenbond.calc_oh(
            self._traj, atom1_list, atom2_list)
        np.save(self.__filename +
                f"_angle_{self._index[0]}_{self._index[1]}_list.npy", bond_vectors)
        # 全ての時系列に対して自己相関を計算 (axis=1で各行に対して自己相関を計算)
        # 'same' モードで時系列の長さを維持
        # !! numpy correlate does not support FFT
        correlations = np.apply_along_axis(lambda x: scipy.signal.correlate(
            x, x, mode='full'), axis=0, arr=bond_vectors)
        correlations = np.sum(correlations, axis=2)  # inner dot

        # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
        mean_correlation = np.mean(correlations, axis=1)[
            len(bond_vectors)-1:]  # acf
        return mean_correlation

    def calc_angleft(self) -> np.ndarray:
        """
        Calculates the angle auto-correlation function.

        This method calculates the angle auto-correlation function between two vectors defined by four atoms specified by their indices.
        It uses the atomic trajectory data to compute the angles between the vectors over time and then calculates the
        auto-correlation function of these angles.

        Returns
        -------
        np.ndarray
            The mean angle auto-correlation function as a NumPy array.

        Raises
        ------
        ValueError
            If the length of the index is not equal to 4.

        Examples
        --------
        >>> dac = distance_vector_autocorrelation("traj.xyz", "molecule.mol", [1, 2, 3, 4], 0.5, 3)
        >>> mean_correlation = dac.calc_angleft()
        >>> print(mean_correlation)
        ... # doctest: +SKIP
        """
        import fourier.hydrogenbond
        if len(self._index) != 4:
            raise ValueError(
                "ERROR :: index should have four elements for angle calculation")
        # OHボンドのO原子のリスト
        start_1_list: list[int] = [self._NUM_ATOM_PER_MOL *
                                   mol_id+self._index[0] for mol_id in range(self.NUM_MOL)]
        end_1_list: list[int] = [self._NUM_ATOM_PER_MOL*mol_id +
                                 self._index[1] for mol_id in range(self.NUM_MOL)]
        start_2_list: list[int] = [self._NUM_ATOM_PER_MOL *
                                   mol_id+self._index[2] for mol_id in range(self.NUM_MOL)]
        end_2_list: list[int] = [self._NUM_ATOM_PER_MOL*mol_id +
                                 self._index[3] for mol_id in range(self.NUM_MOL)]

        # calculate two (normalized) vectors
        bond_vectors_1: np.array = fourier.hydrogenbond.calc_oh(
            self._traj, start_1_list, end_1_list)
        bond_vectors_2: np.array = fourier.hydrogenbond.calc_oh(
            self._traj, start_2_list, end_2_list)
        bond_vector_angle: np.array = np.einsum(
            "ijk,ijk -> ij", bond_vectors_1, bond_vectors_2)

        np.save(self.__filename +
                f"_angle_{self._index[0]}_{self._index[1]}_list.npy", bond_vector_angle)
        # 全ての時系列に対して自己相関を計算 (axis=1で各行に対して自己相関を計算)
        # 'same' モードで時系列の長さを維持
        # !! numpy correlate does not support FFT
        correlations = np.apply_along_axis(lambda x: scipy.signal.correlate(
            x, x, mode='full'), axis=0, arr=bond_vector_angle)

        # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
        mean_correlation = np.mean(correlations, axis=1)[
            len(bond_vector_angle)-1:]  # acf
        return mean_correlation

    def save_files(self, mean_correlation: np.ndarray, strategy: str = "distance"):
        """
        Saves the auto-correlation function and its Fourier transform to CSV files and generates a plot.

        This method saves the calculated auto-correlation function and its Fourier transform to CSV files.
        It also generates a plot of the Fourier transform and saves it as a PNG file.

        Parameters
        ----------
        mean_correlation : np.ndarray
            The mean auto-correlation function as a NumPy array.
        strategy : str, optional
            The strategy used to calculate the auto-correlation function (distance, vector, or angle), by default "distance"

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the auto-correlation function and its Fourier transform as Pandas DataFrames.

        Examples
        --------
        >>> dac = distance_vector_autocorrelation("traj.xyz", "molecule.mol", [1, 2], 0.5, 3)
        >>> mean_correlation = dac.calc_distanceft()
        >>> df_acf, df_roo = dac.save_files(mean_correlation, "distance")
        ... # doctest: +SKIP
        """
        import fourier.hydrogenbond
        df_acf: pd.DataFrame = fourier.hydrogenbond.make_df_acf(
            mean_correlation, self._timestep)

        to_csv_with_comment(df_acf, self.comment, self.__filename +
                            f"_{strategy}_{self._index[0]}_{self._index[1]}_acf.csv")
        # df_acf.to_csv(self.__filename+"_roo_acf.csv",index=False)

        # Fourier Transform
        df_roo: pd.DataFrame = fourier.hydrogenbond.calc_lengthcorr(
            mean_correlation, self._timestep)
        to_csv_with_comment(df_roo, self.comment, self.__filename +
                            f"_{strategy}_{self._index[0]}_{self._index[1]}_ft.csv")
        # df_roo.to_csv(self.__filename+"_roo_ft.csv",index=False)

        # visualize data
        plt.plot(df_roo["freq_kayser"], df_roo["roo"], label="data")
        # plt.ylim(1.4e3,2e3)
        # plt.xlim(0.1, 10000)
        plt.legend(fontsize=15)
        plt.xlabel("cm-1", fontsize=15)
        plt.ylabel("ROO", fontsize=15)
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(
            f"{self.__filename}_{strategy}_{self._index[0]}_{self._index[1]}_ft.png")

        return df_acf, df_roo


def command_cpmd_ft(args):  # distance, vector, angle correlation of two atoms
    """
    Calculates and saves the distance, vector, or angle auto-correlation function based on command-line arguments.

    This function takes command-line arguments, initializes the distance_vector_autocorrelation class,
    calculates the specified auto-correlation function, and saves the results to files.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing the following attributes:
            filename (str): Path to the trajectory file.
            molfile (str): Path to the molecule definition file.
            index (list[int]): List of atom indices.
            timestep (float): Timestep in femtoseconds (fs).
            numatom (int): Number of atoms per molecule.
            initial (int): Initial step to calculate MSD.
            strategy (str): Strategy to calculate the auto-correlation function (distance, vector, or angle).

    Returns
    -------
    int
        0 if the calculation and saving are successful.

    Raises
    ------
    ValueError
        If the strategy is not one of 'distance', 'vector', or 'angle'.

    Examples
    --------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--filename", type=str, default="traj.xyz")
    >>> parser.add_argument("--molfile", type=str, default="molecule.mol")
    >>> parser.add_argument("--index", type=int, nargs="+", default=[1, 2])
    >>> parser.add_argument("--timestep", type=float, default=0.5)
    >>> parser.add_argument("--numatom", type=int, default=3)
    >>> parser.add_argument("--initial", type=int, default=1)
    >>> parser.add_argument("--strategy", type=str, default="distance")
    >>> args = parser.parse_args()
    >>> command_cpmd_ft(args)
    0
    """
    roo = distance_vector_autocorrelation(args.filename, args.molfile, args.index, float(
        args.timestep), int(args.numatom), int(args.initial))
    if args.strategy == "distance":
        mean_correlation = roo.calc_distanceft()
    elif args.strategy == "vector":
        mean_correlation = roo.calc_vectorft()
    elif args.strategy == "angle":
        mean_correlation = roo.calc_angleft()
    else:
        raise ValueError(
            "ERROR :: strategy should be either distance, vector or angle")
    roo.save_files(mean_correlation, args.strategy)
    return 0
