"""
This module provides functions to calculate the assignment of Wannier centers (WCs)
to bond centers and lone pairs using PyTorch. It includes functionalities for
extracting WCs from atomic structures, calculating bond center coordinates,
finding the nearest WCs to bond centers, and computing bond dipole moments.
The module is designed to work with ASE Atoms objects and relies on other modules
within the cpmd package for PBC calculations and distance computations.
"""

# https://qiita.com/junkmd/items/479a8bafa03c8e0428ac
from include.constants import constant
from typing_extensions import deprecated
import ase
import sys
import numpy as np
import torch
# from types import NoneType
from cpmd.pbc.pbc import pbc
from cpmd.pbc.pbc_mol import pbc_mol
from cpmd.bondcenter.bondcenter import calc_bondcenter
import torch
from cpmd.distance.distance import distance_matrix, distance_2d
from include.mlwc_logger import setup_cmdline_logger, get_log_level
logger = setup_cmdline_logger("MLWC."+__name__, level=get_log_level())

# 物理定数
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef = constant.Ang*constant.Charge/constant.Debye

# !! >>>>>>>>>>>>>>  新しい実装 >>>>>>>>>>>>>>
# !! 計算したいのは，各ボンドセンター，ローンペアに対してのWCの割り当て
# !! アルゴリズムとしては，まずボンドに対して最近接を全て攫う．
# !! 次に，残りの中からlpを探索する．

# !! まず，ある座標のリストとwfcリストの最も近いWCを計算する関数を作成する．


def extract_wcs(atoms: ase.Atoms):
    """Extract atomic coordinates and Wannier center coordinates from ASE Atoms object.

    This function separates the atomic coordinates and Wannier center coordinates
    from a given ASE Atoms object. It identifies the Wannier centers by the
    chemical symbol "X".

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing both atomic and Wannier center coordinates.

    Returns
    -------
    atoms_nowan : ase.Atoms
        An ASE Atoms object containing only the atomic coordinates (excluding Wannier centers).
    wfc_list : numpy.ndarray
        A numpy array containing the coordinates of the Wannier centers.

    Examples
    --------
    >>> from ase import Atoms
    >>> atoms = Atoms('H2X', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    >>> atoms_nowan, wfc_list = extract_wcs(atoms)
    >>> print(atoms_nowan)
    Atoms(symbols='H2', pbc=False)
    >>> print(wfc_list)
    [[0. 0. 2.]]
    """

    # ワニエの座標を廃棄する．
    # for debug
    # 配列の原子種&座標を取得
    atom_list = np.array(atoms.get_chemical_symbols())
    coord_list = atoms.get_positions()
    masked_X_list = (atom_list == "X")
    masked_notX_list = ~masked_X_list
    # coods
    atom_nowan_list = coord_list[masked_notX_list]
    wfc_list = coord_list[masked_X_list]
    # symbols
    symbols_nowan_list = atom_list[masked_notX_list]

    UNITCELL_VECTORS = atoms.get_cell()
    atoms_nowan = ase.Atoms(
        symbols_nowan_list, positions=atom_nowan_list, cell=UNITCELL_VECTORS, pbc=[1, 1, 1])
    return atoms_nowan, wfc_list


class atoms_wan():
    """A class to store atomic and Wannier center information.

    This class is a wrapper to store the atomic coordinates (atoms_nowan) and
    Wannier center coordinates (wfc_list) together.

    Attributes
    ----------
    atoms : ase.Atoms
        The original ASE Atoms object.
    atoms_nowan : ase.Atoms
        An ASE Atoms object containing only the atomic coordinates (excluding Wannier centers).
    wfc_list : numpy.ndarray
        A numpy array containing the coordinates of the Wannier centers.

    Examples
    --------
    >>> from ase import Atoms
    >>> atoms = Atoms('H2X', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    >>> atoms_nowan, wfc_list = extract_wcs(atoms)
    >>> atoms_wan_instance = atoms_wan(atoms)
    >>> atoms_wan_instance.set_params(atoms_nowan, wfc_list)
    >>> print(atoms_wan_instance.atoms_nowan)
    Atoms(symbols='H2', pbc=False)
    >>> print(atoms_wan_instance.wfc_list)
    [[0. 0. 2.]]
    """

    def set_params(self, atoms_nowan: ase.Atoms, NUM_MOL: int, dict_mu: dict, dict_bcs: dict):
        """Set the atomic and Wannier center coordinates.

        Parameters
        ----------
        atoms_nowan : ase.Atoms
            An ASE Atoms object containing only the atomic coordinates (excluding Wannier centers).
        wfc_list : numpy.ndarray
            A numpy array containing the coordinates of the Wannier centers.

        Returns
        -------
        None

        Examples
        --------
        >>> from ase import Atoms
        >>> atoms = Atoms('H2X', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        >>> atoms_nowan, wfc_list = extract_wcs(atoms)
        >>> atoms_wan_instance = atoms_wan(atoms)
        >>> atoms_wan_instance.set_params(atoms_nowan, wfc_list)
        >>> print(atoms_wan_instance.atoms_nowan)
        Atoms(symbols='H2', pbc=False)
        >>> print(atoms_wan_instance.wfc_list)
        [[0. 0. 2.]]
        """
        self.atoms_nowan = atoms_nowan
        self.dict_mu = dict_mu
        self.dict_bcs = dict_bcs
        # lpsを計算
        atomic_positions = atoms_nowan.get_positions()
        # Olp
        Olp_coords = atomic_positions[(
            np.array(atoms_nowan.get_chemical_symbols()) == "O")].reshape(NUM_MOL, -1, 3)
        # Nlp
        Nlp_coords = atomic_positions[(
            np.array(atoms_nowan.get_chemical_symbols()) == "N")].reshape(NUM_MOL, -1, 3)
        self.dict_bcs["Olp"] = Olp_coords
        self.dict_bcs["Nlp"] = Nlp_coords

    def make_atoms_with_wc(self) -> ase.Atoms:
        # def make_ase_with_WCs(ase_atomicnumber,NUM_MOL, UNITCELL_VECTORS,list_mol_coords,list_bond_centers,list_bond_wfcs,list_dbond_wfcs,list_lpO_wfcs,list_lpN_wfcs):
        '''
        元の分子座標に加えて，WCsとボンドセンターを加えたase.atomsを作成する．

        2023/6/2：今までは原子/BC,WC/ローンペアの順だったが，わかりやすさの改善のため，
        分子ごとに原子/ボンドセンター/ローンペアの順にappendすることにした．
        '''
        # list_mol_coords,list_bond_centers =results
        # list_bond_wfcs,list_dbond_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_wfcs

        new_coord = []
        new_atomic_num = []

        ase_atomicnumber = self.atoms_nowan.get_atomic_numbers()
        list_atomic_nums = list(
            np.array(ase_atomicnumber).reshape(self.NUM_MOL, -1))
        # loop over molecule
        for mol_r, mol_at, mol_wc, mol_bc, mol_lpO, mol_lpN in zip(self.list_mol_coords, list_atomic_nums, self.list_bond_wfcs, self.list_bond_centers, self.list_lpO_wfcs, self.list_lpN_wfcs):
            for r, at in zip(mol_r, mol_at):  # 原子
                new_atomic_num.append(at)  # 原子番号
                new_coord.append(r)  # 原子座標

            for bond_wc, bond_bc in zip(mol_wc, mol_bc):  # ボンドセンターとボンドWCs
                new_coord.append(bond_bc)
                new_atomic_num.append(2)  # ボンド？（原子番号2：Heを割り当て）
                for wc in bond_wc:
                    new_coord.append(wc)
                    # ボンドワニエセンター（原子番号10：Neを割り当て）（電荷-2e）
                    new_atomic_num.append(10)

            # for dbond_wc in mol_Dwc : # double bond
            #     for wc in dbond_wc :
            #         new_coord.append(wc)
            #         new_atomic_num.append(10) # ワニエセンター（原子番号10：Neを割り当て）

            for lp_wc in mol_lpO:  # Oのローンペア（電荷-4e）
                for wc in lp_wc:
                    new_coord.append(wc)
                    new_atomic_num.append(10)

            for lp_wc in mol_lpN:  # Nのローンペア
                for wc in lp_wc:
                    new_coord.append(wc)
                    new_atomic_num.append(10)

        # change to numpy
        new_coord = np.array(new_coord)

        # WFCsと原子を合体させたAtomsオブジェクトを作成する．
        aseatoms_with_WC = ase.Atoms(new_atomic_num,
                                     positions=new_coord,
                                     cell=self.UNITCELL_VECTORS,
                                     pbc=[1, 1, 1])
        self.atoms_wc = aseatoms_with_WC  # pbcが考慮されたase.Atomsオブジェクト
        return aseatoms_with_WC


def calculate_molcoord(atoms: ase.Atoms, bonds_list, ref_atom_index, NUM_ATOM_PER_MOL: int | None = None):
    """Calculate the coordinates of atoms in a molecule with periodic boundary conditions (PBC).

    This function computes the coordinates of atoms within a molecule, taking into
    account periodic boundary conditions. It uses the `pbc_mol` module to apply PBC
    to the atomic positions.

    Parameters
    ----------
    atoms : ase.Atoms
        The ASE Atoms object containing the atomic coordinates and cell vectors.
    bonds_list : list[list]
        A list of bonds, where each bond is represented by a list of two atom indices.
    ref_atom_index : int
        The index of the reference atom in the molecule.
    NUM_ATOM_PER_MOL : int, optional
        The number of atoms per molecule. If None, it is inferred from the bonds_list.

    Returns
    -------
    atomic_positions : numpy.ndarray
        A numpy array of shape (NUM_MOL, NUM_ATOM_PER_MOL, 3) containing the
        coordinates of the atoms in each molecule, after applying PBC.

    Examples
    --------
    >>> from ase import Atoms
    >>> atoms = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[10, 10, 10])
    >>> bonds_list = [[0, 1], [0, 2]]
    >>> ref_atom_index = 0
    >>> atomic_positions = calculate_molcoord(atoms, bonds_list, ref_atom_index)
    >>> print(atomic_positions)
    [[[0. 0. 0.]
      [0. 0. 1.]
      [0. 0. 2.]]]
    """
    if NUM_ATOM_PER_MOL == None:
        NUM_ATOM_PER_MOL = np.unique(np.array(bonds_list)).size

    # apply pbc to drs
    atomic_positions: np.ndarray = pbc(pbc_mol).compute_pbc(
        vectors_array=atoms.get_positions(),
        cell=atoms.get_cell(),
        bonds_list=bonds_list,
        NUM_ATOM_PAR_MOL=NUM_ATOM_PER_MOL,
        ref_atom_index=ref_atom_index)  # [bondcent,Atom,3]
    return atomic_positions


def find_nearest_wfc(bondcenters: np.ndarray, wfc_list: np.ndarray, UNITCELL_VECTORS) -> np.ndarray:
    """Find the indices of the nearest Wannier centers (WCs) to a set of bond centers.

    This function computes the distances between each bond center and all Wannier centers
    and returns the indices of the Wannier centers that are closest to each bond center.

    Parameters
    ----------
    bondcenters : numpy.ndarray
        A numpy array of shape (N, 3) containing the coordinates of the bond centers.
    wfc_list : numpy.ndarray
        A numpy array of shape (M, 3) containing the coordinates of the Wannier centers.
    UNITCELL_VECTORS : numpy.ndarray
        The unit cell vectors of the simulation cell.

    Returns
    -------
    nearest_indices : numpy.ndarray
        A numpy array of shape (M, N) containing the indices of the nearest Wannier centers
        to each bond center, sorted by distance.

    Examples
    --------
    >>> import numpy as np
    >>> bondcenters = np.array([[0, 0, 0], [1, 1, 1]])
    >>> wfc_list = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> UNITCELL_VECTORS = np.eye(3) * 10
    >>> nearest_indices = find_nearest_wfc(bondcenters, wfc_list, UNITCELL_VECTORS)
    >>> print(nearest_indices)
    [[0 2]
     [2 0]
     [1 1]]
    """
    # 全てのボンドセンターに対してWCを割り当てる
    # bondcenters と wfc_list の各ペア間の距離を計算する
    # `bondcenters` の形状は (N, 3)、`wfc_list` の形状は (M, 3) であると仮定
    distances = distance_matrix.compute_distances(
        bondcenters.reshape(-1, 3), wfc_list, UNITCELL_VECTORS, pbc=True, norm=True)
    logger.debug(f"distances.shape = {np.shape(distances)}")
    # 各 `bondcenters` に対する最も近い2つの `wfc_list` のインデックスを取得(N*M)
    nearest_indices = np.argsort(distances, axis=1)
    return nearest_indices


def calculate_nearest_number_list(num_mols, itp_data):
    # repeat for num_mols
    nearest_number_list = np.repeat(itp_data.bonds_type, num_mols)
    logger.debug(f"nearest_number_list = {nearest_number_list}")


def check_duplicate_indices(*index_lists):
    """Check for duplicate indices across multiple lists.

    Parameters
    ----------
    *index_lists : list[numpy.ndarray] or list[list[int]]
        Multiple lists or numpy arrays containing indices.

    Raises
    ------
    ValueError
        If duplicate indices are found, the function raises an error with detailed information.

    Returns
    -------
    None
        If no duplicates are found, the function returns normally.
    """
    # Flatten all input lists into a single 1D array
    # 各リストをフラットにする
    flattened_indices = []
    for index_list in index_lists:
        flattened_indices.extend(index_list)  # リストに追加
    # flattened_indices = np.concatenate(index_lists)
    flattened_indices = np.array([i for s in flattened_indices for i in s])
    logger.debug(f"Flattened indices = {flattened_indices}")

    # Get unique indices and their counts
    unique_indices, counts = np.unique(flattened_indices, return_counts=True)
    # Elements appearing more than once
    duplicates = unique_indices[counts > 1]

    # If duplicates exist, raise an error with details
    if duplicates.size > 0:
        duplicate_info = ", ".join(
            f"{val} (×{cnt})" for val, cnt in zip(duplicates, counts[counts > 1]))
        raise ValueError(f"Error: 重複するインデックスが含まれています: {duplicate_info}")


def calculate_nearest_wfc(nearest_indices, nearest_number_list: np.ndarray | None = None, num_wcs: int | None = None):
    """Calculate the nearest WFC indices for each bond type and check for duplicates.

    This function takes the indices of the nearest Wannier centers (WCs) to each bond center
    and selects a specific number of WCs for each bond based on the bond type information
    provided in the itp_data. It also checks for duplicate indices to ensure that each
    Wannier center is assigned to only one bond center.

    Parameters
    ----------
    nearest_indices : numpy.ndarray
        A numpy array of shape (M, N) containing the indices of the nearest Wannier centers
        to each bond center, sorted by distance.
    itp_data : object
        An object containing information about the bond types and the number of WCs to
        assign to each bond.

    Returns
    -------
    indices : list[numpy.ndarray]
        A list of numpy arrays, where each array contains the indices of the nearest WCs
        assigned to a specific bond center.

    Raises
    ------
    ValueError
        If there are duplicate indices in the assigned WCs, indicating that a Wannier center
        is assigned to multiple bond centers.

    Examples
    --------
    >>> import numpy as np
    >>> class MockItpData:
    ...     def __init__(self):
    ...         self.num_bonds = 2
    ...         self.bonds_type = [1, 1]
    >>> itp_data = MockItpData()
    >>> nearest_indices = np.array([[0, 1], [1, 0]])
    >>> indices = calculate_nearest_wfc(nearest_indices, itp_data)
    >>> print(indices)
    [array([0]), array([1])]
    """
    """bond種ごとに，うまく割り当てられているかの確認"""
    # `nearest_number_list` と `num_wcs` のどちらも `None` の場合はエラー
    if nearest_number_list is None and num_wcs is None:
        raise ValueError(
            "Error: `nearest_number_list` か `num_wcs` のいずれかを指定してください。")

    # 各 bondcenter に対して `nearest_number_list` に指定された個数を取得
    if nearest_number_list is not None:
        if len(nearest_number_list) != len(nearest_number_list):
            raise ValueError(
                "len(nearest_number_list) should be the same as len(nearest_number_list)")
        indices = [
            nearest_indices[i, :nearest_number_list[i]].tolist()
            for i in range(nearest_indices.shape[0])
        ]
    if num_wcs is not None:
        indices = nearest_indices[:, :num_wcs]
    # indicesの中に重複があればエラーを出す．
    check_duplicate_indices(indices)
    return indices


def calculate_comwfs(indices, wfc_list: np.ndarray):
    """Calculate the center of mass (COM) of Wannier functions (WFs) for each bond.

    This function computes the center of mass of the Wannier functions associated with
    each bond, based on the provided indices and the coordinates of the Wannier functions.

    Parameters
    ----------
    indices : list[numpy.ndarray]
        A list of numpy arrays, where each array contains the indices of the WFs
        assigned to a specific bond center.
    wfc_list : numpy.ndarray
        A numpy array of shape (M, 3) containing the coordinates of the Wannier centers.

    Returns
    -------
    nearest_comwfc : numpy.ndarray
        A numpy array of shape (N, 3) containing the coordinates of the COM of the WFs
        for each bond.

    Raises
    ------
    ValueError
        If the input `wfc_list` is not a 2D array.

    Examples
    --------
    >>> import numpy as np
    >>> indices = [[0, 1], [2, 3]]
    >>> wfc_list = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    >>> nearest_comwfc = calculate_comwfs(indices, wfc_list)
    >>> print(nearest_comwfc)
    [[0.  0.  0.5]
     [0.  1.  0.5]]
    """
    # 各 bondcenter に対して `nearest_number_list` に指定された個数を取得
    if len(np.shape(wfc_list)) != 2:
        raise ValueError(f"wfc_list should be 2D array")
    nearest_comwfc = [
        np.mean(wfc_list[indices[i]], axis=0)
        for i in range(len(indices))
    ]
    return np.array(nearest_comwfc)


def calculate_bondwfs(bondcenters, O_lonepairs, N_lonepairs, wfc_list, UNITCELL_VECTORS, bonds_type: list[int]):
    """Calculate bond dipole moments based on Wannier function centers.

    This function calculates the bond dipole moments by assigning Wannier function
    centers (WFCs) to bond centers and computing the center of mass of the assigned WFCs.
    It uses the itp_data to determine the number of WFCs to assign to each bond.

    Parameters
    ----------
    bondcenters : numpy.ndarray
        A numpy array of shape (N, 3) containing the coordinates of the bond centers.
    wfc_list : numpy.ndarray
        A numpy array of shape (M, 3) containing the coordinates of the Wannier centers.
    UNITCELL_VECTORS : numpy.ndarray
        The unit cell vectors of the simulation cell.
    itp_data : object
        An object containing information about the bond types and the number of WFCs to
        assign to each bond.

    Returns
    -------
    list_bond_mu : numpy.ndarray
        A numpy array of shape (N, 3) containing the bond dipole moments for each bond.

    Examples
    --------
    >>> import numpy as np
    >>> class MockItpData:
    ...     def __init__(self):
    ...         self.num_bonds = 2
    ...         self.bonds_type = [1, 1]
    >>> itp_data = MockItpData()
    >>> bondcenters = np.array([[0, 0, 0], [1, 1, 1]])
    >>> wfc_list = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> UNITCELL_VECTORS = np.eye(3) * 10
    >>> list_bond_mu = calculate_bondwfs(bondcenters, wfc_list, UNITCELL_VECTORS, itp_data)
    >>> print(list_bond_mu)
    [[ 0.          0.          0.16021766]
     [ 0.16021766  0.16021766  0.        ]]
    """
    """indicesからボンドごとの重心wfc
    atoms, wfs, bondtypeの3つがあれば良い．
    bondtypeから複数取り出すリストが出る．
    最終的にlist_bond_muを作成できれば良い．
    """
    if bondcenters.ndim != 3 or bondcenters.shape[2] != 3:
        raise ValueError(
            f"Invalid shape for bondcenters. Expected shape [a, b, 3], but got {bondcenters.shape}.")
    if O_lonepairs.ndim != 3 or O_lonepairs.shape[2] != 3:
        raise ValueError(
            f"Invalid shape for O_lonepairs. Expected shape [a, b, 3], but got {O_lonepairs.shape}.")
    if N_lonepairs.ndim != 3 or N_lonepairs.shape[2] != 3:
        raise ValueError(
            f"Invalid shape for N_lonepairs. Expected shape [a, b, 3], but got {N_lonepairs.shape}.")

    num_mols: int = bondcenters.shape[0]
    logger.debug(f" num_mols = {num_mols}")

    # BC
    nearest_indices = find_nearest_wfc(bondcenters, wfc_list, UNITCELL_VECTORS)
    indices = calculate_nearest_wfc(
        nearest_indices, np.repeat(bonds_type, num_mols))
    nearest_comwfc: np.ndarray = calculate_comwfs(indices, wfc_list)
    logger.debug("BC assign success")
    # BC：bond_mu
    diff_bc_pbc = distance_2d.compute_distances(
        bondcenters.reshape(-1, 3), nearest_comwfc, UNITCELL_VECTORS, pbc=True, norm=False).reshape(bondcenters.shape)
    list_bond_mu = (-2.0)*coef*np.einsum("j,ijk->ijk",
                                         bonds_type, diff_bc_pbc)

    # Olp
    if O_lonepairs.size > 0:
        lonepair_nearest_indices = find_nearest_wfc(
            O_lonepairs, wfc_list, UNITCELL_VECTORS)
        lonepair_indices = calculate_nearest_wfc(
            lonepair_nearest_indices, num_wcs=2)
        nearest_lonepair_comwfc = calculate_comwfs(lonepair_indices, wfc_list)
        logger.debug("Olp assign success")
        # Lp: bond_mu
        diff_Olp_pbc = distance_2d.compute_distances(
            O_lonepairs.reshape(-1, 3), nearest_lonepair_comwfc, UNITCELL_VECTORS, pbc=True, norm=False).reshape(O_lonepairs.shape)
        list_Olp_mu = -4.0*coef * diff_Olp_pbc
    else:
        list_Olp_mu = np.empty_like(O_lonepairs)

    # Nlp
    if N_lonepairs.size > 0:
        Nlonepair_nearest_indices = find_nearest_wfc(
            N_lonepairs, wfc_list, UNITCELL_VECTORS)
        Nlonepair_indices = calculate_nearest_wfc(
            Nlonepair_nearest_indices, num_wcs=2)
        Nnearest_lonepair_comwfc = calculate_comwfs(
            Nlonepair_indices, wfc_list)
        logger.debug("Nlp assign success")
        # Nlp: mu
        diff_Nlp_pbc = distance_2d.compute_distances(
            N_lonepairs.reshape(-1, 3), Nnearest_lonepair_comwfc, UNITCELL_VECTORS, pbc=True, norm=False).reshape(N_lonepairs.shape)
        list_Nlp_mu = -6.0*coef * diff_Nlp_pbc
    else:
        list_Nlp_mu = np.empty_like(N_lonepairs)

    return list_bond_mu, list_Olp_mu, list_Nlp_mu


def convert_atoms_to_bondwfc(atoms_nowan: ase.Atoms, wfc_list, bonds_type, bonds_list, bond_index, ref_atom_index):
    """分割したatoms_nowan, wfc_list，bond情報から，すべてのbondmuを計算する．"""
    molcoord = calculate_molcoord(
        atoms_nowan, bonds_list, ref_atom_index)
    # calculate all the BCs
    bcs_coords = calc_bondcenter(molcoord, bonds_list)

    UNITCELL_VECTORS = atoms_nowan.get_cell()
    atomic_positions = atoms_nowan.get_positions()
    num_atoms_per_mol = len(np.unique(np.array(bonds_list)))
    NUM_MOL = int(len(atomic_positions)/num_atoms_per_mol)
    logger.debug(f"num_bonds = {num_atoms_per_mol} :: NUM_MOL = {NUM_MOL}")
    # Olp
    Olp_coords = atomic_positions[(
        np.array(atoms_nowan.get_chemical_symbols()) == "O")].reshape(NUM_MOL, -1, 3)
    # Nlp
    Nlp_coords = atomic_positions[(
        np.array(atoms_nowan.get_chemical_symbols()) == "N")].reshape(NUM_MOL, -1, 3)
    list_bond_mu, list_Olp_mu, list_Nlp_mu = calculate_bondwfs(
        bcs_coords, Olp_coords, Nlp_coords, wfc_list, UNITCELL_VECTORS, bonds_type)

    dict_bond_mu = {
        key: list_bond_mu[:, np.array(item), :] if item else np.array([])
        for key, item in bond_index.items()
    }
    dict_bond_mu["Olp"] = list_Olp_mu
    dict_bond_mu["Nlp"] = list_Nlp_mu
    return dict_bond_mu
