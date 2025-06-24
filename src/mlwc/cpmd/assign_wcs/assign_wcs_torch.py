"""
This module provides functions to calculate the assignment of Wannier centers (WCs)
to bond centers and lone pairs using PyTorch. It includes functionalities for
extracting WCs from atomic structures, calculating bond center coordinates,
finding the nearest WCs to bond centers, and computing bond dipole moments.
The module is designed to work with ASE Atoms objects and relies on other modules
within the cpmd package for PBC calculations and distance computations.
"""

from dataclasses import dataclass
from typing import List, Tuple

import ase
import numpy as np

from mlwc.cpmd.bondcenter.bondcenter import calc_bondcenter, calc_bondcenter_dict
from mlwc.cpmd.distance.distance import distance_2d, distance_ase, distance_matrix

# from types import NoneType
from mlwc.cpmd.pbc.pbc import pbc
from mlwc.cpmd.pbc.pbc_mol import pbc_mol

# https://qiita.com/junkmd/items/479a8bafa03c8e0428ac
from mlwc.include.constants import Constant
from mlwc.include.mlwc_logger import get_log_level, setup_cmdline_logger

logger = setup_cmdline_logger("MLWC." + __name__, level=get_log_level())

# 物理定数
# Debye   = 3.33564e-30
# charge  = 1.602176634e-019
# ang      = 1.0e-10
coef = Constant.Ang * Constant.Charge / Constant.Debye

# !! >>>>>>>>>>>>>>  新しい実装 >>>>>>>>>>>>>>
# !! 計算したいのは，各ボンドセンター，ローンペアに対してのWCの割り当て
# !! アルゴリズムとしては，まずボンドに対して最近接を全て攫う．
# !! 次に，残りの中からlpを探索する．

# !! まず，ある座標のリストとwfcリストの最も近いWCを計算する関数を作成する．


def extract_wcs(atoms: ase.Atoms) -> Tuple[ase.Atoms, np.array]:
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
    atom_list = np.array(atoms.get_chemical_symbols())
    masked_X_list = atom_list == "X"
    masked_notX_list = ~masked_X_list
    # coods
    coord_list: np.array = atoms.get_positions()
    atom_nowan_list: np.array = coord_list[masked_notX_list]
    wfc_list: np.array = coord_list[masked_X_list]
    atoms_nowan = ase.Atoms(
        atom_list[masked_notX_list],
        positions=atom_nowan_list,
        cell=atoms.get_cell(),
        pbc=[1, 1, 1],
    )
    return atoms_nowan, wfc_list


@dataclass(frozen=True)
class SeparatedAtoms:
    atoms_nowan: ase.Atoms  # ワニエ（X）を除いた構造
    wannier_positions: np.ndarray  # shape: (N_wannier, 3)


@dataclass(frozen=True)
class AtomsAndBc:
    atoms_nowan: ase.Atoms
    dict_bcs: dict  # shape: (N_bonds, 3)


class atoms_wan:
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

    def set_params(
        self, atoms_nowan: ase.Atoms, NUM_MOL: int, dict_mu: dict, dict_bcs: dict
    ):
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
        self.NUM_MOL = NUM_MOL
        # lpsを計算
        atomic_positions = atoms_nowan.get_positions()
        atomic_symbols = np.array(atoms_nowan.get_chemical_symbols())

        self.dict_bcs["Olp"] = atomic_positions[atomic_symbols == "O"].reshape(
            NUM_MOL, -1, 3
        )
        self.dict_bcs["Nlp"] = atomic_positions[atomic_symbols == "N"].reshape(
            NUM_MOL, -1, 3
        )

    def set_params_from_atoms(self, atoms: ase.Atoms, itp_data) -> None:
        NUM_ALL_ATOM: int = len(atoms) - atoms.get_chemical_symbols().count("X")
        NUM_MOL: int = int(NUM_ALL_ATOM / itp_data.num_atoms_per_mol)
        [atoms_nowan, wfc_list] = extract_wcs(atoms)  # atoms, X
        mol_coords: np.array = calculate_atomiccoord_pbcmol(
            atoms_nowan, itp_data.bonds_list, itp_data.representative_atom_index
        )
        dict_bcs: dict = calc_bondcenter_dict(mol_coords, itp_data.bonds)
        dict_mu: dict = convert_atoms_to_bondwfc(
            atoms_nowan,
            wfc_list,
            itp_data.bonds_type,
            itp_data.bonds_list,
            itp_data.bond_index,
            itp_data.representative_atom_index,
        )
        self.mol_coords = mol_coords
        self.set_params(atoms_nowan, NUM_MOL, dict_mu, dict_bcs)

    def make_atoms_with_wc(self) -> ase.Atoms:
        """
        元の分子座標に加えて，WCsとボンドセンターを加えたase.atomsを作成する．

        2023/6/2：今までは原子/BC,WC/ローンペアの順だったが，わかりやすさの改善のため，
        分子ごとに原子/ボンドセンター/ローンペアの順にappendすることにした．
        """
        # list_mol_coords,list_bond_centers =results
        # list_bond_wfcs,list_dbond_wfcs,list_lpO_wfcs,list_lpN_wfcs = results_wfcs

        new_coord = []
        new_atomic_num = []

        ase_atomicnumber = self.atoms_nowan.get_atomic_numbers()
        list_atomic_nums = list(np.array(ase_atomicnumber).reshape(self.NUM_MOL, -1))
        atomic_positions = self.atoms_nowan.get_positions().reshape(self.NUM_MOL, -1, 3)
        # loop over molecule
        for mol_index in range(self.NUM_MOL):
            new_coord.extend(atomic_positions[mol_index].reshape(-1, 3))
            new_atomic_num.extend(list_atomic_nums[mol_index])
            for _, value in self.dict_bcs.items():
                if len(np.ravel(value)) == 0:  # skip empty bonds
                    continue
                new_coord.extend(value[mol_index].reshape(-1, 3))
                new_atomic_num.extend([2] * len(value[mol_index]))
            for key, value in self.dict_mu.items():
                if len(np.ravel(value)) == 0:  # skip empty bonds
                    continue
                if "1_bond" in key:
                    position_wcs = value[mol_index].reshape(-1, 3) / (
                        -2 * coef
                    ) + self.dict_bcs[key][mol_index].reshape(-1, 3)
                if "2_bond" in key:
                    position_wcs = value[mol_index].reshape(-1, 3) / (
                        -4 * coef
                    ) + self.dict_bcs[key][mol_index].reshape(-1, 3)
                if "3_bond" in key:
                    position_wcs = value[mol_index].reshape(-1, 3) / (
                        -3 * coef
                    ) + self.dict_bcs[key][mol_index].reshape(-1, 3)
                if key == "Olp":
                    position_wcs = value[mol_index].reshape(-1, 3) / (
                        -4 * coef
                    ) + self.dict_bcs[key][mol_index].reshape(-1, 3)
                if key == "Nlp":
                    position_wcs = value[mol_index].reshape(-1, 3) / (
                        -2 * coef
                    ) + self.dict_bcs[key][mol_index].reshape(-1, 3)
                new_coord.extend(position_wcs)
                new_atomic_num.extend([10] * len(value[mol_index]))

        # change to numpy
        new_coord = np.array(new_coord)

        # WFCsと原子を合体させたAtomsオブジェクトを作成する．
        aseatoms_with_wc = ase.Atoms(
            new_atomic_num,
            positions=new_coord,
            cell=self.atoms_nowan.get_cell(),
            pbc=[1, 1, 1],
        )
        return aseatoms_with_wc


def calculate_atomiccoord_pbcmol(
    atoms_nowan: ase.Atoms,
    bonds_list,
    ref_atom_index: int,
    NUM_ATOM_PER_MOL: int | None = None,
) -> np.ndarray:
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
    if NUM_ATOM_PER_MOL is None:  # get num_atom_per_mol from bonds_list
        NUM_ATOM_PER_MOL = np.unique(np.array(bonds_list)).size

    # apply pbc to drs
    atomic_positions: np.ndarray = pbc(pbc_mol).compute_pbc(
        vectors_array=atoms_nowan.get_positions(),
        cell=atoms_nowan.get_cell(),
        bonds_list=bonds_list,
        NUM_ATOM_PAR_MOL=NUM_ATOM_PER_MOL,
        ref_atom_index=ref_atom_index,
    )  # [bondcent,Atom,3]
    return atomic_positions


def _sort_wfc_index(
    bondcenters: np.ndarray, wfc_list: np.ndarray, UNITCELL_VECTORS: np.ndarray
) -> np.ndarray:
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
    # bondcenters と wfc_list の各ペア間の距離を計算する
    # `bondcenters` の形状は (N, 3)、`wfc_list` の形状は (M, 3) であると仮定
    distances_bc_wc = distance_matrix.compute_distances(
        bondcenters.reshape(-1, 3), wfc_list, UNITCELL_VECTORS, pbc=True, norm=True
    )
    logger.debug("distances.shape = %s", np.shape(distances_bc_wc))
    # 各 `bondcenters` に対する近さで`wfc_list` のインデックスをソート(N*M)
    nearest_indices = np.argsort(distances_bc_wc, axis=1)
    return nearest_indices


def _check_duplicate_indices(*index_lists) -> None:
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
    flattened_indices = np.concatenate(
        [np.concatenate(index_list) for index_list in index_lists]
    )
    logger.debug("Flattened indices = %s", flattened_indices)

    # Get unique indices and their counts
    unique_indices, counts = np.unique(flattened_indices, return_counts=True)
    # Elements appearing more than once
    duplicates = unique_indices[counts > 1]

    # If duplicates exist, raise an error with details
    if duplicates.size > 0:
        duplicate_info = ", ".join(
            f"{val} (×{cnt})" for val, cnt in zip(duplicates, counts[counts > 1])
        )
        raise ValueError(f"Error: Duplicate indices detected: {duplicate_info}")
    logger.debug("Flattened indices: %s", flattened_indices)


def _calculate_nearest_wfc(
    nearest_indices: np.ndarray,
    nearest_number_list: np.ndarray | None = None,
    num_wcs: int | None = None,
):
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
    if nearest_number_list is None and num_wcs is None:
        raise ValueError("Specify either `nearest_number_list` or `num_wcs`.")

    # 各 bondcenter に対して `nearest_number_list` に指定された個数を取得
    if nearest_number_list is not None:
        if len(nearest_number_list) != len(nearest_number_list):
            raise ValueError(
                "len(nearest_number_list) should be the same as len(nearest_number_list)"
            )
        indices = [
            nearest_indices[i, :count].tolist()
            for i, count in enumerate(nearest_number_list)
        ]
    elif num_wcs is not None:  # for fixed number extraction
        indices = nearest_indices[:, :num_wcs]
    _check_duplicate_indices(indices)
    return indices


def _calculate_comwcs(indices: np.ndarray, wfc_list: np.ndarray, unitcell):
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
    if len(np.shape(wfc_list)) != 2:
        raise ValueError(
            f"wfc_list should be 2D array (a,3). Got {len(np.shape(wfc_list))}."
        )
    nearest_comwfc = []
    for index in indices:
        assined_wfc = wfc_list[index]
        if len(assined_wfc) == 1:
            com_wcs = assined_wfc[0]
        else:
            wfc_distances = distance_ase.compute_distances(
                assined_wfc[0], assined_wfc, cell=unitcell
            )
            com_wcs = np.mean(wfc_distances, axis=0) + assined_wfc[0]
        nearest_comwfc.append(com_wcs)
    # nearest_comwfc = [np.mean(wfc_list[index], axis=0) for index in indices]
    return np.array(nearest_comwfc)


def _assign_dipoles(
    positions: np.ndarray,
    wfc_list: np.ndarray,
    unitcell: np.ndarray,
    nearest_number_list: np.ndarray | None = None,
    num_wcs: int | None = None,
) -> np.ndarray:
    if positions.size == 0:
        return np.empty_like(positions), []

    num_mols: int = positions.shape[0]
    logger.debug(" num_mols = %s", num_mols)

    # find nearest WCs
    nearest_indices = _sort_wfc_index(positions, wfc_list, unitcell)
    indices = _calculate_nearest_wfc(
        nearest_indices, nearest_number_list=nearest_number_list, num_wcs=num_wcs
    )
    nearest_comwfc: np.ndarray = _calculate_comwcs(indices, wfc_list, unitcell)
    distance_bc_wc = distance_2d.compute_distances(
        positions.reshape(-1, 3),
        nearest_comwfc,
        unitcell,
        pbc=True,
        norm=False,
    ).reshape(positions.shape)
    if np.max(np.linalg.norm(distance_bc_wc, axis=2)) > 2:
        logger.warning(
            "distance(bc-wc) is very large %s",
            np.max(np.linalg.norm(distance_bc_wc, axis=2)),
        )
    return distance_bc_wc, indices


def _remove_selected_wcs(indices, wfc_list: List[int]) -> List[int]:
    # indices_bc は shape = (num_mols, num_bonds, num_wfcs_per_bond)
    # これを 1 次元に平坦化してユニークなインデックスに変換
    # np.concatenateはindicesが空だとエラーになる．
    flat_indices_bc = (
        np.array([]) if len(indices) == 0 else np.unique(np.concatenate(indices))
    )

    # すべてのインデックスを作成
    all_indices = np.arange(len(wfc_list))

    # flat_indices_bc を除いたインデックスを取得
    remaining_indices = np.setdiff1d(all_indices, flat_indices_bc)

    # 除いた wfc_list を取得
    filtered_wfc_list = wfc_list[remaining_indices]
    return filtered_wfc_list


def _calculate_bonddipole(
    bondcenters: np.ndarray,
    atoms: ase.Atoms,
    wfc_list: np.ndarray,
    bonds_type: list[int],
):
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
            f"Invalid shape for bondcenters. Expected shape [a, b, 3], but got {bondcenters.shape}."
        )

    num_mols: int = bondcenters.shape[0]
    logger.debug(" num_mols = %s", num_mols)
    UNITCELL_VECTORS = atoms.get_cell()

    # calculate dipole for bond centers
    distance_bc_pbc, indices_bc = _assign_dipoles(
        bondcenters,
        wfc_list,
        UNITCELL_VECTORS,
        nearest_number_list=np.tile(bonds_type, num_mols),
    )
    list_bond_mu = (-2.0) * coef * np.einsum("j,ijk->ijk", bonds_type, distance_bc_pbc)
    if np.max(np.linalg.norm(list_bond_mu, axis=2)) > 10:
        logger.warning(
            "bond_mu is very large %s", np.max(np.linalg.norm(list_bond_mu, axis=2))
        )

    # indices_bc.shape = (num_mols, num_bonds, num_wfcs_per_bond)
    filtered_wfc_list = _remove_selected_wcs(indices_bc, wfc_list)

    # Olp, Nlp
    atomic_positions = atoms.get_positions()
    atomic_symbols = np.array(atoms.get_chemical_symbols())
    list_lp_mu = {}
    for label, element, coef_factor, num_wcs in [
        ("Olp", "O", -4.0, 2),
        ("Nlp", "N", -2.0, 1),
    ]:
        coords = atomic_positions[atomic_symbols == element].reshape(num_mols, -1, 3)
        if np.shape(coords)[1] == 0:  # no O or N
            list_lp_mu[label] = np.array([])
            indices = []
        else:
            distances, indices = _assign_dipoles(
                coords, filtered_wfc_list, UNITCELL_VECTORS, num_wcs=num_wcs
            )
            list_lp_mu[label] = coef_factor * coef * distances

        if list_lp_mu[label].size > 0:
            if np.max(np.linalg.norm(list_lp_mu[label], axis=2)) > 10:
                logger.warning(
                    "%s_mu is very large %s",
                    label.lower(),
                    np.max(np.linalg.norm(list_lp_mu[label], axis=2)),
                )

        filtered_wfc_list = _remove_selected_wcs(indices, filtered_wfc_list)
    return list_bond_mu, list_lp_mu


def convert_atoms_to_bondwfc(
    atoms_nowan: ase.Atoms, wfc_list, bonds_type, bonds_list, bond_index, ref_atom_index
):
    """calculate all the bond_mu from atoms_wan and wfc_list"""
    molcoord = calculate_atomiccoord_pbcmol(atoms_nowan, bonds_list, ref_atom_index)
    # calculate all the BCs
    bcs_coords = calc_bondcenter(molcoord, bonds_list)

    list_bond_mu, list_lp_mu = _calculate_bonddipole(
        bcs_coords, atoms_nowan, wfc_list, bonds_type
    )

    dict_bond_mu = {
        key: list_bond_mu[:, np.array(item), :] if item else np.array([])
        for key, item in bond_index.items()
    }
    dict_bond_mu["Olp"] = list_lp_mu["Olp"]
    dict_bond_mu["Nlp"] = list_lp_mu["Nlp"]
    return dict_bond_mu
