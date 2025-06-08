import os
from typing import List, Literal, Tuple

import numpy as np
from rdkit import Chem

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


class ReadMolFile:
    """RDKit implementation to retrieve bond information from mol file.
    https://future-chem.com/rdkit-mol/

     Attributes:
    ----------
    num_atoms_per_mol : int
        number of atoms in a single molecule
    atom_list : list[int]
        list of atomic numbers
        atomic species (H,C,O,N,S)
    bonds_list : list[list[int]]
        list of bond index
    bonds_type : list[int]
        list of bond type (1:single,2:double,3:triple,-1:aromatic)
    representative_atom_index : int
        index of the most "central" atom in the molecule

    Notes:
    -----------
    - Do not kekulize for amortic bonds
    """

    allowed_atom: list[str] = ["H", "C", "O", "N", "S", "F"]
    bond_definitions: list[
        Literal["H", "C", "O", "N", "S", "F"],
        Literal["H", "C", "O", "N", "S", "F"],
        int,
    ] = [
        # single bonds
        ("C", "H", 1),
        ("C", "O", 1),
        ("C", "C", 1),
        ("C", "S", 1),
        ("C", "F", 1),
        ("C", "N", 1),
        ("S", "H", 1),
        ("S", "O", 1),
        ("S", "N", 1),
        ("S", "F", 1),
        ("S", "S", 1),
        ("N", "H", 1),
        ("N", "O", 1),
        ("N", "N", 1),
        ("O", "H", 1),
        ("O", "O", 1),
        # double bonds
        ("C", "O", 2),
        ("C", "C", 2),
        ("C", "S", 2),
        ("C", "F", 2),
        ("C", "N", 2),
        ("S", "N", 2),
        ("S", "S", 2),
        ("N", "N", 2),
        ("N", "O", 2),
        ("S", "O", 2),
        # triple bonds
        ("C", "C", 3),
        ("C", "N", 3),
        ("N", "N", 3),
        # amortic bonds
        ("C", "C", 10),
        ("C", "N", 10),
        ("C", "O", 10),
    ]

    def __init__(self, filename: str):
        if os.path.isfile(filename) is False:
            raise ValueError(f"ERROR :: {filename} does not exist.")
        # read mol file
        mol_rdkit = Chem.MolFromMolFile(filename, sanitize=True, removeHs=False)
        self._mol_rdkit = mol_rdkit
        # number of atoms in a single molecule
        self._num_atoms_per_mol: int = mol_rdkit.GetNumAtoms()
        # atom list (in atomic symbol)
        self._atom_list: list[str] = [atom.GetSymbol() for atom in mol_rdkit.GetAtoms()]
        self._bonds_list: list[list[int]] = _init_bonds_list(
            mol_rdkit
        )  # list of bond index
        self._num_bonds: int = len(self._bonds_list)  # number of bonds
        # list of bond type (1:single,2:double,3:triple,-1:aromatic)
        self._bonds_type: list[int] = _init_bonds_type(mol_rdkit)
        self._bonds, self._bond_index = self._get_all_bond()
        # TODO implement amorphic bond
        self.ring_bond: list[list[int]] = []
        self.ring_bond_index: list[int] = []
        # self._get_all_bondindex()
        self._get_atomic_species()  # atomic species
        self._get_atomic_index()  # O/N lonepair
        # calculate the most "central" atom in the molecule usin the center of mass
        self._representative_atom_index: int = _find_representative_atom_index(
            mol_rdkit
        )
        logger.info(" -----  bond.atomtype.ReadMolFile :: parse results... -------")
        logger.info(" bonds_list ::  %s", self._bonds_list)
        logger.info(" num atoms per mol  :: %s", self._num_atoms_per_mol)
        logger.info(" atom_list  :: %s", self._atom_list)
        logger.info(" bonds_type :: %s", self._bonds_type)
        logger.info(
            " representative_atom_index  :: %s", self._representative_atom_index
        )
        logger.info(" -----------------------------------------------")

        # * get COC/COH bond
        self._coh_index, self._coc_index = self._get_coc_and_coh_bond()

        # * CO/OHの結合（COC,COHに含まれないやつ）
        # self._get_co_oh_without_coc_and_coh_bond()

    def _get_atomic_species(self) -> int:
        """get atomic species from atom_list (H,C,O,N,S)
        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        c_ch = []
        c_co = []
        c_cc = []
        o_oh = []
        o_co = []
        h_ch = []
        h_oh = []
        for bond in self._bonds_list:
            # convert to atomic species (H,C,O,N,S)
            tmp: list[str] = [self._atom_list[bond[0]], self._atom_list[bond[1]]]

            if tmp == ["H", "C"]:
                h_ch.append(bond[0])
                c_ch.append(bond[1])
            elif tmp == ["C", "H"]:
                h_ch.append(bond[1])
                c_ch.append(bond[0])
            elif tmp == ["O", "C"]:
                o_co.append(bond[0])
                c_co.append(bond[1])
            elif tmp == ["C", "O"]:
                o_co.append(bond[1])
                c_co.append(bond[0])
            elif tmp == ["O", "H"]:
                o_oh.append(bond[0])
                h_oh.append(bond[1])
            elif tmp == ["H", "O"]:
                o_oh.append(bond[1])
                h_oh.append(bond[0])
            elif tmp == ["C", "C"]:
                c_cc.append(bond[0])
                c_cc.append(bond[1])
            else:
                continue
                # raise ValueError("ERROR :: Undefined bond type")
        self.c_ch = list(set(c_ch))
        self.c_co = list(set(c_co))
        self.c_cc = list(set(c_cc))
        self.o_oh = list(set(o_oh))
        self.o_co = list(set(o_co))
        self.h_ch = list(set(h_ch))
        self.h_oh = list(set(h_oh))
        return 0

    def _get_specific_bond(
        self,
        atom1: Literal["H", "C", "O", "N", "S", "F"],
        atom2: Literal["H", "C", "O", "N", "S", "F"],
        bondtype: Literal[1, 2, 3, 10],
    ) -> list[list[int]]:
        """
        Extract specific bond indices from bonds_list specified by atomic species and bond type.

        This method retrieves bond indices based on the specified atomic species and bond type.
        It supports various bond species including C-H, C-O, C-C, C-F, C-S, C-N, S-H, S-O, S-N, S-F, S-S, N-H, N-O, N-N, O-H, and O-O.

        Parameters
        ----------
        atom1 : Literal["H", "C", "O", "N", "S", "F"]
            The atomic symbol of the first atom in the bond.
        atom2 : Literal["H", "C", "O", "N", "S", "F"]
            The atomic symbol of the second atom in the bond.
        bondtype : Literal[1, 2, 3]
            The bond type: 1 for single, 2 for double, 3 for triple.

        Returns
        -------
        list[list[int]]
            A list of bond pairs (atom indices) that match the specified criteria.

        Raises
        ------
        ValueError
            If the bondtype is not 1, 2, or 3.
        ValueError
            If atom1 or atom2 is not one of "H", "C", "O", "N", "S", or "F".

        Examples
        --------
        >>> read_mol_instance = read_mol("example.mol")
        >>> ch_bonds = read_mol_instance._get_general_bond("C", "H", 1)
        >>> print(ch_bonds)
        [[0, 1], [2, 3]]
        """
        if bondtype not in [1, 2, 3, 10]:
            raise ValueError(
                f"ERROR :: bondtype must be 1,2,3,10 (single,double,triple) :: bondtype = {bondtype}"
            )
        if atom1 not in self.allowed_atom or atom2 not in self.allowed_atom:
            raise ValueError(
                f"Invalid atom type: atom1='{atom1}', atom2='{atom2}'. Allowed atoms: {self.allowed_atom}"
            )
        bond_list: list[list[int]] = []
        target_pair = {atom1, atom2}
        is_same_atom = atom1 == atom2
        for bond, btype in zip(self._bonds_list, self._bonds_type):
            if btype != bondtype:
                continue
            atom_pair = {
                self._atom_list[bond[0]],
                self._atom_list[bond[1]],
            }  # ex) {H,C}
            if atom_pair == target_pair and (is_same_atom or len(atom_pair) == 2):
                bond_list.append(bond)
        return bond_list

    def _get_all_bond(self) -> Tuple[dict, dict]:
        """
        Get all bond indices from bonds_list.

        This method retrieves all bond indices based on predefined bond definitions.
        It iterates through a list of bond definitions, including single, double, and triple bonds,
        and retrieves the corresponding bond indices using the `_get_general_bond` method.

        Returns
        -------
        int
            0 if the operation is successful.

        Examples
        --------
        >>> read_mol_instance = read_mol("example.mol")
        >>> read_mol_instance._get_all_bond()
        0
        """
        # define bonds
        bonds = {}
        bond_index = {}
        for elem1, elem2, order in self.bond_definitions:
            bond_key = f"{elem1}{elem2}_{order}_bond"
            bond_pairs = self._get_specific_bond(elem1, elem2, order)
            bonds[bond_key] = bond_pairs
            bond_index[bond_key] = raw_convert_bondpair_to_bondindex(
                bond_pairs, self._bonds_list
            )
        # TODO :: implement aromatic bond

        logger.info(" ================ ")
        for key, value in bonds.items():
            if value:
                logger.info("%s %s", key, value)
        logger.info(" ================== ")
        for key, value in bond_index.items():
            if value:
                logger.info("%s :: %s", key, value)
        return bonds, bond_index

    def _get_atomic_index(self) -> int:
        """
        Extract specific atomic indices from self.atom_list (O,N,C,H,S,F).

        This method retrieves the indices of specific atoms (O, N, C, H, S, F) from the atom list.
        It populates the o_list, n_list, c_list, h_list, s_list, and f_list attributes with the
        indices of the corresponding atoms.

        Returns
        -------
        int
            0 if the operation is successful.

        Examples
        --------
        >>> read_mol_instance = read_mol("example.mol")
        >>> read_mol_instance._get_atomic_index()
        0
        """
        # TODO :: use dict
        self.o_list = [i for i, x in enumerate(self._atom_list) if x == "O"]
        self.n_list = [i for i, x in enumerate(self._atom_list) if x == "N"]
        self.c_list = [i for i, x in enumerate(self._atom_list) if x == "C"]
        self.h_list = [i for i, x in enumerate(self._atom_list) if x == "H"]
        self.s_list = [i for i, x in enumerate(self._atom_list) if x == "S"]
        self.f_list = [i for i, x in enumerate(self._atom_list) if x == "F"]
        logger.info(" ===========  _get_atomic_index ========== ")
        logger.info(" O atoms (lonepair)...      %s", self.o_list)
        logger.info(" N atoms (lonepair)...      %s", self.n_list)
        logger.info(" C atoms ...                %s", self.c_list)
        logger.info(" H atoms ...                %s", self.h_list)
        logger.info(" S atoms ...                %s", self.s_list)
        logger.info(" F atoms ...                %s", self.f_list)
        logger.info(" ========================================= ")
        return 0

    def raw_convert_bondpair_to_bondindex(
        self, bonds: List[List[int]], bonds_list: List[List[int]]
    ) -> List[int]:
        """
        Converts a list of bond pairs to a list of bond indices.

        This method takes a list of bond pairs (atom indices) and a list of all bonds,
        and returns a list of indices corresponding to the bond pairs in the bonds list.

        Parameters
        ----------
        bonds : list[list[int]]
            A list of bond pairs, where each bond is represented as a list of two atom indices.
        bonds_list : list[list[int]]
            A list of all bonds in the molecule.

        Returns
        -------
        list[int]
            A list of bond indices corresponding to the input bond pairs.

        Examples
        --------
        >>> bonds = [[0, 1], [2, 3]]
        >>> bonds_list = [[0, 1], [2, 3], [4, 5]]
        >>> read_mol_instance = read_mol("example.mol")
        >>> bond_indices = read_mol_instance.raw_convert_bondpair_to_bondindex(bonds, bonds_list)
        >>> print(bond_indices)
        [0, 1]
        """
        bond_index: List[int] = []
        for b in bonds:
            if b in bonds_list:
                bond_index.append(bonds_list.index(b))
            elif b[::-1] in bonds:
                bond_index.append(bonds_list.index(b[::-1]))
            else:
                logger.info("there is no bond %s in bonds list.", b)
        return bond_index

    def _find_neighbors(self, atom_idx: int) -> List[int]:
        """Get neighbor atoms of a given atom index."""
        neighbor_atoms: List[int] = []
        for bond in self._bonds_list:  # search o_index in self.bonds_list
            if bond[0] == atom_idx:
                neighbor_atoms.append([self._atom_list[bond[1]], bond])
            elif bond[1] == atom_idx:
                neighbor_atoms.append([self._atom_list[bond[0]], bond])
        return neighbor_atoms

    def _get_coc_and_coh_bond(self):
        """
        Identifies C-O-C and C-O-H bonds in the molecule.

        This method identifies C-O-C and C-O-H bonds by analyzing the neighbors of each oxygen atom.
        It populates the coc_index and coh_index attributes with the indices of the oxygen atoms
        involved in these bonds.

        Returns
        -------
        int
            0 if the operation is successful.

        Examples
        --------
        >>> read_mol_instance = read_mol("example.mol")
        >>> read_mol_instance._get_coc_and_coh_bond()
        0
        """
        #
        # * 次にtrue_yの分離のために，各true_COC,true_COHに属するcoボンド,ohボンドのインデックスを得る
        # あくまで，ch_bond,oh_bondの中で何番目かという情報が重要．
        # TODO :: もちろん，原子indexだけ取得しておいて後から.indexで何番目にあるかを取得した方が綺麗かもしれないが．
        # TODO :: 同様に，ボンドの番号もbond_indexの番号で取得しておいた方が楽かもしれない．

        # cocとなるoのindex(indexとはo_listの中で何番目かということで，atom_listのindexではない)
        coc_index = []
        coh_index = []

        # !! o_num = the number of O
        for o_num, o_index in enumerate(self.o_list):
            neighbor_atoms = self._find_neighbors(
                o_index
            )  # o_indexに隣接する原子の情報を格納する
            # もしも隣接原子が2つでない場合はスキップする．C=Oなど
            if len(neighbor_atoms) != 2:
                continue
            # 原子種情報だけ取り出す
            neighbor_atoms_tmp = [neighbor_atoms[0][0], neighbor_atoms[1][0]]
            if neighbor_atoms_tmp == ["C", "H"]:  # COH
                index_co = self._bonds["CO_1_bond"].index(neighbor_atoms[0][1])
                index_oh = self._bonds["OH_1_bond"].index(neighbor_atoms[1][1])
                coh_index.append([o_num, o_index, {"CO": index_co, "OH": index_oh}])
            elif neighbor_atoms_tmp == ["H", "C"]:  # COH
                index_co = self._bonds["CO_1_bond"].index(neighbor_atoms[1][1])
                index_oh = self._bonds["OH_1_bond"].index(neighbor_atoms[0][1])
                coh_index.append([o_num, o_index, {"CO": index_co, "OH": index_oh}])
            elif neighbor_atoms_tmp == ["C", "C"]:  # COC
                index_co1 = self._bonds["CO_1_bond"].index(neighbor_atoms[0][1])
                index_co2 = self._bonds["CO_1_bond"].index(neighbor_atoms[1][1])
                coc_index.append([o_num, o_index, {"CO1": index_co1, "CO2": index_co2}])
        logger.info(" ================ ")
        logger.info(
            " coh_index/coc_index :: [o indx(in O atoms only), o indx(atomic index), {co bond indx(count in co_bond_index from 0),oh bond indx}]"
        )
        # !! TODO :: もしかしたらbond_indexを使った方が全体的にやりやすいかもしれない
        logger.info(" coh_index :: %s", coh_index)
        logger.info(" coc_index :: %s", coc_index)
        return coh_index, coc_index

    @DeprecationWarning
    def _get_co_oh_without_coc_and_coh_bond(self):
        """
        Recalculates CO and OH bonds excluding those in COC and COH bonds.

        This method recalculates the CO and OH bonds, excluding those that are part of COC and COH bonds.
        It modifies the co_without_bond_index and oh_without_bond_index attributes to store the
        indices of the remaining CO and OH bonds.

        Returns
        -------
        int
            0 if the operation is successful.

        Examples
        --------
        >>> read_mol_instance = read_mol("example.mol")
        >>> read_mol_instance._get_co_oh_without_coc_and_coh_bond()
        0
        """
        co_without_bond_index = self._bond_index["CO_1_bond"]
        oh_without_bond_index = self._bond_index["OH_1_bond"]
        for bond in self.coc_index:
            co_without_bond_index.remove(
                self._bonds_list.index(self.bonds["CO_1_bond"][bond[1]["CO1"]])
            )
            co_without_bond_index.remove(
                self._bonds_list.index(self.bonds["CO_1_bond"][bond[1]["CO2"]])
            )
        for bond in self.coh_index:
            co_without_bond_index.remove(
                self._bonds_list.index(self.bonds["CO_1_bond"][bond[1]["CO"]])
            )
            oh_without_bond_index.remove(
                self._bonds_list.index(self.bonds["OH_1_bond"][bond[1]["OH"]])
            )
        logger.info(" ================ ")
        logger.info(
            " oh_bond_indexとco_bond_indexから，coc,cohに関わるバンドを削除しているので注意．"
        )
        logger.info(" co_without_index :: %s", oh_without_bond_index)
        logger.info(" oh_without_index :: %s", co_without_bond_index)
        return oh_without_bond_index, co_without_bond_index

    @property
    def num_atoms_per_mol(self) -> int:
        """num_atoms_per_mol :int"""
        return self._num_atoms_per_mol

    @property
    def atom_list(self) -> list[str]:
        """atom_list"""
        return self._atom_list

    @property
    def bonds_list(self) -> list[list[int]]:
        """bonds_list"""
        return self._bonds_list

    @property
    def num_bonds(self) -> int:
        """num_bonds"""
        return self._num_bonds

    @property
    def bonds_type(self) -> list[int]:
        """bonds_type"""
        return self._bonds_type

    @property
    def bonds(self) -> List[List[int]]:
        """bonds"""
        return self._bonds

    @property
    def bond_index(self) -> list[int]:
        """bond_index"""
        return self._bond_index

    @property
    def representative_atom_index(self) -> int:
        """representative_atom_index"""
        return self._representative_atom_index

    @property
    def coh_index(self):
        """coh_index"""
        return self._coh_index

    @property
    def coc_index(self):
        """coc_index"""
        return self._coc_index


def _init_bonds_list(mol_rdkit: Chem.Mol) -> list[list[int]]:
    bonds_list = []
    for bond in mol_rdkit.GetBonds():
        indx0 = bond.GetBeginAtomIdx()
        indx1 = bond.GetEndAtomIdx()
        bonds_list.append([indx0, indx1])
    return bonds_list


def _init_bonds_type(mol_rdkit: Chem.Mol) -> list[int]:
    bonds_type = []
    for bond in mol_rdkit.GetBonds():
        bond_type = str(bond.GetBondType())
        match bond_type:
            case "SINGLE":
                bonds_type.append(1)
            case "DOUBLE":
                bonds_type.append(2)
            case "TRIPLE":
                bonds_type.append(3)
            case "AROMATIC":
                bonds_type.append(10)
            case _:
                raise ValueError(f"Undefined bond type: {bond_type}")
    return bonds_type


def _find_representative_atom_index(mol_rdkit: Chem.Mol) -> int:
    """
    Finds the index of the atom closest to the center of mass of the non-hydrogen atoms.

    This method calculates the center of mass of the non-hydrogen atoms in the molecule
    and returns the index of the atom that is closest to this center of mass.

    Returns
    -------
    int
        The index of the atom closest to the center of mass of the non-hydrogen atoms.

    Examples
    --------
    >>> read_mol_instance = read_mol("example.mol")
    >>> representative_atom_index = read_mol_instance._find_representative_atom_index()
    >>> print(representative_atom_index)
    0
    """
    positions_skelton = []
    index_tmp = []
    logger.info(" ===================== ")
    logger.info("  Atomic coordinates ")
    for i, atom in enumerate(mol_rdkit.GetAtoms()):
        pos = mol_rdkit.GetConformer().GetAtomPosition(i)
        if atom.GetSymbol() == "H":  # H以外の原子のみを取り出す
            continue
        logger.info(
            " %s %s %s %s",
            atom.GetSymbol(),
            pos.x,
            pos.y,
            pos.z,
        )
        positions_skelton.append(np.array([pos.x, pos.y, pos.z]))
        index_tmp.append(i)
    positions_skelton = np.array(positions_skelton)
    positions_mean = np.mean(positions_skelton, axis=0)
    distance = np.linalg.norm(positions_skelton - positions_mean, axis=1)
    # return atomic index which gives the minimal distance
    representative_atom_index: int = index_tmp[np.argmin(distance)]
    return representative_atom_index


def raw_convert_bondpair_to_bondindex(
    bonds: list[list[int]], bonds_list: list[list[int]]
) -> list[int]:
    """
    Converts a list of bond pairs to a list of bond indices.

    This function takes a list of bond pairs (atom indices) and a list of all bonds,
    and returns a list of indices corresponding to the bond pairs in the bonds list.

    Parameters
    ----------
    bonds : list[list[int]]
        A list of bond pairs, where each bond is represented as a list of two atom indices.
    bonds_list : list[list[int]]
        A list of all bonds in the molecule.

    Returns
    -------
    list[int]
        A list of bond indices corresponding to the input bond pairs.
    """
    bond_index = []
    # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
    for b in bonds:
        if b in bonds_list:  # ボンドがリストに存在する場合
            bond_index.append(bonds_list.index(b))
        elif b[::-1] in bonds:  # これはボンドの向きが逆の場合（b[1],b[0]）
            bond_index.append(bonds_list.index(b[::-1]))
        else:
            logger.info("there is no bond %s in bonds list.", b)
    return bond_index
