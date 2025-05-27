"""
This module defines classes and functions for handling atom types and reading topology files.

It provides functionalities for defining atom types used in various force fields,
reading and parsing ITP topology files, and extracting bond information.
The module also includes classes for representing molecular graph structures.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Tuple

import numpy as np
from rdkit import Chem

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


class BondType(Enum):
    """Definition of bond types"""

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 10


@dataclass(frozen=True)
class BondDefinition:
    """Definition of each bond"""

    atom1: str
    atom2: str
    bond_type: BondType

    def get_key(self) -> str:
        """generate a unique key for the bond definition"""
        return f"{self.atom1}{self.atom2}_{self.bond_type.value}_bond"


@dataclass
class MolecularInfo:
    """Class to hold molecular information extracted from a molecule."""

    # Basic molecular information
    mol_rdkit: Chem.Mol
    num_atoms_per_mol: int
    atom_list: List[str]
    bonds_list: List[List[int]]
    num_bonds: int
    bonds_type: List[int]
    representative_atom_index: int

    # Additional bond information
    bonds: Dict[str, List[List[int]]]  # Dict of bond pairs
    bond_index: Dict[str, List[int]]  # Dict of bond index

    # atomic index
    o_list: List[int]
    n_list: List[int]
    c_list: List[int]
    h_list: List[int]
    s_list: List[int]
    f_list: List[int]

    # special bond information
    coh_index: List[List]  # COH
    coc_index: List[List]  # COC

    # amorphic bonds
    ring_bond: List[List] = []
    ring_bond_index: List[int] = []


class AtomType:
    """
    Represents an atomic type and its description used in various force fields.

    Attributes
    ----------
    atom : str
        The atomic symbol (e.g., "H", "C", "O").
    description : str
        A description of the atomic type (e.g., "H on aliphatic C").

    Examples
    --------
    >>> atom_type = atom_type("H", "H on aliphatic C")
    >>> print(atom_type.atom)
    H
    >>> print(atom_type.description)
    H on aliphatic C
    """

    def __init__(self, atom, description):
        self.atom = atom
        self.description = description


class GaffAtomType:
    """
    Defines atom types as specified in the GAFF force field.

    See https://ambermd.org/antechamber/gaff.html#atomtype for details on the atom type definitions.

    Attributes
    ----------
    atomlist : dict
        A dictionary mapping GAFF atom type names to `atom_type` objects.

    Examples
    --------
    >>> gaff_atom_type.atomlist["hc"].atom
    'H'
    >>> gaff_atom_type.atomlist["hc"].description
    'H on aliphatic C'
    """

    atomlist = {
        "hc": AtomType("H", "H on aliphatic C"),
        "ha": AtomType("H", "H on aromatic C"),
        "hn": AtomType("H", "H on N"),
        "ho": AtomType("H", "H on O"),
        "hs": AtomType("H", "H on S"),
        "hp": AtomType("H", "H on P"),
        "o": AtomType("O", "sp2 O in C=O, COO-"),
        "oh": AtomType("O", "sp3 O in hydroxyl group"),
        "os": AtomType("O", "sp3 O in ether and ester"),
        "c": AtomType("C", "sp2 C in C=O, C=S"),
        "c1": AtomType("C", "sp1 C"),
        "c2": AtomType("C", "sp2 C, aliphatic"),
        "c3": AtomType("C", "sp3 C"),
        "c6": AtomType("C", "sp3 C"),  # 2023/10/21 added for 14-dioxane
        "ca": AtomType("C", "sp2 C, aromatic"),
        "n": AtomType("N", "sp2 N in amide"),
        "n1": AtomType("N", "sp1 N "),
        "n2": AtomType("N", "sp2 N with 2 subst."),
        "n3": AtomType("N", "sp3 N with 3 subst."),  # readl double bond  ?
        "n4": AtomType("N", "sp3 N with 4 subst."),
        "na": AtomType("N", "sp2 N with 3 subst. "),
        "nh": AtomType("N", "amine N connected to the aromatic rings."),
        "no": AtomType("N", "N in nitro group."),
        "s2": AtomType("S", "sp2 S (p=S, C=S etc)"),
        "sh": AtomType("S", "sp2 S (p=S, C=S etc)"),
        "ss": AtomType("S", "sp2 S (p=S, C=S etc)"),
        "s4": AtomType("S", "sp2 S (p=S, C=S etc)"),
        "s6": AtomType("S", "sp2 S (p=S, C=S etc)"),
        # 以下 urlでspecial atom typeと言われているもの
        "h1": AtomType("H", "H on aliphatic C with 1 EW group"),
        "h2": AtomType("H", "H on aliphatic C with 2 EW group"),
        "h3": AtomType("H", "H on aliphatic C with 3 EW group"),
        "h4": AtomType("H", "H on aliphatic C with 4 EW group"),
        "h5": AtomType("H", "H on aliphatic C with 5 EW group"),
        "n": AtomType("N", "aromatic nitrogen"),
        "nb": AtomType("N", "inner sp2 N in conj. ring systems"),
        "nc": AtomType("N", "inner sp2 N in conj. chain systems"),
        "nd": AtomType("N", "inner sp2 N in conj. chain systems"),
        "sx": AtomType("S", "conj. S, 3 subst."),
        "sy": AtomType("S", "conj. S, 4 subst."),
        "cc": AtomType("C", "inner sp2 C in conj. ring systems"),
        "cd": AtomType("C", "inner sp2 C in conj. ring systems"),
        "ce": AtomType("C", "inner sp2 C in conj. chain systems"),
        "cf": AtomType("C", "inner sp2 C in conj. chain systems"),
        "cp": AtomType("C", "bridge aromatic C"),
        "cq": AtomType("C", "bridge aromatic C"),
        "cu": AtomType("C", "sp2 C in three-memberred rings"),
        "cv": AtomType("C", "sp2 C in four-memberred rings "),
        "cx": AtomType("C", "sp3 C in three-memberred rings"),
        "cy": AtomType("C", "sp3 C in four-memberred rings "),
        "pb": AtomType("P", "aromatic phosphorus"),
        "pc": AtomType("P", "inner sp2 P in conj. ring systems "),
        "pd": AtomType("P", "inner sp2 P in conj. ring systems "),
        "pe": AtomType("P", "inner sp2 P in conj. chain systems"),
        "pf": AtomType("P", "inner sp2 P in conj. chain systems"),
        "px": AtomType("P", "conj. P, 3 subst."),
        "py": AtomType("P", "conj. P, 4 subst."),
    }


class ReadItpFile:
    """
    Reads and parses a Gromacs ITP topology file.

    This class extracts information such as bonds, atom types, and atom lists from an ITP file.
    Currently, it only supports the GAFF force field.

    Attributes
    ----------
    bonds_list : list[list[int]]
        A list of bonds, where each bond is represented as a list of two atom indices.
    num_atoms_per_mol : int
        The number of atoms per molecule.
    atomic_type : list[str]
        A list of GAFF atom types for each atom.
    atom_list : list[str]
        A list of atomic symbols (e.g., "H", "C", "O") for each atom.
    ch_bond : list[list[int]]
        A list of C-H bonds.
    co_bond : list[list[int]]
        A list of C-O bonds.
    cc_bond : list[list[int]]
        A list of C-C bonds.
    oh_bond : list[list[int]]
        A list of O-H bonds.
    oo_bond : list[list[int]]
        A list of O-O bonds.
    ring_bond : list[list[int]]
        A list of bonds forming rings.
    o_list : list[int]
        A list of indices for oxygen atoms.
    n_list : list[int]
        A list of indices for nitrogen atoms.
    representative_atom_index : int
        The index of a representative atom in the molecule.

    Examples
    --------
    >>> import ml.atomtype
    >>> tol_data = ml.atomtype.read_itp("input1.itp")
    >>> tol_data.ch_bond  # A list containing C-H bonds

    Notes
    -----
    - Currently, only the GAFF force field is supported.
    - Only C, H, and O atoms are fully implemented. P, N, and S are not yet implemented.
    """

    def __init__(self, filename):
        with open(filename, encoding="utf-8") as f:
            lines = f.read().splitlines()
        lines = [l.split() for l in lines]
        logger.info(" -----------------------------------------------")
        logger.info(" CAUTION !! COC/COH bond is not implemented in read_itp.")
        logger.info(" PLEASE use read_mol")
        logger.info(" -----------------------------------------------")
        # * ボンドの情報を読み込む．
        for i, l in enumerate(lines):
            if "bonds" in l:
                indx = i
        #
        bonds_list = []
        bi = 0
        while (
            len(lines[indx + 2 + bi]) > 5
        ):  # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            p = int(lines[indx + 2 + bi][0]) - 1
            q = int(lines[indx + 2 + bi][1]) - 1
            bonds_list.append([p, q])
            bi = bi + 1
        self.bonds_list = bonds_list

        # * 原子数を読み込む
        for i, l in enumerate(lines):
            if "atoms" in l:
                indx = i
            counter = 0
        while (
            len(lines[indx + 2 + counter]) > 5
        ):  # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            counter = counter + 1
        # １つの分子内の総原子数
        self.num_atoms_per_mol = counter

        # * 原子タイプを読み込む
        atomic_type = []
        for i, l in enumerate(lines):
            if "atoms" in l:
                indx = i
        counter = 0
        while (
            len(lines[indx + 2 + counter]) > 5
        ):  # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            atomic_type.append(lines[indx + 2 + counter][1])
            counter = counter + 1
        self.atomic_type = atomic_type

        # * 原子種を割り当てる．
        atom_list = []
        for i in atomic_type:
            atom_list.append(GaffAtomType.atomlist[i].atom)
        self.atom_list = atom_list

        logger.info(" -----  ml.read_itp  :: parse results... -------")
        logger.info(" bonds_list :: %s", self.bonds_list)
        logger.info(" counter    :: %s", self.num_atoms_per_mol)
        logger.info(" atomic_type:: %s", self.atomic_type)
        logger.info(" atom_list  :: %s", self.atom_list)
        logger.info(" -----------------------------------------------")

        # bond情報の取得
        self._get_bonds()
        # O/N lonepair情報の取得
        self._get_atomic_index()

        # 分子を表現するための原子のindexを指定
        # TODO :: itpファイルからこれを計算する部分を実装したい．
        self.representative_atom_index = 0

    def _get_bonds(self):
        """
        Extracts specific bond types (C-H, C-O, O-H, O-O, C-C) from the bonds_list.

        This method identifies and categorizes bonds based on the atomic symbols of the bonded atoms.
        It distinguishes between C-C bonds in aromatic rings and other C-C bonds.
        """
        ch_bond = []
        co_bond = []
        oh_bond = []
        oo_bond = []
        cc_bond = []
        ring_bond = []  # これがベンゼン環
        for bond in self.bonds_list:
            # 原子タイプに変換
            tmp_type = [self.atomic_type[bond[0]], self.atomic_type[bond[1]]]
            # 原子種に変換
            tmp = [
                GaffAtomType.atomlist[self.atomic_type[bond[0]]].atom,
                GaffAtomType.atomlist[self.atomic_type[bond[1]]].atom,
            ]
            if tmp == ["H", "C"] or tmp == ["C", "H"]:
                ch_bond.append(bond)
            if tmp == ["O", "C"] or tmp == ["C", "O"]:
                co_bond.append(bond)
            if tmp == ["O", "H"] or tmp == ["H", "O"]:
                oh_bond.append(bond)
            if tmp == ["O", "O"]:
                oo_bond.append(bond)
            if tmp == ["C", "C"]:  # CC結合はベンゼンとそれ以外で分ける
                if tmp_type != ["ca", "ca"]:  # ベンゼン環以外
                    cc_bond.append(bond)
                if tmp_type == ["ca", "ca"]:  # ベンゼン
                    ring_bond.append(bond)

        # TODO :: ベンゼン環は複数のリングに分解する．
        # この時，ナフタレンのようなことを考えると，完全には繋がっていない部分で分割するのが良い．
        # divide_cc_ring(ring_bond)

        self.ch_bond = ch_bond
        self.co_bond = co_bond
        self.oh_bond = oh_bond
        self.oo_bond = oo_bond
        self.cc_bond = cc_bond
        self.ring_bond = ring_bond

        if len(ch_bond) + len(co_bond) + len(oh_bond) + len(oo_bond) + len(
            cc_bond
        ) + len(ring_bond) != len(self.bonds_list):
            logger.info(" ")
            logger.info(" WARNING :: There are unkown bonds in self.bonds_list... ")
            logger.info(" ")

        logger.info(" ================ ")
        logger.info(" CH bonds...      %s", self.ch_bond)
        logger.info(" CO bonds...      %s", self.co_bond)
        logger.info(" OH bonds...      %s", self.oh_bond)
        logger.info(" OO bonds...      %s", self.oo_bond)
        logger.info(" CC bonds...      %s", self.cc_bond)
        logger.info(" CC ring bonds... %s", self.ring_bond)
        logger.info(" ")

        # さらに，ボンドペアのリストをボンドインデックスに変換する
        # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        self.ring_bond_index = raw_convert_bondpair_to_bondindex(
            ring_bond, self.bonds_list
        )
        self.bond_index["CH_1_bond"] = raw_convert_bondpair_to_bondindex(
            ch_bond, self.bonds_list
        )
        self.bond_index["CO_1_bond"] = raw_convert_bondpair_to_bondindex(
            co_bond, self.bonds_list
        )
        self.bond_index["OH_1_bond"] = raw_convert_bondpair_to_bondindex(
            oh_bond, self.bonds_list
        )
        self.oo_bond_index = raw_convert_bondpair_to_bondindex(oo_bond, self.bonds_list)
        self.bond_index["CC_1_bond"] = raw_convert_bondpair_to_bondindex(
            cc_bond, self.bonds_list
        )

        logger.info("")
        logger.info(" ================== ")
        logger.info(" ring_bond_index %s", self.ring_bond_index)
        logger.info(" ch_bond_index   %s", self.bond_index["CH_1_bond"])
        logger.info(" oh_bond_index   %s", self.bond_index["OH_1_bond"])
        logger.info(" co_bond_index   %s", self.bond_index["CO_1_bond"])
        logger.info(" cc_bond_index   %s", self.bond_index["CC_1_bond"])
        return 0

    def divide_cc_ring(self):
        """
        TODO :: Divides the CC rings into connected and unconnected parts.
        """
        return 0

    def _get_atomic_index(self):
        """
        Finds and returns the indices of atoms with lone pairs (O, N) from self.atom_list.

        Returns
        -------
        None
        """
        self.o_list = [i for i, x in enumerate(self.atom_list) if x == "O"]
        self.n_list = [i for i, x in enumerate(self.atom_list) if x == "N"]
        logger.info(" ================ ")
        logger.info(" O atoms (lonepair)...     %s ", self.o_list)
        logger.info(" N atoms (lonepair)...     %s ", self.n_list)
        return 0


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
        self.ring_bond: list[list] = []
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

    @DeprecationWarning
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

    def _get(self):
        # 各 bondcenter に対して，取得するべきWCsの数を取得
        for key, value in itp_data.bond_index.items():
            logger.debug(key, value)
            if "1" in key:
                nearest_number_list[np.array(value)] = 1
            elif "2" in key:
                nearest_number_list[np.array(value)] = 2
            elif "3" in key:
                nearest_number_list[np.array(value)] = 2
        if 0 in nearest_number_list:
            raise ValueError("failed to make nearest_number_list !!")
        # repeat for num_mols
        nearest_number_list = np.repeat(nearest_number_list, num_mols)

    def raw_convert_bondpair_to_bondindex(self, bonds, bonds_list):
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
        bond_index = []
        for b in bonds:
            if b in bonds_list:
                bond_index.append(bonds_list.index(b))
            elif b[::-1] in bonds:
                bond_index.append(bonds_list.index(b[::-1]))
            else:
                logger.info("there is no bond %s in bonds list.", b)
        return bond_index

    def _find_neighbors(self, atom_idx: int):
        neighbor_atoms = []  # o_indexに隣接する原子の情報を格納する
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
        self.co_without_bond_index = self._bond_index["CO_1_bond"]
        self.oh_without_bond_index = self._bond_index["OH_1_bond"]
        for bond in self.coc_index:
            self.co_without_bond_index.remove(
                self._bonds_list.index(self.co_bond[bond[1]["CO1"]])
            )
            self.co_without_bond_index.remove(
                self._bonds_list.index(self.co_bond[bond[1]["CO2"]])
            )
        for bond in self.coh_index:
            self.co_without_bond_index.remove(
                self._bonds_list.index(self.co_bond[bond[1]["CO"]])
            )
            self.oh_without_bond_index.remove(
                self._bonds_list.index(self.oh_bond[bond[1]["OH"]])
            )
        logger.info(" ================ ")
        logger.info(
            " oh_bond_indexとco_bond_indexから，coc,cohに関わるバンドを削除しているので注意．"
        )
        logger.info(" co_without_index :: %s", self.oh_without_bond_index)
        logger.info(" oh_without_index :: %s", self.co_without_bond_index)
        return 0

    @property
    def num_atoms_per_mol(self) -> int:
        return self._num_atoms_per_mol

    @property
    def atom_list(self) -> list[str]:
        return self._atom_list

    @property
    def bonds_list(self) -> list[list[int]]:
        return self._bonds_list

    @property
    def num_bonds(self) -> int:
        return self._num_bonds

    @property
    def bonds_type(self) -> list[int]:
        return self._bonds_type

    @property
    def bonds(self) -> list[int]:
        return self._bonds

    @property
    def bond_index(self) -> list[int]:
        return self._bond_index

    @property
    def representative_atom_index(self) -> int:
        return self._representative_atom_index

    @property
    def coh_index(self):
        return self._coh_index

    @property
    def coc_index(self):
        return self._coc_index


class Node:  # 分子情報（itp）をグラフ情報に格納するためのクラス
    """
    Represents a node in a molecular graph.

    This class stores information about a node (atom) in a molecular graph,
    including its index, neighboring atoms, and parent node.

    Attributes
    ----------
    index : int
        The index of the node (atom).
    nears : list[int]
        A list of indices of neighboring atoms.
    parent : int
        The index of the parent node in a tree traversal of the graph.
    """

    def __init__(self, index):
        self.index = index
        self.nears = []
        self.parent = -1  # 親はまだ決まっていないので-1としておく

    def __repr__(self):
        return f"(index:{self.index}, nears:{self.nears}, parent:{self.parent})"


def make_bondgraph(bonds_list: list, num_atoms_per_mol: int):
    """
    Creates a bond graph from a list of bonds and the number of atoms per molecule.

    This function takes a list of bonds and the number of atoms per molecule
    and returns a list of Node objects representing the bond graph.

    Parameters
    ----------
    bonds_list : list[list[int]]
        A list of bonds, where each bond is represented as a list of two atom indices.
    num_atoms_per_mol : int
        The number of atoms per molecule.

    Returns
    -------
    list[Node]
        A list of Node objects representing the bond graph.

    Examples
    --------
    >>> bonds_list = [[0, 1], [1, 2], [2, 0]]
    >>> num_atoms_per_mol = 3
    >>> graph = make_bondgraph(bonds_list, num_atoms_per_mol)
    >>> print(graph)
    [(index:0, nears:[1, 2], parent:-1), (index:1, nears:[0, 2], parent:-1), (index:2, nears:[1, 0], parent:-1)]
    """
    # Nodeインスタンスを作成しnodesに格納
    nodes = [Node(i) for i in range(num_atoms_per_mol)]

    # 隣接リストを付与
    for bond in bonds_list:
        nodes[bond[0]].nears.append(bond[1])
        nodes[bond[1]].nears.append(bond[0])

    return nodes


def raw_make_graph_from_itp(itp_data):
    """
    Creates a bond graph from ITP data.

    This function takes ITP data containing bond information and the number of atoms per molecule
    and returns a list of Node objects representing the bond graph.

    Parameters
    ----------
    itp_data : read_itp
        An instance of the read_itp class containing bond and atom information.

    Returns
    -------
    list[Node]
        A list of Node objects representing the bond graph.

    Examples
    --------
    >>> from mlwc.ml.atomtype import read_itp
    >>> itp_data = read_itp("example.itp")
    >>> graph = raw_make_graph_from_itp(itp_data)
    >>> print(graph)
    [(index:0, nears:[1, 2], parent:-1), (index:1, nears:[0, 2], parent:-1), (index:2, nears:[1, 0], parent:-1)]
    """
    # Nodeインスタンスを作成しnodesに格納
    nodes = [Node(i) for i in range(itp_data.num_atoms_per_mol)]

    # 隣接リストを付与
    for bond in itp_data.bonds_list:
        nodes[bond[0]].nears.append(bond[1])
        nodes[bond[1]].nears.append(bond[0])

    return nodes


def _init_bonds_list(mol_rdkit) -> list[list[int]]:
    bonds_list = []
    for bond in mol_rdkit.GetBonds():
        indx0 = bond.GetBeginAtomIdx()
        indx1 = bond.GetEndAtomIdx()
        bonds_list.append([indx0, indx1])
    return bonds_list


def _init_bonds_type(mol_rdkit) -> list[int]:
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


def _find_representative_atom_index(mol_rdkit) -> int:
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
    return index_tmp[np.argmin(distance)]
