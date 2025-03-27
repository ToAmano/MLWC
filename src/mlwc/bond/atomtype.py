

"""
This module defines classes and functions for handling atom types and reading topology files.

It provides functionalities for defining atom types used in various force fields,
reading and parsing ITP topology files, and extracting bond information.
The module also includes classes for representing molecular graph structures.
"""
import numpy as np
import os
from typing import Literal
from rdkit import Chem
from mlwc.include.mlwc_logger import setup_library_logger
logger = setup_library_logger("MLWC."+__name__)


class atom_type():
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


class gaff_atom_type():
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
        "hc": atom_type("H", "H on aliphatic C"),
        "ha": atom_type("H", "H on aromatic C"),
        "hn": atom_type("H", "H on N"),
        "ho": atom_type("H", "H on O"),
        "hs": atom_type("H", "H on S"),
        "hp": atom_type("H", "H on P"),
        "o": atom_type("O", "sp2 O in C=O, COO-"),
        "oh": atom_type("O", "sp3 O in hydroxyl group"),
        "os": atom_type("O", "sp3 O in ether and ester"),
        "c": atom_type("C", "sp2 C in C=O, C=S"),
        "c1": atom_type("C", "sp1 C"),
        "c2": atom_type("C", "sp2 C, aliphatic"),
        "c3": atom_type("C", "sp3 C"),
        "c6": atom_type("C", "sp3 C"),  # 2023/10/21 added for 14-dioxane
        "ca": atom_type("C", "sp2 C, aromatic"),
        "n": atom_type("N", "sp2 N in amide"),
        "n1": atom_type("N", "sp1 N "),
        "n2": atom_type("N", "sp2 N with 2 subst."),
        "n3": atom_type("N", "sp3 N with 3 subst."),  # readl double bond  ?
        "n4": atom_type("N", "sp3 N with 4 subst."),
        "na": atom_type("N", "sp2 N with 3 subst. "),
        "nh": atom_type("N", "amine N connected to the aromatic rings."),
        "no": atom_type("N", "N in nitro group."),
        "s2": atom_type("S", "sp2 S (p=S, C=S etc)"),
        "sh": atom_type("S", "sp2 S (p=S, C=S etc)"),
        "ss": atom_type("S", "sp2 S (p=S, C=S etc)"),
        "s4": atom_type("S", "sp2 S (p=S, C=S etc)"),
        "s6": atom_type("S", "sp2 S (p=S, C=S etc)"),
        # 以下 urlでspecial atom typeと言われているもの
        "h1": atom_type("H", "H on aliphatic C with 1 EW group"),
        "h2": atom_type("H", "H on aliphatic C with 2 EW group"),
        "h3": atom_type("H", "H on aliphatic C with 3 EW group"),
        "h4": atom_type("H", "H on aliphatic C with 4 EW group"),
        "h5": atom_type("H", "H on aliphatic C with 5 EW group"),
        "n": atom_type("N", "aromatic nitrogen"),
        "nb": atom_type("N", "inner sp2 N in conj. ring systems"),
        "nc": atom_type("N", "inner sp2 N in conj. chain systems"),
        "nd": atom_type("N", "inner sp2 N in conj. chain systems"),
        "sx": atom_type("S", "conj. S, 3 subst."),
        "sy": atom_type("S", "conj. S, 4 subst."),
        "cc": atom_type("C", "inner sp2 C in conj. ring systems"),
        "cd": atom_type("C", "inner sp2 C in conj. ring systems"),
        "ce": atom_type("C", "inner sp2 C in conj. chain systems"),
        "cf": atom_type("C", "inner sp2 C in conj. chain systems"),
        "cp": atom_type("C", "bridge aromatic C"),
        "cq": atom_type("C", "bridge aromatic C"),
        "cu": atom_type("C", "sp2 C in three-memberred rings"),
        "cv": atom_type("C", "sp2 C in four-memberred rings "),
        "cx": atom_type("C", "sp3 C in three-memberred rings"),
        "cy": atom_type("C", "sp3 C in four-memberred rings "),
        "pb": atom_type("P", "aromatic phosphorus"),
        "pc": atom_type("P", "inner sp2 P in conj. ring systems "),
        "pd": atom_type("P", "inner sp2 P in conj. ring systems "),
        "pe": atom_type("P", "inner sp2 P in conj. chain systems"),
        "pf": atom_type("P", "inner sp2 P in conj. chain systems"),
        "px": atom_type("P", "conj. P, 3 subst."),
        "py": atom_type("P", "conj. P, 4 subst."),
    }


class read_itp():
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
        with open(filename) as f:
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
        while len(lines[indx+2+bi]) > 5:  # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            p = int(lines[indx+2+bi][0])-1
            q = int(lines[indx+2+bi][1])-1
            bonds_list.append([p, q])
            bi = bi+1
        self.bonds_list = bonds_list

        # * 原子数を読み込む
        for i, l in enumerate(lines):
            if "atoms" in l:
                indx = i
            counter = 0
        while len(lines[indx+2+counter]) > 5:  # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            counter = counter+1
        # １つの分子内の総原子数
        self.num_atoms_per_mol = counter

        # * 原子タイプを読み込む
        atomic_type = []
        for i, l in enumerate(lines):
            if "atoms" in l:
                indx = i
        counter = 0
        while len(lines[indx+2+counter]) > 5:  # bondsを見つけてから，空行へ行くまで．カラムが6以上ならば読み込む．
            atomic_type.append(lines[indx+2+counter][1])
            counter = counter+1
        self.atomic_type = atomic_type

        # * 原子種を割り当てる．
        atom_list = []
        for i in atomic_type:
            atom_list.append(gaff_atom_type.atomlist[i].atom)
        self.atom_list = atom_list

        logger.info(" -----  ml.read_itp  :: parse results... -------")
        logger.info(" bonds_list :: ", self.bonds_list)
        logger.info(" counter    :: ", self.num_atoms_per_mol)
        logger.info(" atomic_type:: ", self.atomic_type)
        logger.info(" atom_list  :: ", self.atom_list)
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
            tmp = [gaff_atom_type.atomlist[self.atomic_type[bond[0]]].atom,
                   gaff_atom_type.atomlist[self.atomic_type[bond[1]]].atom]
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

        if len(ch_bond)+len(co_bond)+len(oh_bond)+len(oo_bond)+len(cc_bond)+len(ring_bond) != len(self.bonds_list):
            logger.info(" ")
            logger.info(
                " WARNING :: There are unkown bonds in self.bonds_list... ")
            logger.info(" ")

        logger.info(" ================ ")
        logger.info(" CH bonds...      ", self.ch_bond)
        logger.info(" CO bonds...      ", self.co_bond)
        logger.info(" OH bonds...      ", self.oh_bond)
        logger.info(" OO bonds...      ", self.oo_bond)
        logger.info(" CC bonds...      ", self.cc_bond)
        logger.info(" CC ring bonds... ", self.ring_bond)
        logger.info(" ")

        # さらに，ボンドペアのリストをボンドインデックスに変換する
        # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        self.ring_bond_index = raw_convert_bondpair_to_bondindex(
            ring_bond, self.bonds_list)
        self.bond_index['CH_1_bond'] = raw_convert_bondpair_to_bondindex(
            ch_bond, self.bonds_list)
        self.bond_index['CO_1_bond'] = raw_convert_bondpair_to_bondindex(
            co_bond, self.bonds_list)
        self.bond_index['OH_1_bond'] = raw_convert_bondpair_to_bondindex(
            oh_bond, self.bonds_list)
        self.oo_bond_index = raw_convert_bondpair_to_bondindex(
            oo_bond, self.bonds_list)
        self.bond_index['CC_1_bond'] = raw_convert_bondpair_to_bondindex(
            cc_bond, self.bonds_list)

        logger.info("")
        logger.info(" ================== ")
        logger.info(" ring_bond_index ", self.ring_bond_index)
        logger.info(" ch_bond_index   ", self.bond_index['CH_1_bond'])
        logger.info(" oh_bond_index   ", self.bond_index['OH_1_bond'])
        logger.info(" co_bond_index   ", self.bond_index['CO_1_bond'])
        logger.info(" cc_bond_index   ", self.bond_index['CC_1_bond'])
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
        logger.info(" O atoms (lonepair)...      ", self.o_list)
        logger.info(" N atoms (lonepair)...      ", self.n_list)
        return 0


def raw_convert_bondpair_to_bondindex(bonds: list[list[int]], bonds_list: list[list[int]]) -> list[int]:
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
            logger.info("there is no bond{} in bonds list.".format(b))
    return bond_index


class read_mol():
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

    """

    def __init__(self, filename: str):
        if os.path.isfile(filename) == False:
            raise ValueError(f"ERROR :: {filename} does not exist.")
        # read mol file
        mol_rdkit = Chem.MolFromMolFile(
            filename, sanitize=True, removeHs=False)
        # Chem.Kekulize(mol_rdkit)  # Do not kekulize for amortic bonds
        self.mol_rdkit = mol_rdkit  # 外部から制御できるように！（主にデバッグ用）
        # number of atoms in a single molecule
        self.num_atoms_per_mol: int = mol_rdkit.GetNumAtoms()
        # atom list (in atomic number)
        self.atom_list: list[int] = [atom.GetSymbol()
                                     for atom in mol_rdkit.GetAtoms()]

        # instance variables
        self.bonds_list: list[list[int]] = []  # list of bond index
        # list of bond type (1:single,2:double,3:triple,-1:aromatic)
        self.bonds_type: list[int] = []
        self.bonds: dict = {}
        self.bond_index: dict = {}

        for i, b in enumerate(mol_rdkit.GetBonds()):  # loop over bond
            indx0: int = b.GetBeginAtomIdx()
            indx1: int = b.GetEndAtomIdx()
            bond_type: Literal["SINGLE", "DOUBLE", "TRIPLE",
                               "AROMATIC"] = b.GetBondType()  # SINGLE,DOUBLE,TRIPLE

            self.bonds_list.append([indx0, indx1])  # append bond list
            if str(bond_type) == "SINGLE":
                self.bonds_type.append(1)
            elif str(bond_type) == "DOUBLE":
                self.bonds_type.append(2)
            elif str(bond_type) == "TRIPLE":
                self.bonds_type.append(3)
            elif str(bond_type) == "AROMATIC":
                self.bonds_type.append(10)
            else:
                raise ValueError(
                    f"ERROR :: Undefined bond type :: {str(bond_type)}")

        self._get_all_bond()  # get bond info
        # self._get_all_bondindex()
        # atomic speciesの取得
        self._get_atomic_species()
        # O/N lonepair情報の取得
        self._get_atomic_index()
        # set number of bonds
        self.num_bonds: int = len(self.bonds_list)

        # calculate the most "central" atom in the molecule usin the center of mass
        self.representative_atom_index: int = self._find_representative_atom_index()
        logger.info(" -----  ml.read_mol :: parse results... -------")
        logger.info(f" bonds_list ::  {self.bonds_list}")
        logger.info(f" num atoms per mol  :: {self.num_atoms_per_mol}")
        logger.info(f" atom_list  :: {self.atom_list}")
        logger.info(f" bonds_type :: {self.bonds_type}")
        logger.info(
            f" representative_atom_index  :: {self.representative_atom_index}")
        logger.info(" -----------------------------------------------")

        # * get COC/COH bond
        self._get_coc_and_coh_bond()

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
        for bond in self.bonds_list:
            # convert to atomic species (H,C,O,N,S)
            tmp: list[str] = [self.atom_list[bond[0]], self.atom_list[bond[1]]]

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

    def _get_general_bond(self,
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
                f"ERROR :: bondtype must be 1,2,3,10 (single,double,triple) :: bondtype = {bondtype}")
        if atom1 not in ["H", "C", "O", "N", "S", "F"] or atom2 not in ["H", "C", "O", "N", "S", "F"]:
            raise ValueError("ERROR :: atom1,atom2 must be H,C,O,N,S,F")
        bond_list = []
        if atom1 != atom2:
            for bond, type in zip(self.bonds_list, self.bonds_type):
                # 原子番号に変換
                tmp = [self.atom_list[bond[0]], self.atom_list[bond[1]]]
                if (tmp == [atom1, atom2]) & (type == bondtype):  # CH
                    bond_list.append(bond)
                elif (tmp == [atom2, atom1]) & (type == bondtype):  # HC
                    bond_list.append(bond)
        elif atom1 == atom2:
            for bond, type in zip(self.bonds_list, self.bonds_type):
                # convert to atomic species (H,C,O,N,S)
                tmp = [self.atom_list[bond[0]], self.atom_list[bond[1]]]
                if (tmp == [atom1, atom2]) & (type == bondtype):  # CC
                    bond_list.append(bond)
        return bond_list

    def _get_all_bond(self) -> int:
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
        bond_definitions = [
            # single bonds
            ("C", "H", 1), ("C", "O", 1), ("C", "C", 1), ("C", "S", 1),
            ("C", "F", 1), ("C", "N", 1), ("S", "H", 1), ("S", "O", 1),
            ("S", "N", 1), ("S", "F", 1), ("S", "S", 1), ("N", "H", 1),
            ("N", "O", 1), ("N", "N", 1), ("O", "H", 1), ("O", "O", 1),
            # double bonds
            ("C", "O", 2), ("C", "C", 2), ("C", "S", 2), ("C", "F", 2),
            ("C", "N", 2), ("S", "N", 2), ("S", "S", 2), ("N", "N", 2),
            ("N", "O", 2), ("S", "O", 2),
            # triple bonds
            ("C", "C", 3), ("C", "N", 3), ("N", "N", 3),
            # amortic bonds
            ("C", "C", 10), ("C", "N", 10), ("C", "O", 10),
        ]

        # define bonds
        for elem1, elem2, order in bond_definitions:
            bond_key = f"{elem1}{elem2}_{order}_bond"
            self.bonds[bond_key] = self._get_general_bond(elem1, elem2, order)
            self.bond_index[bond_key] = raw_convert_bondpair_to_bondindex(
                self.bonds[bond_key], self.bonds_list)

        # TODO :: aromatic bond
        self.ring_bond = []
        self.ring_bond_index = []

        logger.info(" ================ ")
        logger.info(f" CH bonds...        {self.bonds['CH_1_bond']}")
        logger.info(f" CO bonds...        {self.bonds['CO_1_bond']}")
        logger.info(f" OH bonds...        {self.bonds['OH_1_bond']}")
        logger.info(f" OO bonds...        {self.bonds['OO_1_bond']}")
        logger.info(f" CC bonds...        {self.bonds['CC_1_bond']}")
        logger.info(f" CO double bonds... {self.bonds['CO_2_bond']}")
        logger.info(" ================== ")
        logger.info(f" ch_bond_index        :: {self.bond_index['CH_1_bond']}")
        logger.info(f" oh_bond_index        :: {self.bond_index['OH_1_bond']}")
        logger.info(f" co_bond_index        :: {self.bond_index['CO_1_bond']}")
        logger.info(f" cc_bond_index        :: {self.bond_index['CC_1_bond']}")
        logger.info(f" cc_double_bond_index :: {self.bond_index['CC_2_bond']}")
        return 0

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
        self.o_list = [i for i, x in enumerate(self.atom_list) if x == "O"]
        self.n_list = [i for i, x in enumerate(self.atom_list) if x == "N"]
        self.c_list = [i for i, x in enumerate(self.atom_list) if x == "C"]
        self.h_list = [i for i, x in enumerate(self.atom_list) if x == "H"]
        self.s_list = [i for i, x in enumerate(self.atom_list) if x == "S"]
        self.f_list = [i for i, x in enumerate(self.atom_list) if x == "F"]
        logger.info(" ===========  _get_atomic_index ========== ")
        logger.info(f" O atoms (lonepair)...      {self.o_list}")
        logger.info(f" N atoms (lonepair)...      {self.n_list}")
        logger.info(f" C atoms ...                {self.c_list}")
        logger.info(f" H atoms ...                {self.h_list}")
        logger.info(f" S atoms ...                {self.s_list}")
        logger.info(f" F atoms ...                {self.f_list}")
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
                logger.info("there is no bond{} in bonds list.".format(b))
        return bond_index

    def _find_representative_atom_index(self):
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
        logger.info(" ================ ")
        logger.info("  Atomic coordinates ")
        for i, atom in enumerate(self.mol_rdkit.GetAtoms()):
            positions = self.mol_rdkit.GetConformer().GetAtomPosition(i)
            # logger.info(atom.GetSymbol(), positions.x, positions.y, positions.z)
            if atom.GetSymbol() != "H":  # H以外の原子のみを取り出す
                logger.info(
                    f" {atom.GetSymbol()} {positions.x} {positions.y} {positions.z}")
                positions_skelton.append(
                    np.array([positions.x, positions.y, positions.z]))
                index_tmp.append(i)
        # 平均値を求める
        positions_skelton = np.array(positions_skelton)
        positions_mean = np.mean(positions_skelton, axis=0)
        # positions_meanに一番近い原子を探す
        distance = np.linalg.norm(positions_skelton - positions_mean, axis=1)
        # return atomic index which gives the minimal distance
        return index_tmp[np.argmin(distance)]

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
        # * o_listのindexをcocとcohへ割り振る
        # o_indexが入っているボンドリストを探索する．

        #
        # * 次にtrue_yの分離のために，各true_COC,true_COHに属するcoボンド,ohボンドのインデックスを得る
        # あくまで，ch_bond,oh_bondの中で何番目かという情報が重要．
        # TODO :: もちろん，原子indexだけ取得しておいて後から.indexで何番目にあるかを取得した方が綺麗かもしれないが．
        # TODO :: 同様に，ボンドの番号もbond_indexの番号で取得しておいた方が楽かもしれない．

        # cocとなるoのindex(indexとはo_listの中で何番目かということで，atom_listのindexではない)
        self.coc_index = []
        self.coh_index = []  # cohとなるoのindex

        # !! o_num = the number of O
        for o_num, o_index in enumerate(self.o_list):
            # logger.info(o_index)
            neighbor_atoms = []  # o_indexに隣接する原子の情報を格納する
            for bond in self.bonds_list:  # search o_index in self.bonds_list
                if bond[0] == o_index:
                    neighbor_atoms.append([self.atom_list[bond[1]], bond])
                elif bond[1] == o_index:
                    neighbor_atoms.append([self.atom_list[bond[0]], bond])
            # もしも隣接原子が2つでない場合はスキップする．COなど
            if len(neighbor_atoms) != 2:
                continue

            # 原子種情報だけ取り出す
            neighbor_atoms_tmp = [neighbor_atoms[0][0], neighbor_atoms[1][0]]

            if neighbor_atoms_tmp == ["C", "H"]:  # COH
                index_co = self.bonds['CO_1_bond'].index(neighbor_atoms[0][1])
                index_oh = self.bonds['OH_1_bond'].index(neighbor_atoms[1][1])

                # index_C = itp_data.c_list.index(neighbor_atoms[0][1])
                # index_H = itp_data.h_list.index(neighbor_atoms[1][1])
                self.coh_index.append(
                    [o_num, o_index, {"CO": index_co, "OH": index_oh}])
            elif neighbor_atoms_tmp == ["H", "C"]:  # COH
                index_co = self.bonds['CO_1_bond'].index(neighbor_atoms[1][1])
                index_oh = self.bonds['OH_1_bond'].index(neighbor_atoms[0][1])

                # index_C = itp_data.c_list.index(neighbor_atoms[1][1])
                # index_H = itp_data.h_list.index(neighbor_atoms[0][1])
                self.coh_index.append(
                    [o_num, o_index, {"CO": index_co, "OH": index_oh}])
            elif neighbor_atoms_tmp == ["C", "C"]:  # COC
                index_co1 = self.bonds['CO_1_bond'].index(neighbor_atoms[0][1])
                index_co2 = self.bonds['CO_1_bond'].index(neighbor_atoms[1][1])

                # index_C1 = itp_data.c_list.index(neighbor_atoms[0][1])
                # index_C2 = itp_data.c_list.index(neighbor_atoms[1][1])
                self.coc_index.append(
                    [o_num, o_index, {"CO1": index_co1, "CO2": index_co2}])
        logger.info(" ================ ")
        logger.info(
            " coh_index/coc_index :: [o indx(in O atoms only), o indx(atomic index), {co bond indx(count in co_bond_index from 0),oh bond indx}]")
        # !! TODO :: もしかしたらbond_indexを使った方が全体的にやりやすいかもしれない
        logger.info(" coh_index :: {}".format(self.coh_index))
        logger.info(" coc_index :: {}".format(self.coc_index))
        return 0

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
        self.co_without_bond_index = self.bond_index['CO_1_bond']
        self.oh_without_bond_index = self.bond_index['OH_1_bond']
        for bond in self.coc_index:
            self.co_without_bond_index.remove(
                self.bonds_list.index(self.co_bond[bond[1]["CO1"]]))
            self.co_without_bond_index.remove(
                self.bonds_list.index(self.co_bond[bond[1]["CO2"]]))
        for bond in self.coh_index:
            self.co_without_bond_index.remove(
                self.bonds_list.index(self.co_bond[bond[1]["CO"]]))
            self.oh_without_bond_index.remove(
                self.bonds_list.index(self.oh_bond[bond[1]["OH"]]))
        logger.info(" ================ ")
        logger.info(" oh_bond_indexとco_bond_indexから，coc,cohに関わるバンドを削除しているので注意．")
        logger.info(" co_without_index :: {}".format(
            self.oh_without_bond_index))
        logger.info(" oh_without_index :: {}".format(
            self.co_without_bond_index))
        return 0

    @property
    def get_num_atoms_per_mol(self) -> int:
        """
        Returns the number of atoms per molecule.

        This method returns the value of the num_atoms_per_mol attribute, which represents
        the number of atoms in a single molecule.

        Returns
        -------
        int
            The number of atoms per molecule.

        Examples
        --------
        >>> read_mol_instance = read_mol("example.mol")
        >>> num_atoms = read_mol_instance.get_num_atoms_per_mol
        >>> print(num_atoms)
        10
        """
        return self.num_atoms_per_mol


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
