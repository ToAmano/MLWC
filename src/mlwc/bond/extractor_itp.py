from dataclasses import dataclass

from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


@dataclass(frozen=True)
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

    atom: str
    description: str


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
        # "n": AtomType("N", "aromatic nitrogen"),
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
        lines = [line.split() for line in lines]
        logger.info(" -----------------------------------------------")
        logger.info(" CAUTION !! COC/COH bond is not implemented in read_itp.")
        logger.info(" PLEASE use read_mol")
        logger.info(" -----------------------------------------------")

        self.num_atoms_per_mol = self._get_num_atoms_per_mol(lines)
        self.bonds_list = self._get_bonds_list(lines)
        self.atomic_type = self._get_atomic_type(lines)

        # * 原子種を割り当てる．
        atom_list = []
        for i in self.atomic_type:
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

        # TODO :: itpファイルからこれを計算する部分を実装したい．
        self.representative_atom_index = 0

    def _get_num_atoms_per_mol(self, lines):
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
        return counter

    def _get_bonds_list(self, lines):
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
        return bonds_list

    def _get_atomic_type(self, lines):
        """read atomic type"""
        for i, l in enumerate(lines):
            if "bonds" in l:
                indx = i
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
        return atomic_type

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

        if len(ch_bond) + len(co_bond) + len(oh_bond) + len(oo_bond) + len(
            cc_bond
        ) + len(ring_bond) != len(self.bonds_list):
            logger.info(" ")
            logger.info(" WARNING :: There are unkown bonds in self.bonds_list... ")
            logger.info(" ")

        # さらに，ボンドペアのリストをボンドインデックスに変換する
        # 実際のボンド[a,b]から，ボンド番号（bonds.index）への変換を行う
        self.ring_bond_index = raw_convert_bondpair_to_bondindex(
            ring_bond, self.bonds_list
        )
        self.bond_index = {}
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
