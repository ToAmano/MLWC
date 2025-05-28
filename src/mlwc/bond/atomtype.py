"""
This module defines classes and functions for handling atom types and reading topology files.

It provides functionalities for defining atom types used in various force fields,
reading and parsing ITP topology files, and extracting bond information.
The module also includes classes for representing molecular graph structures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple

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
    atomic_index: Dict[str, List[int]]  # Dict of atomic index

    # special bond information
    coh_index: List[List]  # COH
    coc_index: List[List]  # COC

    # amorphic bonds
    ring_bond: List[List] = None
    ring_bond_index: List[int] = None

    def __post_init__(self):
        if self.ring_bond is None:
            self.ring_bond = []
        if self.ring_bond_index is None:
            self.ring_bond_index = []


class BondAnalyzer:
    """Class to analyze bond types"""

    ALLOWED_ATOMS: Set[str] = {"H", "C", "O", "N", "S", "F"}

    BOND_DEFINITIONS = [
        # Single bonds
        BondDefinition("C", "H", BondType.SINGLE),
        BondDefinition("C", "O", BondType.SINGLE),
        BondDefinition("C", "C", BondType.SINGLE),
        BondDefinition("C", "S", BondType.SINGLE),
        BondDefinition("C", "F", BondType.SINGLE),
        BondDefinition("C", "N", BondType.SINGLE),
        BondDefinition("S", "H", BondType.SINGLE),
        BondDefinition("S", "O", BondType.SINGLE),
        BondDefinition("S", "N", BondType.SINGLE),
        BondDefinition("S", "F", BondType.SINGLE),
        BondDefinition("S", "S", BondType.SINGLE),
        BondDefinition("N", "H", BondType.SINGLE),
        BondDefinition("N", "O", BondType.SINGLE),
        BondDefinition("N", "N", BondType.SINGLE),
        BondDefinition("O", "H", BondType.SINGLE),
        BondDefinition("O", "O", BondType.SINGLE),
        # Double bonds
        BondDefinition("C", "O", BondType.DOUBLE),
        BondDefinition("C", "C", BondType.DOUBLE),
        BondDefinition("C", "S", BondType.DOUBLE),
        BondDefinition("C", "F", BondType.DOUBLE),
        BondDefinition("C", "N", BondType.DOUBLE),
        BondDefinition("S", "N", BondType.DOUBLE),
        BondDefinition("S", "S", BondType.DOUBLE),
        BondDefinition("N", "N", BondType.DOUBLE),
        BondDefinition("N", "O", BondType.DOUBLE),
        BondDefinition("S", "O", BondType.DOUBLE),
        # Triple bonds
        BondDefinition("C", "C", BondType.TRIPLE),
        BondDefinition("C", "N", BondType.TRIPLE),
        BondDefinition("N", "N", BondType.TRIPLE),
        # Aromatic bonds
        BondDefinition("C", "C", BondType.AROMATIC),
        BondDefinition("C", "N", BondType.AROMATIC),
        BondDefinition("C", "O", BondType.AROMATIC),
    ]

    def __init__(
        self, atom_list: List[str], bonds_list: List[List[int]], bonds_type: List[int]
    ):
        self.atom_list = atom_list
        self.bonds_list = bonds_list
        self.bonds_type = bonds_type

    def analyze_all_bonds(
        self,
    ) -> Tuple[Dict[str, List[List[int]]], Dict[str, List[int]]]:
        """Get dictionary for bond and bond_index"""
        bonds_dict = {}
        bond_indices_dict = {}
        for bond_def in self.BOND_DEFINITIONS:
            bond_pairs = self._find_specific_bonds(bond_def)
            key = bond_def.get_key()
            bonds_dict[key] = bond_pairs
            bond_indices_dict[key] = self._convert_to_bond_indices(bond_pairs)
        return bonds_dict, bond_indices_dict

    def _find_specific_bonds(self, bond_def: BondDefinition) -> List[List[int]]:
        """extract specific bond type indices from whole bonds_list"""
        self._validate_bond_definition(bond_def)
        bond_list: List[List[int]] = []
        target_pair = {bond_def.atom1, bond_def.atom2}
        is_same_atom = bond_def.atom1 == bond_def.atom2
        for bond, bond_type in zip(self.bonds_list, self.bonds_type):
            if bond_type != bond_def.bond_type.value:
                continue
            atom_pair = {self.atom_list[bond[0]], self.atom_list[bond[1]]}
            if atom_pair == target_pair and (is_same_atom or len(atom_pair) == 2):
                bond_list.append(bond)
        return bond_list

    def _validate_bond_definition(self, bond_def: BondDefinition) -> None:
        """validate bond info"""
        if (
            bond_def.atom1 not in self.ALLOWED_ATOMS
            or bond_def.atom2 not in self.ALLOWED_ATOMS
        ):
            raise ValueError(f"Invalid atom type: {bond_def.atom1}, {bond_def.atom2}")

    def _convert_to_bond_indices(self, bond_pairs: List[List[int]]) -> List[int]:
        """convert bonds_list to bond_index"""
        bond_indices = []
        for bond in bond_pairs:
            if bond in self.bonds_list:
                bond_indices.append(self.bonds_list.index(bond))
            elif bond[::-1] in self.bonds_list:
                bond_indices.append(self.bonds_list.index(bond[::-1]))
            else:
                logger.warning("Bond %s not found in bonds list", bond)
        return bond_indices


class AtomicIndexExtractor:
    """Class to extract atomic index"""

    ALLOWED_ATOMS = ["O", "N", "C", "H", "S", "F"]

    def __init__(self, atom_list: List[str]):
        self._atom_list = atom_list

    def extract_atomic_indices(self) -> Dict[str, List[int]]:
        """Extract atomic indices from atom_list"""
        indices = {}
        for atom_type in self.ALLOWED_ATOMS:
            indices[atom_type.lower() + "_list"] = [
                i for i, atom in enumerate(self._atom_list) if atom == atom_type
            ]
        return indices


class SpecialBondDetector:
    """Class to extract COC/COH bond"""

    def __init__(
        self,
        atom_list: List[str],
        bonds_list: List[List[int]],
        bonds_dict: Dict[str, List[List[int]]],
    ):
        self.atom_list = atom_list
        self.bonds_list = bonds_list
        self.bonds_dict = bonds_dict

    def detect_coc_coh_bonds(self) -> Tuple[List[List], List[List]]:
        """get COC/COH bonds"""
        o_indices = [i for i, atom in enumerate(self.atom_list) if atom == "O"]

        coc_indices = []
        coh_indices = []

        for o_num, o_index in enumerate(o_indices):
            neighbors = self._find_neighbors(o_index)

            if len(neighbors) != 2:
                continue

            neighbor_atoms = [neighbor[0] for neighbor in neighbors]

            if self._is_coh_pattern(neighbor_atoms):
                coh_data = self._build_coh_data(o_num, o_index, neighbors)
                coh_indices.append(coh_data)
            elif self._is_coc_pattern(neighbor_atoms):
                coc_data = self._build_coc_data(o_num, o_index, neighbors)
                coc_indices.append(coc_data)

        return coh_indices, coc_indices

    def _find_neighbors(self, atom_idx: int) -> List[Tuple[str, List[int]]]:
        """Get neighboring atoms"""
        neighbors = []
        for bond in self.bonds_list:
            if bond[0] == atom_idx:
                neighbor_atom = self.atom_list[bond[1]]
                neighbors.append((neighbor_atom, bond))
            elif bond[1] == atom_idx:
                neighbor_atom = self.atom_list[bond[0]]
                neighbors.append((neighbor_atom, bond))
        return neighbors

    def _is_coh_pattern(self, neighbor_atoms: List[str]) -> bool:
        """check COH"""
        return set(neighbor_atoms) == {"C", "H"}

    def _is_coc_pattern(self, neighbor_atoms: List[str]) -> bool:
        """check COC"""
        return neighbor_atoms == ["C", "C"]

    def _build_coh_data(self, o_num: int, o_index: int, neighbors: List) -> List:
        """build COH"""
        co_bonds = self.bonds_dict["CO_1_bond"]
        oh_bonds = self.bonds_dict["OH_1_bond"]

        co_index = oh_index = None
        for neighbor_atom, bond in neighbors:
            if neighbor_atom == "C":
                co_index = co_bonds.index(bond)
            else:  # "H"
                oh_index = oh_bonds.index(bond)

        return [o_num, o_index, {"CO": co_index, "OH": oh_index}]

    def _build_coc_data(self, o_num: int, o_index: int, neighbors: List) -> List:
        """build COC"""
        co_bonds = self.bonds_dict["CO_1_bond"]
        co1_index = co_bonds.index(neighbors[0][1])
        co2_index = co_bonds.index(neighbors[1][1])

        return [o_num, o_index, {"CO1": co1_index, "CO2": co2_index}]


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


def make_graph_from_itp(itp_data):
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
