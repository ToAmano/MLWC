"""Atomtype extractor for .mol files using RDKit"""

from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem

from mlwc.bond.atomtype import (
    AtomicIndexExtractor,
    BondAnalyzer,
    MolecularInfo,
    SpecialBondDetector,
)
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


class MolecularDataExtractor:
    """Class to extract data from rdkit.mol"""

    def __init__(self, mol_rdkit: Chem.Mol):
        self._mol_rdkit = mol_rdkit

    def extract_basic_info(
        self,
    ) -> Tuple[int, List[str], List[List[int]], List[int], int]:
        """extract data from rdkit.chem.mol"""
        num_atoms_per_mol: int = self._mol_rdkit.GetNumAtoms()
        atom_list: List[str] = [atom.GetSymbol() for atom in self._mol_rdkit.GetAtoms()]
        bonds_list = [
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for bond in self._mol_rdkit.GetBonds()
        ]
        bonds_type: List[int] = self._extract_bond_types()
        representative_atom_index = self._find_representative_atom_index()

        return (
            num_atoms_per_mol,
            atom_list,
            bonds_list,
            bonds_type,
            representative_atom_index,
        )

    def _extract_bond_types(self) -> List[int]:
        """extract bonding type from chem.mol"""
        bond_type_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 10,
        }
        return [
            bond_type_map.get(bond.GetBondType(), 1)
            for bond in self._mol_rdkit.GetBonds()
        ]

    def _find_representative_atom_index(self) -> int:
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
        positions_skelton: List[np.ndarray] = []
        index_tmp: List[int] = []
        logger.info(" ===================== ")
        logger.info("  Atomic coordinates ")
        for i, atom in enumerate(self._mol_rdkit.GetAtoms()):
            pos = self._mol_rdkit.GetConformer().GetAtomPosition(i)
            if atom.GetSymbol() == "H":  # use only non-hydrogen atoms
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


class ReadMolFile:
    """Factory class to generate MolecularInfo from .mol files"""

    def __init__(self, filename: str):
        self.molecular_info = self._create_molecular_info(filename)

    def _create_molecular_info(self, filename: str) -> MolecularInfo:
        """generate MolecularInfo"""
        # 1. Read .mol using RDKit
        mol_rdkit = Chem.MolFromMolFile(filename, sanitize=True, removeHs=False)

        # 2. Extract basic information from the molecule
        extractor = MolecularDataExtractor(mol_rdkit)
        num_atoms, atom_list, bonds_list, bonds_type, representative_atom_index = (
            extractor.extract_basic_info()
        )

        # 3. Bond analysis
        bond_analyzer = BondAnalyzer(atom_list, bonds_list, bonds_type)
        bonds_dict, bond_indices_dict = bond_analyzer.analyze_all_bonds()

        # 4. Atomic index extraction
        atomic_extractor = AtomicIndexExtractor(atom_list)
        atomic_indices: Dict[str, List[int]] = atomic_extractor.extract_atomic_indices()

        # 5. COC/COH bond detection
        special_detector = SpecialBondDetector(atom_list, bonds_list, bonds_dict)
        coh_indices, coc_indices = special_detector.detect_coc_coh_bonds()

        # 6. MolecularInfo
        molecular_info = MolecularInfo(
            mol_rdkit=mol_rdkit,
            num_atoms_per_mol=num_atoms,
            atom_list=atom_list,
            bonds_list=bonds_list,
            num_bonds=len(bonds_list),
            bonds_type=bonds_type,
            representative_atom_index=representative_atom_index,
            bonds=bonds_dict,
            bond_index=bond_indices_dict,
            atomic_index=atomic_indices,
            coh_index=coh_indices,
            coc_index=coc_indices,
        )

        self._log_results(molecular_info)
        return molecular_info

    def _log_results(self, molecular_info: MolecularInfo) -> None:
        """Output results"""
        logger.info("=== ReadMolFile Analysis Results ===")
        logger.info("Number of atoms: %s", molecular_info.num_atoms_per_mol)
        logger.info("Atom list: %s", molecular_info.atom_list)
        logger.info("Number of bonds: %s", molecular_info.num_bonds)
        logger.info("COH indices: %s", molecular_info.coh_index)
        logger.info("COC indices: %s", molecular_info.coc_index)

        logger.info("================ Bond Analysis ================")
        for key, value in molecular_info.bonds.items():
            if value:
                logger.info("%s: %s", key, value)

        logger.info("========== Atomic Indices ==========")
        for key, value in molecular_info.atomic_index.items():
            if value:
                logger.info("%s atoms: %s", key, value)

    def get_molecular_info(self) -> MolecularInfo:
        """MolecularInfo"""
        return self.molecular_info


def create_molecular_info(filename: str) -> MolecularInfo:
    """Factory to generate MolecularInfo instance"""
    reader = ReadMolFile(filename)
    return reader.get_molecular_info()
