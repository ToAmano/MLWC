# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
from ase import Atoms
from cpmd.pbc.pbc_mol import pbc_mol
from cpmd.bondcenter.bondcenter import calc_bondcenter
from dataset.methanol.dataset_aseatom_met import methanol_atoms, methanol_bond, methanol_pbc_atoms, methanol_bc

def test_bondcenter_met(methanol_atoms, methanol_bond, methanol_bc):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = methanol_bond

    list_mol_coords = pbc_mol.compute_pbc(methanol_atoms.get_positions(),
                                        np.array(methanol_atoms.get_cell()),
                                        bondlist, 
                                        NUM_MOL_PAR_MOL, 
                                        ref_atom_index)
    # calculate bond centers
    list_bond_centers = calc_bondcenter(list_mol_coords,bondlist)

    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(list_bond_centers.reshape(-1,3), methanol_bc, decimal=5)
