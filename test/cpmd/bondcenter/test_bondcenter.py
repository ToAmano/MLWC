# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
import ase.io
from ase import Atoms
from cpmd.pbc.pbc_mol import pbc_mol
from cpmd.bondcenter.bondcenter import calc_bondcenter
from dataset.methanol.dataset_aseatom_met import methanol_atoms, methanol_bond, methanol_bc
from dataset.ethanol.dataset_aseatom_eth import ethanol_atoms, ethanol_bond, ethanol_bc
from dataset.pg2.dataset_aseatom_pg2     import pg2_atoms, pg2_bond, pg2_bc
from dataset.pg12.dataset_aseatom_pg12   import pg12_atoms, pg12_bond, pg12_bc

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


def test_bondcenter_eth(ethanol_atoms, ethanol_bond, ethanol_bc):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = ethanol_bond

    list_mol_coords = pbc_mol.compute_pbc(ethanol_atoms.get_positions(),
                                        np.array(ethanol_atoms.get_cell()),
                                        bondlist, 
                                        NUM_MOL_PAR_MOL, 
                                        ref_atom_index)
    # calculate bond centers
    list_bond_centers = calc_bondcenter(list_mol_coords,bondlist)
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(list_bond_centers.reshape(-1,3), ethanol_bc, decimal=5)


def test_bondcenter_pg2(pg2_atoms, pg2_bond, pg2_bc):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg2_bond

    list_mol_coords = pbc_mol.compute_pbc(pg2_atoms.get_positions(),
                                        np.array(pg2_atoms.get_cell()),
                                        bondlist, 
                                        NUM_MOL_PAR_MOL, 
                                        ref_atom_index)
    # calculate bond centers
    list_bond_centers = calc_bondcenter(list_mol_coords,bondlist)
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(list_bond_centers.reshape(-1,3), pg2_bc, decimal=5)


def test_bondcenter_pg12(pg12_atoms, pg12_bond, pg12_bc):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg12_bond
    
    list_mol_coords = pbc_mol.compute_pbc(pg12_atoms.get_positions(),
                                        np.array(pg12_atoms.get_cell()),
                                        bondlist, 
                                        NUM_MOL_PAR_MOL, 
                                        ref_atom_index)
    # calculate bond centers
    list_bond_centers = calc_bondcenter(list_mol_coords,bondlist)
    test = pg12_atoms
    test.set_positions(list_mol_coords.reshape(-1,3))
    ase.io.write("test.xyz",test)
    np.savetxt("test.txt", list_mol_coords.reshape(-1,3))
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(list_bond_centers.reshape(-1,3), pg12_bc, decimal=5)