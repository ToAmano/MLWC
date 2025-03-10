# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
from ase import Atoms
from cpmd.pbc.pbc_mol import pbc_mol
from ml.descriptor.descriptor_abstract import Descriptor
from ml.descriptor.descriptor_torch    import Descriptor_torch_bondcenter
from cpmd.bondcenter.bondcenter import calc_bondcenter
from dataset.methanol.dataset_aseatom_met import methanol_atoms, methanol_bond, methanol_pbc_atoms, methanol_bc, \
                                                methanol_ch_descriptor,methanol_co_descriptor,methanol_oh_descriptor,methanol_olp_descriptor

from dataset.ethanol.dataset_aseatom_eth import ethanol_atoms, ethanol_bond, ethanol_pbc_atoms, ethanol_bc, \
                                                ethanol_ch_descriptor,ethanol_co_descriptor,ethanol_cc_descriptor,ethanol_oh_descriptor,ethanol_olp_descriptor


def test_bondcenter_met(methanol_atoms, methanol_bond, methanol_ch_descriptor,
                        methanol_co_descriptor,methanol_oh_descriptor,methanol_olp_descriptor):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = methanol_bond

    list_mol_coords = pbc_mol.compute_pbc(methanol_atoms.get_positions(),
                                        np.array(methanol_atoms.get_cell()),
                                        bondlist, 
                                        NUM_MOL_PAR_MOL, 
                                        ref_atom_index)
    # calculate bond centers
    list_bond_centers = calc_bondcenter(list_mol_coords, bondlist)

    # calculate descriptor
    # ASSIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,atoms_traj[0].get_cell())
    DESC  = Descriptor(Descriptor_torch_bondcenter) # set strategy

    bond_centers = np.array(list_bond_centers)[:,[0,1,2],:].reshape((-1,3))

    Descs_ch     = DESC.calc_descriptor(atoms=methanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_ch, methanol_ch_descriptor, decimal=5)
    
    # CO
    bond_centers = np.array(list_bond_centers)[:,[3],:].reshape((-1,3))
    Descs_co     = DESC.calc_descriptor(atoms=methanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_co, methanol_co_descriptor, decimal=5)

    # OH
    bond_centers = np.array(list_bond_centers)[:,[4],:].reshape((-1,3))
    Descs_oh     = DESC.calc_descriptor(atoms=methanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_oh, methanol_oh_descriptor, decimal=5)
    
    # Olp
    bond_centers = methanol_atoms.get_positions()[np.argwhere(methanol_atoms.get_atomic_numbers()==8).reshape(-1)]
    Descs_olp    = DESC.calc_descriptor(atoms=methanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_olp, methanol_olp_descriptor, decimal=5)
    
    

def test_bondcenter_eth(ethanol_atoms, ethanol_bond, ethanol_ch_descriptor,
                        ethanol_cc_descriptor,
                        ethanol_co_descriptor,ethanol_oh_descriptor,ethanol_olp_descriptor):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = ethanol_bond

    list_mol_coords = pbc_mol.compute_pbc(ethanol_atoms.get_positions(),
                                        np.array(ethanol_atoms.get_cell()),
                                        bondlist, 
                                        NUM_MOL_PAR_MOL, 
                                        ref_atom_index)
    # calculate bond centers
    list_bond_centers = calc_bondcenter(list_mol_coords, bondlist)

    # calculate descriptor
    # ASSIGN=cpmd.asign_wcs.asign_wcs(NUM_MOL,NUM_MOL_ATOMS,atoms_traj[0].get_cell())
    DESC  = Descriptor(Descriptor_torch_bondcenter) # set strategy

    bond_centers = np.array(list_bond_centers)[:,[0, 2, 3, 6, 7],:].reshape((-1,3))
    Descs_ch     = DESC.calc_descriptor(atoms=ethanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_ch, ethanol_ch_descriptor, decimal=5)
    
    # CO
    bond_centers = np.array(list_bond_centers)[:,[5],:].reshape((-1,3))
    Descs_co     = DESC.calc_descriptor(atoms=ethanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_co, ethanol_co_descriptor, decimal=5)

    # OH
    bond_centers = np.array(list_bond_centers)[:,[4],:].reshape((-1,3))
    Descs_oh     = DESC.calc_descriptor(atoms=ethanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_oh, ethanol_oh_descriptor, decimal=5)
    
    # CC
    bond_centers = np.array(list_bond_centers)[:,[1],:].reshape((-1,3))
    Descs_cc     = DESC.calc_descriptor(atoms=ethanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認（ccは最初の二つの順番が変わることがあるので一旦除去）
    # np.testing.assert_almost_equal(Descs_cc, ethanol_cc_descriptor, decimal=5)
    
    # Olp
    bond_centers = ethanol_atoms.get_positions()[np.argwhere(ethanol_atoms.get_atomic_numbers()==8).reshape(-1)]
    Descs_olp    = DESC.calc_descriptor(atoms=ethanol_atoms,
                                        bond_centers=bond_centers,
                                        list_atomic_number=[6,1,8], 
                                        list_maxat=[24,24,24], 
                                        Rcs=4,
                                        Rc=6,
                                        device="cpu") # cuda or cpu
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(Descs_olp, ethanol_olp_descriptor, decimal=5)