# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
from ase import Atoms
from cpmd.pbc.pbc_mol import pbc_mol
from dataset.methanol.dataset_aseatom_met import methanol_atoms, methanol_bond, methanol_pbc_atoms, methanol_bc


@pytest.fixture
def water_molecule():
    """水分子 (H2O) の Atoms オブジェクトを作成"""
    return Atoms(
        symbols="H2O",
        positions=[[5, 5, 5], [5, 5.9572, 5], [5.9266, 5-0.2396, 5]],  # 水分子の原子位置
        cell=[10, 10, 10],
        pbc=True
    )

@pytest.fixture
def water_molecule_bond():
    bondlist = [[0, 2], [1, 2]]
    NUM_MOL_PAR_MOL=3
    ref_atom_index=2
    return bondlist, NUM_MOL_PAR_MOL, ref_atom_index

def test_raw_get_distances_mic_without_mic(water_molecule,water_molecule_bond):
    # 最小画像規則なし (mic=False)
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = water_molecule_bond
    
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(water_molecule.get_positions(),
                                            np.array(water_molecule.get_cell()),
                                             bondlist,
                                             NUM_MOL_PAR_MOL,
                                             ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    expected_positions = water_molecule.get_positions()
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), expected_positions, decimal=5)

def test_raw_get_distances_mic_with_mic(water_molecule,water_molecule_bond):
    # 最小画像規則なし (mic=False)
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = water_molecule_bond
    
    # 実行
    result_atoms = pbc_mol.compute_pbc(water_molecule.get_positions(),
                                        np.array(water_molecule.get_cell()),
                                             bondlist,
                                             NUM_MOL_PAR_MOL,
                                             ref_atom_index)
    
    # 最小画像規則の適用で、原子の位置が変更されているか確認
    # 期待結果の例: 適切なテスト結果を入れる
    assert result_atoms is not None  # ここでは単純に結果が返されることを確認

def test_raw_get_distances_mic_invalid_ref_index(water_molecule,water_molecule_bond):
    # 不正な ref_atom_index のケース
    # 最小画像規則なし (mic=False)
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = water_molecule_bond
    ref_atom_index = 10  # 存在しないインデックス
    
    with pytest.raises(IndexError):
        pbc_mol.compute_pbc(water_molecule.get_positions(),
                            np.array(water_molecule.get_cell()),
                            bondlist,
                            NUM_MOL_PAR_MOL,
                            ref_atom_index)

def test_raw_get_distances_mic_empty_mol_inds(water_molecule,water_molecule_bond):
    # 最小画像規則なし (mic=False)
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = water_molecule_bond
    # different ref_atom_index
    ref_atom_index = 0
    
    # 結果が元の位置と一致するか確認
    result_vectors = pbc_mol.compute_pbc(water_molecule.get_positions(),
                                             np.array(water_molecule.get_cell()),
                                             bondlist,
                                             NUM_MOL_PAR_MOL,
                                             ref_atom_index)
    expected_positions = water_molecule.get_positions()
    np.testing.assert_almost_equal(result_vectors.reshape(-1,3), expected_positions, decimal=5)

def test_pbc_mol_met(methanol_atoms, methanol_bond, methanol_pbc_atoms):
    # 最小画像規則なし (mic=False)
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = methanol_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(methanol_atoms.get_positions(),
                                            np.array(methanol_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    expected_positions = methanol_pbc_atoms.get_positions()
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), expected_positions, decimal=5)
