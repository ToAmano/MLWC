# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
from ase import Atoms
from cpmd.pbc.pbc_mol import pbc_mol
from dataset.methanol.dataset_aseatom_met import methanol_atoms, methanol_bond, methanol_pbc_atoms, methanol_bc
from dataset.ethanol.dataset_aseatom_eth  import ethanol_atoms, ethanol_bond, ethanol_pbc_atoms, ethanol_bc
from dataset.pg.dataset_aseatom_pg        import pg_atoms, pg_bond, pg_pbc_atoms, pg_bc
from dataset.pg2.dataset_aseatom_pg2      import pg2_atoms, pg2_bond, pg2_pbc_atoms, pg2_bc
from dataset.pg12.dataset_aseatom_pg12    import pg12_atoms, pg12_1mol_atoms, pg12_bond, pg12_pbc_atoms, pg12_bc
from dataset.pmma.dataset_aseatom_pmma    import pmma_atoms, pmma_bond, pmma_pbc_atoms, pmma_bc


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


def test_pbc_mol_met(ethanol_atoms, ethanol_bond, ethanol_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = ethanol_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(ethanol_atoms.get_positions(),
                                            np.array(ethanol_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), ethanol_pbc_atoms, decimal=5)


def test_pbc_mol_pg(pg_atoms, pg_bond, pg_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(pg_atoms.get_positions(),
                                            np.array(pg_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pg_pbc_atoms, decimal=5)


def test_pbc_mol_pg2(pg2_atoms, pg2_bond, pg2_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg2_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(pg2_atoms.get_positions(),
                                            np.array(pg2_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pg2_pbc_atoms, decimal=5)
    
    
def test_pbc_mol_pg12(pg12_atoms, pg12_bond, pg12_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg12_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(pg12_atoms.get_positions(),
                                            np.array(pg12_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pg12_pbc_atoms, decimal=5)
    

def test_pbc_mol_pg12_1mol(pg12_1mol_atoms,pg12_bond,pg12_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg12_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(pg12_1mol_atoms.get_positions(),
                                            np.array(pg12_1mol_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pg12_pbc_atoms[123:2*123], decimal=5)
    
def test_pbc_mol_pg12_1mol_2(pg12_atoms,pg12_bond,pg12_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg12_bond
    for mol_id in range(10):
        # mol_id = 0
        vectors = pg12_atoms.get_positions()[mol_id*123:(mol_id+1)*123]
        # 実行
        calculated_vectors = pbc_mol.compute_pbc(vectors,
                                                np.array(pg12_atoms.get_cell()),
                                                bondlist,
                                                NUM_MOL_PAR_MOL,
                                                ref_atom_index) 
        # 結果の検証: 移動後の位置が元の位置と一致することを確認
        np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pg12_pbc_atoms[mol_id*123:(mol_id+1)*123], decimal=5)

def test_pbc_mol_pg12_1mol_3(pg12_atoms,pg12_bond,pg12_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pg12_bond
    for mol_id in range(1):
        mol_id = 6
        vectors = pg12_atoms.get_positions()[mol_id*123:(mol_id+3)*123]
        # 実行
        calculated_vectors = pbc_mol.compute_pbc(vectors,
                                                np.array(pg12_atoms.get_cell()),
                                                bondlist,
                                                NUM_MOL_PAR_MOL,
                                                ref_atom_index) 
        # 結果の検証: 移動後の位置が元の位置と一致することを確認
        np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pg12_pbc_atoms[mol_id*123:(mol_id+3)*123], decimal=5)
        
def test_pbc_mol_pmma(pmma_atoms, pmma_bond, pmma_pbc_atoms):
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index = pmma_bond
    # 実行
    calculated_vectors = pbc_mol.compute_pbc(pmma_atoms.get_positions(),
                                            np.array(pmma_atoms.get_cell()),
                                            bondlist,
                                            NUM_MOL_PAR_MOL,
                                            ref_atom_index) 
    
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(calculated_vectors.reshape(-1,3), pmma_pbc_atoms, decimal=5)