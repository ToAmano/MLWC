import ase.io
import numpy as np
import pytest
from ase import Atoms

from mlwc.cpmd.asign_wcs import raw_find_all_lonepairs
from mlwc.cpmd.assign_wcs.assign_wcs_torch import (
    _assign_dipoles,
    _calculate_bonddipole,
    _calculate_comwcs,
    _calculate_nearest_wfc,
    _sort_wfc_index,
    atoms_wan,
    calculate_atomiccoord_pbcmol,
    convert_atoms_to_bondwfc,
    extract_wcs,
)
from mlwc.cpmd.bondcenter.bondcenter import calc_bondcenter
from mlwc.dataset.methanol.dataset_aseatom_met import (
    methanol_atoms,
    methanol_atoms_X,
    methanol_bc,
    methanol_bond,
    methanol_ch_truey,
    methanol_co_truey,
    methanol_itpdata,
    methanol_oh_truey,
    methanol_olp_truey,
)
from mlwc.include.constants import Constant

coef = Constant.Ang * Constant.Charge / Constant.Debye


def test_extract_wcs():
    # テスト用の Atoms オブジェクトを作成
    atoms = Atoms("H2X", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])

    # extract_wcs 関数を実行
    atoms_nowan, wfc_list = extract_wcs(atoms)

    # 結果を検証
    assert isinstance(atoms_nowan, Atoms)
    assert len(atoms_nowan) == 2
    assert np.allclose(atoms_nowan.get_positions(), [[0, 0, 0], [0, 0, 1]])
    assert np.allclose(wfc_list, [[0, 0, 2]])


# def test_atoms_wan():
#     # テスト用の Atoms オブジェクトを作成
#     atoms = Atoms('H2X', positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
#     atoms_nowan, wfc_list = extract_wcs(atoms)

#     # atoms_wan オブジェクトを作成
#     atoms_wan_instance = atoms_wan(atoms)
#     atoms_wan_instance.set_params(atoms_nowan, wfc_list)

#     # 結果を検証
#     assert isinstance(atoms_wan_instance.atoms, Atoms)
#     assert isinstance(atoms_wan_instance.atoms_nowan, Atoms)
#     assert np.allclose(atoms_wan_instance.atoms_nowan.get_positions(), [
#                        [0, 0, 0], [0, 0, 1]])
#     assert np.allclose(atoms_wan_instance.wfc_list, [[0, 0, 2]])


def test_calculate_molcoord():
    # テスト用の Atoms オブジェクトを作成
    atoms = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]], cell=[10, 10, 10])
    bonds_list = [[0, 1], [0, 2]]
    ref_atom_index = 0

    # calculate_molcoord 関数を実行
    atomic_positions = calculate_atomiccoord_pbcmol(atoms, bonds_list, ref_atom_index)

    # 結果を検証
    assert np.allclose(atomic_positions, [[[0, 0, 0], [0, 0, 1], [0, 0, 2]]])


def test_find_nearest_wfc():
    # テスト用の bondcenters, wfc_list, UNITCELL_VECTORS オブジェクトを作成
    bondcenters = np.array([[0, 0, 0], [1, 1, 1]])
    wfc_list = np.array([[0, 0, 0.5], [1, 1, 0], [0, 1, 0]])
    UNITCELL_VECTORS = np.eye(3) * 10

    # find_nearest_wfc 関数を実行
    nearest_indices = _sort_wfc_index(bondcenters, wfc_list, UNITCELL_VECTORS)
    # 結果を検証
    assert np.allclose(nearest_indices, [[0, 2, 1], [1, 2, 0]])


class MockItpData:
    def __init__(self):
        self.num_bonds = 2
        self.bonds_type = [1, 1]


def test_calculate_nearest_wfc():
    # テスト用の nearest_indices, itp_data オブジェクトを作成
    nearest_indices = np.array([[0, 1], [1, 0]])
    # calculate_nearest_wfc 関数を実行
    indices = _calculate_nearest_wfc(nearest_indices, num_wcs=1)

    # 結果を検証
    assert np.allclose(indices, [np.array([0]), np.array([1])])


def test_calculate_comwfs():
    # テスト用の indices, wfc_list オブジェクトを作成
    indices = [[0, 1], [2, 3]]
    wfc_list = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    unitcell = 2 * np.eye(3)

    # calculte_comwfs 関数を実行
    nearest_comwfc = _calculate_comwcs(indices, wfc_list, unitcell)

    # 結果を検証
    assert np.allclose(nearest_comwfc, [[0, 0, 0.5], [0, 1, 0.5]])


def test_calculate_comwfs_mic():
    # テスト用の indices, wfc_list オブジェクトを作成
    indices = [[0, 1], [2, 3]]
    wfc_list = np.array([[0, 0, 0.01], [0, 0, 0.98], [0, 1, 0.02], [0, 1, 0.99]])
    unitcell = 1 * np.eye(3)

    # calculte_comwfs 関数を実行
    nearest_comwfc = _calculate_comwcs(indices, wfc_list, unitcell)

    # 結果を検証
    assert np.allclose(nearest_comwfc, [[0, 0, -0.005], [0, 1, 0.005]])


def test_calculate_bondwfs():
    # テスト用の bondcenters, wfc_list, UNITCELL_VECTORS, itp_data オブジェクトを作成
    print(" // test_calculate_bondwfs //")
    bondcenters = np.array([[[0, 0, 0], [1, 1, 1]]])
    lonepairs = np.array([[[2, 2, 2]]])
    nlonepairs = np.array([[[4, 4, 4]]])
    wfc_list = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0.5], [5, 5, 5], [2.1, 2.1, 2.1]])
    atoms = ase.Atoms(
        [8, 7],
        positions=np.array([[2, 2, 2], [4, 4, 4]]),
        cell=10 * np.eye(3),
        pbc=[1, 1, 1],
    )
    # calculate_bondwfs 関数を実行
    list_bond_mu, list_lp_mu = _calculate_bonddipole(
        bondcenters,
        atoms,
        wfc_list,
        bonds_type=[1, 1],
    )

    # 結果を検証
    assert np.allclose(list_bond_mu, [[0, 0, -2 * coef], [0, 0, 2 * coef]])


def test_bondmu_met(
    methanol_itpdata,
    methanol_bond,
    methanol_atoms_X,
    methanol_olp_truey,
    methanol_co_truey,
    methanol_ch_truey,
    methanol_oh_truey,
):
    print(" // test_bondmu_test //")
    bondlist, NUM_MOL_PAR_MOL, ref_atom_index, bonds = methanol_bond
    bonds_type = [1, 1, 1, 1, 1]  # all are the single bonds

    atoms_nowan, wfc_list = extract_wcs(methanol_atoms_X)
    molcoord = calculate_atomiccoord_pbcmol(atoms_nowan, bondlist, ref_atom_index)

    bcs_coord = calc_bondcenter(molcoord, bondlist)

    unitcell = atoms_nowan.get_cell()
    nearest_indices = _sort_wfc_index(bcs_coord, wfc_list, unitcell)
    extracted_bcs_indices = _calculate_nearest_wfc(
        nearest_indices, np.repeat(bonds_type, 32)
    )  # 32: num_mol
    nearest_wfcs = _calculate_comwcs(extracted_bcs_indices, wfc_list, unitcell)

    list_bond_mu, list_lp_mu = _calculate_bonddipole(
        bcs_coord,
        atoms_nowan,
        wfc_list,
        bonds_type=bonds_type,
    )
    # 結果の検証: 移動後の位置が元の位置と一致することを確認
    np.testing.assert_almost_equal(
        list_lp_mu["Olp"].reshape(-1, 3), methanol_olp_truey, decimal=5
    )

    dict_mu = convert_atoms_to_bondwfc(
        atoms_nowan,
        wfc_list,
        bonds_type,
        bondlist,
        methanol_itpdata.bond_index,
        methanol_itpdata.representative_atom_index,
    )
    try:
        np.testing.assert_almost_equal(
            dict_mu["OH_1_bond"].reshape(-1, 3), methanol_oh_truey, decimal=5
        )
    except AssertionError as e:
        # どのインデックスが異なるかを表示
        diff_indices = np.where(
            np.abs(dict_mu["OH_1_bond"].reshape(-1, 3) - methanol_oh_truey) > 10**-2
        )
        diff_values = [
            (idx, dict_mu["OH_1_bond"].reshape(-1, 3)[idx], methanol_oh_truey[idx])
            for idx in zip(*diff_indices)
        ]
        print("Assertion failed! Differences found at:")
        for idx, val_a, val_b in diff_values:
            print(f"  Index {idx}: Expected {val_b}, but got {val_a}")
        # pytest の assert に失敗させる
        pytest.fail(
            f"Arrays are not almost equal. {len(diff_values)} differences found."
        )
    #
    try:
        np.testing.assert_almost_equal(
            dict_mu["CO_1_bond"].reshape(-1, 3), methanol_co_truey, decimal=5
        )
    except AssertionError as e:
        # どのインデックスが異なるかを表示
        diff_indices = np.where(
            np.abs(dict_mu["CO_1_bond"].reshape(-1, 3) - methanol_co_truey) > 10**-5
        )
        diff_values = [
            (idx, dict_mu["CO_1_bond"].reshape(-1, 3)[idx], methanol_co_truey[idx])
            for idx in zip(*diff_indices)
        ]
        print("Assertion failed! Differences found at:")
        for idx, val_a, val_b in diff_values:
            print(f"  Index {idx}: Expected {val_b}, but got {val_a}")
        # pytest の assert に失敗させる
        pytest.fail(
            f"Arrays are not almost equal. {len(diff_values)} differences found."
        )
    np.testing.assert_almost_equal(
        dict_mu["CH_1_bond"].reshape(-1, 3), methanol_ch_truey, decimal=5
    )
    np.testing.assert_almost_equal(
        dict_mu["Olp"].reshape(-1, 3), methanol_olp_truey, decimal=5
    )


def test_caculate_bond_met(methanol_itpdata):
    """check"""
    atoms: Atoms = ase.io.read("testframe_met.xyz")

    [atoms_nowan, wfc_list] = extract_wcs(atoms)  # atoms, X
    mol_coords = calculate_atomiccoord_pbcmol(
        atoms_nowan,
        methanol_itpdata.bonds_list,
        methanol_itpdata.representative_atom_index,
    )

    atomic_positions = atoms.get_positions()
    atomic_symbols = np.array(atoms.get_chemical_symbols())
    unitcell_vectors = atoms.get_cell()
    num_mols = 32
    olp_coords = atomic_positions[atomic_symbols == "O"].reshape(num_mols, -1, 3)
    distance_olp_wcs, indices_olp = _assign_dipoles(
        olp_coords, wfc_list, unitcell_vectors, num_wcs=2
    )
    list_olp_mu = -4.0 * coef * distance_olp_wcs
    if len(np.ravel(list_olp_mu)) > 0:
        if np.max(np.linalg.norm(list_olp_mu, axis=2)) > 10:
            print(
                "Olp_mu is very large %s", np.max(np.linalg.norm(list_olp_mu, axis=2))
            )

    # calculate from reference implementation
    # Oローンペア
    list_atomic_nums = list(
        np.array(atoms_nowan.get_atomic_numbers()).reshape(num_mols, -1)
    )
    atO_list = [np.argwhere(js == 8).reshape(-1) for js in list_atomic_nums]
    list_mu_lpO, list_lpO_wfcs, picked_wcs_O = raw_find_all_lonepairs(
        wfc_list, atO_list, mol_coords, [], 2, unitcell_vectors
    )
    np.testing.assert_almost_equal(list_mu_lpO, list_olp_mu, decimal=5)
