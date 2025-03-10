# Test functions for src/cpmd/pbc/pbc_mol.py
import pytest
import numpy as np
from cpmd.pbc.pbc_numpy import pbc_2d, pbc_1d


def test_compute_pbc_1d_1():
    # 正常な入力データ
    vectors_array = np.array([5.5, 8.5, -11.0])
    cell = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0]
    ])

    # 正解データ（手計算や別途確認）
    expected_output = np.array([-0.5, -0.5, 1.0])

    # 実行して結果を比較
    result = pbc_1d.compute_pbc(vectors_array, cell)
    np.testing.assert_almost_equal(result, expected_output, decimal=5)


def test_compute_pbc_1d_2():
    # 正常な入力データ
    vectors_array = np.array([5.8, 8.9, -11.4])
    cell = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0]
    ])

    # 正解データ（手計算や別途確認）
    expected_output = np.array([1.8, 0.9, 0.6])

    # 実行して結果を比較
    result = pbc_1d.compute_pbc(vectors_array, cell)
    np.testing.assert_almost_equal(result, expected_output, decimal=5)


def test_compute_pbc_1d_3():
    # 正常な入力データ
    vectors_array = np.array([26.8488305 , 18.26883029, 12.7801397])
    cell = np.array([
        [22.782507693241886, 0.0, 0.0],
        [0.0, 22.782507693241886, 0.0],
        [0.0, 0.0, 22.782507693241886]
    ])

    # 正解データ（手計算や別途確認）
    # expected_output = np.array([ 4.06632281, 18.26883029, 12.7801397 ])
    expected_output = np.array([ 4.06632281, -4.5136774 , -10.00236799])

    # 実行して結果を比較
    result = pbc_1d.compute_pbc(vectors_array, cell)
    np.testing.assert_almost_equal(result, expected_output, decimal=5)



def test_compute_pbc_2d():
    # 正常な入力データ
    vectors_array = np.array([
        [1.5, 2.5, -3.0],
        [-1.5, 4.0, 2.0],
        [0.0, 0.0, 0.0]
    ])
    cell = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0]
    ])

    # 正解データ（手計算や別途確認）
    expected_output = np.array([
        [-0.5, -0.5, 1.0],
        [0.5, 1.0, 2.0],
        [0.0, 0.0, 0.0]
    ])

    # 実行して結果を比較
    result = pbc_2d.compute_pbc(vectors_array, cell)
    np.testing.assert_almost_equal(result, expected_output, decimal=5)

def test_compute_pbc_invalid_vectors_array_shape():
    # vectors_array の形状が不正な場合
    invalid_vectors_array = np.array([[1.0, 2.0]])  # shape (1, 2)
    cell = np.eye(3)  # 単位行列

    with pytest.raises(ValueError, match="Invalid shape for vectors_array"):
        pbc_2d.compute_pbc(invalid_vectors_array, cell)

def test_compute_pbc_invalid_cell_shape():
    # cell の形状が不正な場合
    vectors_array = np.array([[1.0, 2.0, 3.0]])
    invalid_cell = np.array([[2.0, 0.0]])  # shape (1, 2)

    with pytest.raises(ValueError):  # np.linalg.inv でエラーが出る可能性
        pbc_2d.compute_pbc(vectors_array, invalid_cell)