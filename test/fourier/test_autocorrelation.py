import numpy as np
import pytest

from src.mlwc.fourier.autocorrelation import (
    autocorr,
    autocorr_numpy,
    autocorr_scipy,
    autocorr_statsmodels,
)

EXPECTED = np.array([2.0, 0.8, -0.2, -0.8, -0.8])


def test_autocorr_init():
    # autocorr クラスの初期化をテスト
    strategy = autocorr_numpy()
    instance = autocorr(strategy)
    assert instance.strategy == strategy


def test_autocorr_compute_autocorr1d():
    # autocorr クラスの compute_autocorr1d メソッドをテスト
    strategy = autocorr_numpy()
    instance = autocorr(strategy)
    x = np.array([1, 2, 3, 4, 5])
    result = instance.compute_autocorr1d(x=x)
    expected = autocorr_numpy.compute_autocorr1d(x=x)
    assert np.allclose(result, expected)


def test_autocorr_compute_autocorr2d():
    # autocorr クラスの compute_autocorr2d メソッドをテスト
    strategy = autocorr_numpy()
    instance = autocorr(strategy)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = instance.compute_autocorr2d(x=x)
    expected = autocorr_numpy.compute_autocorr2d(x=x)
    assert np.allclose(result, expected)


def test_autocorr_numpy_compute_autocorr1d_odd():
    # autocorr_numpy クラスの compute_autocorr1d メソッドをテスト
    x = np.array([1, 2, 3, 4, 5])
    result = autocorr_numpy.compute_autocorr1d(x=x)
    print(f"result = {result}")
    print(f"len(result) = {len(result)}")

    assert np.allclose(result, EXPECTED)
    assert len(result) == 5


def test_autocorr_numpy_compute_autocorr1d_even():
    # autocorr_numpy クラスの compute_autocorr1d メソッドをテスト
    x = np.array([1, 2, 3, 4, 5, 6])
    result = autocorr_numpy.compute_autocorr1d(x=x)
    expected = np.array(
        [2.91666667, 1.45833333, 0.16666667, -0.79166667, -1.25, -1.04166667]
    )
    assert np.allclose(result, expected)


def test_autocorr_numpy_compute_autocorr1d_normalize():
    # autocorr_numpy クラスの compute_autocorr1d メソッドをテスト (normalize=True)
    x = np.array([1, 2, 3, 4, 5])
    result = autocorr_numpy.compute_autocorr1d(x=x, normalize=True)
    assert np.allclose(result, EXPECTED / 2)


def test_autocorr_numpy_compute_autocorr1d_valueerror():
    # autocorr_numpy クラスの compute_autocorr1d メソッドをテスト (ValueError)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        autocorr_numpy.compute_autocorr1d(x=x)


def test_autocorr_numpy_compute_autocorr2d():
    """autocorr_numpy クラスの compute_autocorr2d メソッドをテスト
    2*3配列を入れると長さ2で返ってくる．
    """

    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = autocorr_numpy.compute_autocorr2d(x=x)
    print(f"result = {result}")
    print(f"len(result) = {len(result)}")
    expected = np.array([2.25, -1.125])

    assert np.allclose(result, expected)


def test_autocorr_numpy_compute_autocorr2d_ifmean_false():
    # autocorr_numpy クラスの compute_autocorr2d メソッドをテスト (ifmean=False)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = autocorr_numpy.compute_autocorr2d(x=x, ifmean=False)
    expected = np.array(
        [[-1.125, -1.125, -1.125], [2.25, 2.25, 2.25], [-1.125, -1.125, -1.125]]
    )
    print(f"result={result}")
    assert np.allclose(result, expected)


def test_autocorr_numpy_compute_autocorr2d_valueerror():
    # autocorr_numpy クラスの compute_autocorr2d メソッドをテスト (ValueError)
    x = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        autocorr_numpy.compute_autocorr2d(x=x)


def test_autocorr_scipy_compute_autocorr1d():
    # autocorr_scipy クラスの compute_autocorr1d メソッドをテスト
    x = np.array([1, 2, 3, 4, 5])
    result = autocorr_scipy.compute_autocorr1d(x=x)
    assert np.allclose(result, EXPECTED)


def test_autocorr_scipy_compute_autocorr1d_normalize():
    # autocorr_scipy クラスの compute_autocorr1d メソッドをテスト (normalize=True)
    x = np.array([1, 2, 3, 4, 5])
    result = autocorr_scipy.compute_autocorr1d(x=x, normalize=True)
    print(f"result = {result}")
    assert np.allclose(result, EXPECTED / 2)


def test_autocorr_scipy_compute_autocorr1d_valueerror():
    # autocorr_scipy クラスの compute_autocorr1d メソッドをテスト (ValueError)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        autocorr_scipy.compute_autocorr1d(x=x)


def test_autocorr_scipy_compute_autocorr2d():
    # autocorr_scipy クラスの compute_autocorr2d メソッドをテスト
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = autocorr_scipy.compute_autocorr2d(x=x)
    expected = np.array([2.25, -1.125])
    print(f"result = {result}")
    assert np.allclose(result, expected)


def test_autocorr_scipy_compute_autocorr2d_ifmean_false():
    # autocorr_scipy クラスの compute_autocorr2d メソッドをテスト (ifmean=False)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = autocorr_scipy.compute_autocorr2d(x=x, ifmean=False)
    expected = np.array(
        [[-1.125, -1.125, -1.125], [2.25, 2.25, 2.25], [-1.125, -1.125, -1.125]]
    )
    assert np.allclose(result, expected)


def test_autocorr_scipy_compute_autocorr2d_valueerror():
    # autocorr_scipy クラスの compute_autocorr2d メソッドをテスト (ValueError)
    x = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        autocorr_scipy.compute_autocorr2d(x=x)


def test_autocorr_statsmodels_compute_autocorr1d():
    # autocorr_statsmodels クラスの compute_autocorr1d メソッドをテスト
    x = np.array([1, 2, 3, 4, 5])
    result = autocorr_statsmodels.compute_autocorr1d(x=x)
    assert np.allclose(result, EXPECTED)


def test_autocorr_statsmodels_compute_autocorr1d_normalize():
    # autocorr_statsmodels クラスの compute_autocorr1d メソッドをテスト (normalize=True)
    x = np.array([1, 2, 3, 4, 5])
    result = autocorr_statsmodels.compute_autocorr1d(x=x, normalize=True)
    assert np.allclose(result, EXPECTED / 2)


def test_autocorr_statsmodels_compute_autocorr1d_valueerror():
    # autocorr_statsmodels クラスの compute_autocorr1d メソッドをテスト (ValueError)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        autocorr_statsmodels.compute_autocorr1d(x=x)


def test_autocorr_statsmodels_compute_autocorr2d():
    # autocorr_statsmodels クラスの compute_autocorr2d メソッドをテスト
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = autocorr_statsmodels.compute_autocorr2d(x=x)
    expected = np.array([2.25, -1.125])
    assert np.allclose(result, expected)


def test_autocorr_statsmodels_compute_autocorr2d_ifmean_false():
    # autocorr_statsmodels クラスの compute_autocorr2d メソッドをテスト (ifmean=False)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = autocorr_statsmodels.compute_autocorr2d(x=x, ifmean=False)
    expected = np.array([[2.25, 2.25, 2.25], [-1.125, -1.125, -1.125]])
    assert np.allclose(result, expected)


def test_autocorr_statsmodels_compute_autocorr2d_valueerror():
    # autocorr_statsmodels クラスの compute_autocorr2d メソッドをテスト (ValueError)
    x = np.array([1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        autocorr_statsmodels.compute_autocorr2d(x=x)


def test_three_autocorr1d():
    """3つのmethodが同じ値を返すことを確認"""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    x_statsmodels = autocorr_statsmodels.compute_autocorr1d(x=x)
    x_scipy = autocorr_scipy.compute_autocorr1d(x=x)
    x_numpy = autocorr_numpy.compute_autocorr1d(x=x)
    assert np.allclose(x_numpy, x_scipy)
    assert np.allclose(x_numpy, x_statsmodels)


def test_three_autocorr2d():
    """3つのmethodが同じ値を返すことを確認"""
    # even
    x = np.random.rand(20, 2)
    x_statsmodels = autocorr_statsmodels.compute_autocorr2d(x=x)
    x_scipy = autocorr_scipy.compute_autocorr2d(x=x)
    x_numpy = autocorr_numpy.compute_autocorr2d(x=x)
    assert np.allclose(x_numpy, x_scipy)
    assert np.allclose(x_numpy, x_statsmodels)
    # odd
    x = np.random.rand(21, 2)
    x_statsmodels = autocorr_statsmodels.compute_autocorr2d(x=x)
    x_scipy = autocorr_scipy.compute_autocorr2d(x=x)
    x_numpy = autocorr_numpy.compute_autocorr2d(x=x)
    assert np.allclose(x_numpy, x_scipy)
    assert np.allclose(x_numpy, x_statsmodels)
