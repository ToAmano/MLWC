"""
This module provides classes for computing 1D autocorrelation using different numerical libraries.
It includes implementations using NumPy, SciPy, and Statsmodels.

File Description:
This file defines abstract and concrete classes for computing the 1D and 2D autocorrelation of a given array.
The autocorrelation is computed using different numerical libraries such as NumPy, SciPy, and Statsmodels.
The module follows the strategy pattern, where the `autocorr1d` class serves as the abstract base class
and the `autocorr1d_numpy`, `autocorr1d_scipy`, and `autocorr1d_statsmodels` classes provide concrete implementations.

This module provides classes for computing 1D autocorrelation using different numerical libraries.
It includes implementations using NumPy, SciPy, and Statsmodels.

Autocorrelation for 
- velocity
- dipole moment

NumPy, SciPy, and Statsmodels have slightly different implementations of auto-correlation.

## References

- https://elcorto.github.io/pwtools/_modules/pwtools/signal.html#acorr

"""
import abc
import statsmodels.api as sm
import numpy as np
import scipy
from include.mlwc_logger import setup_cmdline_logger

logger = setup_cmdline_logger(__name__)


class autocorr_abstract(abc.ABC):
    """
    Abstract base class for computing 1D autocorrelation.

    This class defines the interface for concrete strategy classes that implement different
    algorithms for computing 1D autocorrelation.
    """
    @classmethod
    @abc.abstractmethod
    def compute_autocorr1d(cls):
        """
        Abstract method to compute the 1D autocorrelation. This method should be implemented by concrete strategy classes.
        """
        pass


class autocorr():
    """
    ConcreteStrategy をインスタンス変数として持つクラス
    """

    def __init__(self, strategy):  # pbc_abstract
        self.strategy = strategy

    def compute_autocorr1d(self, **kwargs):
        # Call ConcreteStrategy method to consignment processing
        return self.strategy.compute_autocorr1d(**kwargs)

    def compute_autocorr2d(self, **kwargs):
        # Call ConcreteStrategy method to consignment processing
        return self.strategy.compute_autocorr2d(**kwargs)


class autocorr_numpy(autocorr_abstract):
    """
    Concrete strategy class for computing 1D autocorrelation using NumPy.
    """
    @classmethod
    def compute_autocorr1d(x: np.ndarray, normalize: bool = False) -> np.ndarray:  # x is 1D array
        """
        Compute the 1D autocorrelation of a NumPy array.

        Parameters
        ----------
        x : np.ndarray
            A 1D NumPy array.

        Returns
        -------
        np.ndarray
            The 1D autocorrelation of the input array.

        Raises
        ------
        ValueError
            If the input array is not 1D.

        Examples
        --------
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> autocorr1d_numpy.compute_autocorr1d(arr)
        array([15, 22, 28, 33, 35])
        """
        if len(x.shape) != 1:
            raise ValueError("Only 1D array is supported")
        # note:: numpy correlate do not support fft
        result = np.correlate(x, x, mode='full')[int(result.size/2):]
        if normalize:
            result = result/result[0]
        return result

    def compute_autocorr2d(x: np.ndarray, ifmean: bool = True) -> np.ndarray:
        if len(x.shape) != 2:
            raise ValueError("Only 2D array is supported")
        auto_correlations: np.ndarray = np.apply_along_axis(lambda x: np.correlate(
            x, x, mode='full'), axis=0, arr=x)
        if ifmean:
            # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
            mean_autocorrelation: np.ndarray = np.mean(
                auto_correlations, axis=1)[len(x)-1:]  # acf
            return mean_autocorrelation
        else:
            return auto_correlations


class autocorr_scipy(autocorr_abstract):
    """
    Concrete strategy class for computing 1D autocorrelation using SciPy.
    """
    @classmethod
    def compute_autocorr1d(cls, x: np.ndarray, normalize: bool = False) -> np.ndarray:  # x is 1D array
        """
        Compute the 1D autocorrelation of a NumPy array using SciPy.

        Parameters
        ----------
        x : np.ndarray
            A 1D NumPy array.

        Returns
        -------
        np.ndarray
            The 1D autocorrelation of the input array.

        Raises
        ------
        ValueError
            If the input array is not 1D.

        Examples
        --------
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> autocorr1d_scipy.autocorr1d_scipy(arr)
        array([ 5. ,  7. ,  9. , 11. , 11.5])
        """
        if len(x.shape) != 1:
            raise ValueError("Only 1D array is supported")
        # /len(x) # !! do not normalize
        result = scipy.signal.correlate(x, x, mode="same", method="fft")[
            int(result.size/2):]
        if normalize:
            result = result/result[0]
        return result

    def compute_autocorr2d(cls, x: np.ndarray, ifmean: bool = True) -> np.ndarray:
        if len(x.shape) != 2:
            raise ValueError("Only 2D array is supported")
        auto_correlations: np.ndarray = np.apply_along_axis(lambda x: scipy.signal.correlate(
            x, x, mode='full'), axis=0, arr=x)
        if ifmean:
            # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
            mean_autocorrelation: np.ndarray = np.mean(
                auto_correlations, axis=1)[len(x)-1:]  # acf
            return mean_autocorrelation
        else:
            return auto_correlations


class autocorr_statsmodels(autocorr_abstract):
    """
    Concrete strategy class for computing 1D autocorrelation using Statsmodels.
    """
    @classmethod
    def compute_autocorr1d(cls, x: np.ndarray, normalize: bool = False) -> np.ndarray:  # x is 1D array
        """
        Compute the 1D autocorrelation of a NumPy array using Statsmodels.

        Parameters
        ----------
        x : np.ndarray
            A 1D NumPy array.

        Returns
        -------
        np.ndarray
            The 1D autocorrelation of the input array.

        Raises
        ------
        ValueError
            If the input array is not 1D.

        Examples
        --------
        >>> arr = np.array([1, 2, 3, 4, 5])
        >>> autocorr1d_statsmodels.compute_autocorr1d(arr)
        array([35., 33., 28., 22., 15.])
        """
        if len(x.shape) != 1:
            raise ValueError("Only 1D array is supported")
        if not normalize:
            result = sm.tsa.stattools.acf(x, fft=True, nlags=len(
                x))*np.std(x)*np.std(x)
        if normalize:
            result = sm.tsa.stattools.acf(x, fft=True, nlags=len(
                x))
        return result

    def compute_autocorr2d(cls, x: np.ndarray) -> np.ndarray:
        if len(x.shape) != 2:
            raise ValueError("Only 2D array is supported")
        auto_correlations: np.ndarray = np.apply_along_axis(lambda x: sm.tsa.stattools.acf(
            x, fft=True, nlags=len(x))*np.std(x)*np.std(x), axis=0, arr=x)
        # 自己相関の平均化 (axis=1で全ての時系列に対する平均を取る)
        mean_autocorrelation: np.ndarray = np.mean(
            auto_correlations, axis=1)[len(x)-1:]  # acf
        return mean_autocorrelation
