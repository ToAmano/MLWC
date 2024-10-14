

#
# * 計算したhydrogen_bond_listの自己相関を求める
#
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
#
import statsmodels.api as sm
import numpy as np
import scipy
from include.mlwc_logger import root_logger
logger = root_logger(__name__)
import abc

class autocorr1d(abc.ABC):
    """
    アルゴリズム（ConcreteStrategy）が実装する共通のインターフェイス
    """
    @classmethod
    @abc.abstractmethod
    def compute_autocorr1d(cls):
        pass

class autocorr1d_numpy(autocorr1d):
    @classmethod
    def compute_autocorr1d(x:np.ndarray) -> np.ndarray: # x is 1D array
        if len(x.shape) != 1:
            raise ValueError("Only 1D array is supported")
        # note:: numpy correlate do not support fft
        result = np.correlate(x, x, mode='full') # !! do not normalize
        return result[int(result.size/2):]

class autocorr1d_scipy(autocorr1d):
    @classmethod
    def autocorr1d_scipy(x:np.ndarray) -> np.ndarray: # x is 1D array
        if len(x.shape) != 1:
            raise ValueError("Only 1D array is supported")
        result = scipy.signal.correlate(x, x, mode="same",method="fft") #/len(x) # !! do not normalize
        return result[int(result.size/2):]

class autocorr1d_statsmodels(autocorr1d):
    @classmethod
    def compute_autocorr1d(x:np.ndarray) -> np.ndarray: # x is 1D array
        if len(x.shape) != 1:
            raise ValueError("Only 1D array is supported")
        result = sm.tsa.stattools.acf(x,fft=True,nlags=len(x))*np.std(x)*np.std(x) # !! do not normalize
        return result

