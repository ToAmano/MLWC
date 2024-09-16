

#
# * 計算したhydrogen_bond_listの自己相関を求める
#
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
#
import statsmodels.api as sm
import numpy as np
import scipy

def autocorr1d_numpy(x:np.ndarray) -> np.ndarray: # x is 1D array
    if len(x.shape) != 1:
        raise ValueError("Only 1D array is supported")
    result = np.correlate(x, x, mode='full') # !! do not normalize
    return result[int(result.size/2):]

def autocorr1d_scipy(x:np.ndarray) -> np.ndarray: # x is 1D array
    if len(x.shape) != 1:
        raise ValueError("Only 1D array is supported")
    result = scipy.signal.correlate(x, x, mode="same",method="fft") # /len(x) # !! do not normalize
    return result[int(result.size/2):]


