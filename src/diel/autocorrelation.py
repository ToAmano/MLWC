

#
# * 計算したhydrogen_bond_listの自己相関を求める
#
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
#
import statsmodels.api as sm
import numpy as np
import scipy

def autocorr_numpy(x): 
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def autocorr_scipy(x):
    result = scipy.signal.correlate(x, x, mode="same",method="fft")/len(x)
    return result[int(result.size/2):]
