


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class DielModel:
    # The imaginary part of Havriliak-Negami function
    # omega = 2pi f
    # (alpha,beta) = (1,1) corresponds Debye
    # 虚部には-がつく定義により，先頭にマイナスがつく．
    @classmethod
    def havriliak_negami_imag(cls,omega:np.ndarray, delta_epsilon:float, tau:float, alpha:float, beta:float):
        return -(delta_epsilon / (1 + (1j * omega * tau)**alpha)**beta).imag

    @classmethod
    def debye_imag(cls,omega:np.ndarray, delta_epsilon:float, tau:float):
        return -(delta_epsilon / (1 + 1j * omega * tau)).imag

    # 複数のHavriliak-Negami関数を合計した関数の定義
    @classmethod
    def havriliak_negami_sum(cls,omega:np.ndarray, params):
        # paramsがdelta,tau,alpha,betaを与える
        num_terms = len(params) // 4
        epsilon = np.zeros_like(omega, dtype=np.float64)
        for i in range(num_terms):
            delta_epsilon = params[4 * i ]
            tau   = params[4 * i + 1]
            alpha = params[4 * i + 2]
            beta  = params[4 * i + 3]
            # epsilon += (delta_epsilon / (1 + (1j * omega * tau)**alpha)**beta).imag
            epsilon += havriliak_negami_imag(omega, delta_epsilon, tau, alpha, beta)
        return epsilon

    @classmethod
    # 残差関数の定義
    def residuals(params, omega:np.ndarray, y:np.ndarray):
        return y - havriliak_negami_sum(omega, params)
