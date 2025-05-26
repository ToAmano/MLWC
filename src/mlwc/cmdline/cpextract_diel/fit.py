"""
# FIXME :: Under Maintenance, DO NOT WORK

"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import mlwc.fourier.fit_diel
from mlwc.fourier.acf_fourier import (
    calc_total_mol_acf_cross,
    calc_total_mol_acf_self,
    dielec,
)
from mlwc.fourier.dipole_core import diel_function
from mlwc.include.mlwc_logger import setup_library_logger

logger = setup_library_logger("MLWC." + __name__)


def fit_diel(
    freq: np.ndarray,
    imag_diel: np.ndarray,
    num_hn_functions: int = 1,
    lower_bound: float = 0.1,
    upper_bound: float = 1.0,
):

    # keep initial freq
    init_freq = freq

    # フィッティング範囲の設定
    if lower_bound is not None:
        freq = freq[freq >= lower_bound]
        epsilon_imag = imag_diel[-len(freq) :]  # 周波数に対応する範囲で誘電率を切り取る

    if upper_bound is not None:
        freq = freq[freq <= upper_bound]
        epsilon_imag = imag_diel[: len(freq)]  # 周波数に対応する範囲で誘電率を切り取る

    # 初期推定値の設定
    initial_guess = [1, 1, 1e-3, 0.5, 0.5] * args.num_hn_functions

    # 制約条件の設定 (alphaとbetaは0から1の間)
    bounds_lower = [0, 0, 0, 0] * num_hn_functions
    bounds_upper = [np.inf, np.inf, 1, 1] * num_hn_functions
    # 最小二乗法によるフィッティングを実行
    result = least_squares(
        mlwc.fourier.fit_diel.residuals,
        initial_guess,
        bounds=(bounds_lower, bounds_upper),
        args=(freq, imag_diel),
    )
    logger.info(" ====================== ")
    logger.info("   fitting result       ")
    logger.info(f" {result.x}            ")
    logger.info(" ====================== ")

    # フィッティング結果
    epsilon_fit = havriliak_negami_sum(freq, result.x)

    # save to pd.dataframe
    df = pd.DataFrame()
    df["freq_kayser"] = init_freq
    df["fit_imag_diel"] = havriliak_negami_sum(init_freq, result.x)
    df.to_csv("fit_hn_diel_imag.csv")
    return df


def command_diel_fit(args):
    df = pd.read_csv(args.Filename)
    fit_diel(
        df["freq_kayser"].values,
        df["imag_diel"].values,
        args.num_hn_functions,
        args.lower_bound,
        args.upper_bound,
    )
    return 0
