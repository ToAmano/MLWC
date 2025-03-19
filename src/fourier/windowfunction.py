from scipy import signal
from typing import Literal
import numpy as np


def apply_windowfunction(data: np.ndarray, window: Literal["hann", "hamming", "blackman", "gaussian", None]):
    # https://dango-study.hatenablog.jp/entry/2021/06/22/201222
    fw1 = signal.windows.hann(len(data)*2)[len(data):]
    fw2 = signal.windows.hamming(len(data)*2)[len(data):]
    fw3 = signal.windows.blackman(len(data)*2)[len(data):]
    fw4 = signal.windows.gaussian(len(data)*2, std=len(data)/5)[len(data):]
    if window == "hann":
        return data*fw1
    elif window == "hamming":
        return data*fw2
    elif window == "blackman":
        return data*fw3
    elif window == "gaussian":
        return data*fw4
    elif window == None:
        return data
    else:
        print(f"ERROR: window function is not defined :: {window}")
        return 0
