

# Fourier Transform ACF data. Various different types of data can be used.
# Limitation:: The input data should be 1D array. For vector quantities, we have to take the inner product first.

# pattern 1: <v*v> type correlation. In this case, we only use the real part. Example: velocity autocorrelation function.
# pattern 2: omega <M*M> type correlaton. In this case, we use both real and imaginary parts. Example: dielectric function.
# pattern 3: <M*M>/omega type correlation. Example: dielectric function from time derivative of dipole moment.


"""
二つの計算方法のパターン

1. M(t)を入力として，acf(M)を計算，acf(M)をdenoise後，FFTしてeps(omega)を計算．
2. M(t)をdenoise後，FFTしてM(omega)を計算，これを畳み込んでeps(omega)を計算 -> めっちゃ早いが，多分実部の計算には使えない．

二つの計算したい値のパターン
1. 誘電関数(omegaあり)，実部と虚部が必要
2. VDOSのように実部/虚部のみ必要
3. そもそも，ACFだけほしいパターンや，FFTした値のみほしいパターンもある．

大元の入力
1. atoms -> 速度，距離などの情報を計算して入力とする． -> これはデータがtime+2次元データの3次元 -> 最後に平均化
2. totaldipole(微分タイプも) -> これはデータがtime+dataの2次元
3. moldipole(微分タイプも)   -> これはデータがtime+2次元データの3次元 -> 最後に平均化
だから，最終的にはデータが2次元の場合と，3次元の場合の2つのパターンを網羅すれば入力形式は二つですむ．

次に，二つの計算方法のパターンについては，このfouriertransformとautocorrelationをベースにしてwrapperにするのが良さそう．
- M(t)
   - 1. M(t)を入力すると，誘電関数を計算する関数 (methodでふた通りから選べる)  -> totaldipoleに実装
   - 2. M'(t)を入力すると，誘電関数を計算する関数 (methodでふた通りから選べる) -> totaldipoleに実装？

現在の実装で利用しているところ．
CPextract.py cpmd ROO, angleOH
CPextract.py cpmd vdos

"""

import pandas as pd
import numpy as np
from include.mlwc_logger import setup_cmdline_logger
logger = setup_cmdline_logger(__name__)


class fft:

    def __init__(self):
        pass

    @staticmethod
    def calculate_fft_core(time_array: np.ndarray, TIMESTEP_fs: float) -> list[np.ndarray, np.ndarray]:
        """calculate fft from time-series data

        This function is fft code specialized for time series analysis of MD trajectories.
        Tiically, time_array is the 

        NOTE
        --------------------
        フーリエ変換用のtimesteps (これが周波数・THz単位になるようにしたい．)
        !! 振動数ではないので注意 !!
        https://helve-blog.com/posts/python/numpy-fast-fourier-transform/

        1Hz=1/s．
        1THz=10^12 Hz
        1psec=10^(-12)s
        従って1THz=1/psecの関係にある． よってfourier変換の時間単位をpsec
        にしておけば返ってくる周波数はTHzということになる．


        dがサンプリング周期．単位をnsにすると横軸がちょうどTHzになる．
        例：1fsの時，1/1000

        - 上の方でeps_0=1+<M^2>みたいにしているため，本来のeps_0=eps_inf+<M^2>との辻褄合わせをここでやっている．
        - 公式としてもどれを使うかみたいなのが結構むずかしい．ここら辺はまた後でちゃんとまとめた方がよい．

        Args:
            acf_array (np.ndarray): _description_
            TIMESTEP (float): _description_

        Returns:
            list[np.ndarray, np.ndarray]: _description_
        """
        TIMESTEP: float = TIMESTEP_fs/1000  # fs to ps
        freq = np.fft.fftfreq(len(time_array), d=TIMESTEP)  # omega
        length = freq.shape[0]//2 + 1  # rfftでは，fftfreqのうちの半分しか使わない．
        rfreq_array_thz: np.ndarray = freq[0:length]

        # usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
        # norm="backward" for normalization with 1, norm="ortho" for normalization with 1/sqrt(N)
        fft_array: np.ndarray = np.fft.rfft(
            time_array, norm="forward")  # normalization with 1/N

        # denoise real part
        fft_real_denoise_array = fft_array.real - fft_array.real[-1]
        fft_array = fft_real_denoise_array + fft_array.imag*1j  # redifine fft_array

        return rfreq_array_thz, fft_array

    @staticmethod
    def calculate_fft_vdos(acf_array: np.ndarray, TIMESTEP_fs: float) -> pd.DataFrame:
        rfreq_array_thz, fft_array = fft.calc_fft_core(acf_array, TIMESTEP_fs)
        # VDOS:: time_data*TIMESTEP = Total MD time
        total_simulation_time: float = len(acf_array)*TIMESTEP_fs
        fftvdos: np.ndarray = 2*fft_array.real*total_simulation_time

        df = pd.DataFrame()
        df["freq_thz"] = rfreq_array_thz
        df["freq_kayser"] = rfreq_array_thz*33.3  # cm-1 = 33.3*THz
        # integral from -inf to inf. assure vdos(0)=0
        df["vdos"] = fftvdos-fftvdos[0]
        return df

    @staticmethod
    def calculate_fft_dielfunction(acf_array: np.ndarray, TIMESTEP_fs: float) -> pd.DataFrame:
        '''
        比例係数は無しで，単純に自己相関のfourier変換だけを計算する．
        input
        --------------------
        TIMESTEP :: データのtimestep[fs]. 入力後，1000で割ってpsに変換する．
        fft_data :: ACFの平均値を入れる．この量がFFTされる．
        eps_n2   :: 高周波誘電定数：ナフタレン=1.5821**2 (屈折率の二乗．高周波誘電定数~屈折率^2とかけることから)

        output
        --------------------
        rfreq :: THz単位の周波数グリッド

        NOTE
        --------------------
        フーリエ変換用のtimesteps (これが周波数・THz単位になるようにしたい．)
        !! 振動数ではないので注意 !!
        https://helve-blog.com/posts/python/numpy-fast-fourier-transform/

        1Hz=1/s．
        1THz=10^12 Hz
        1psec=10^(-12)s
        従って1THz=1/psecの関係にある． よってfourier変換の時間単位をpsec
        にしておけば返ってくる周波数はTHzということになる．


        dがサンプリング周期．単位をnsにすると横軸がちょうどTHzになる．
        例：1fsの時，1/1000

        - 上の方でeps_0=1+<M^2>みたいにしているため，本来のeps_0=eps_inf+<M^2>との辻褄合わせをここでやっている．
        - 公式としてもどれを使うかみたいなのが結構むずかしい．ここら辺はまた後でちゃんとまとめた方がよい．
        '''
        rfreq_array_thz, fft_array = fft.calc_fft_core(acf_array, TIMESTEP_fs)
        total_simulation_time: float = len(acf_array)*TIMESTEP_fs
        fourier_real = fft_array.real*total_simulation_time
        fourier_imag = fft_array.imag*total_simulation_time
        df = pd.DataFrame()
        df["freq_thz"] = rfreq_array_thz
        df["freq_kayser"] = rfreq_array_thz*33.3  # cm-1 = 33.3*THz
        df["real"] = fourier_real
        df["imag"] = fourier_imag
        return df
