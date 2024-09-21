



# Fourier Transform ACF data. Various different types of data can be used.
# Limitation:: The input data should be 1D array. For vector quantities, we have to take the inner product first.

# pattern 1: <v*v> type correlation. In this case, we only use the real part. Example: velocity autocorrelation function.
# pattern 2: omega <M*M> type correlaton. In this case, we use both real and imaginary parts. Example: dielectric function.
# pattern 3: <M*M>/omega type correlation. Example: dielectric function from time derivative of dipole moment.

import pandas as pd
import numpy as np
from include.mlwc_logger import root_logger

class fft:
    
    def __init__(self):
        pass

    @staticmethod
    def calc_fft_core(acf_array:np.ndarray, TIMESTEP:float )-> list[np.ndarray, np.ndarray]:
        """calculate fft from acf data

        Args:
            acf_array (np.ndarray): _description_
            TIMESTEP (float): _description_

        Returns:
            list[np.ndarray, np.ndarray]: _description_
        """
        TIMESTEP = TIMESTEP/1000 # fs to ps
        freq=np.fft.fftfreq(len(acf_array), d=TIMESTEP) # omega
        length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
        rfreq_array:np.ndarray=freq[0:length] # rfreq in THz
        
        #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
        fft_array:np.ndarray=np.fft.rfft(acf_array, norm="forward" ) #こっちが1/Nがかかる規格化．
        #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
        #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる
        
        # denoise real part
        fft_real_denoise_array= fft_array.real-fft_array.real[-1] # 振幅が閾値未満はゼロにする（ノイズ除去）
        fft_array = fft_real_denoise_array + fft_array.imag*1j # redifine fft_array
        
        return rfreq_array, fft_array

    @staticmethod
    def calc_fft_vdostype(acf_array:np.ndarray, TIMESTEP:float)-> pd.DataFrame:
        import pandas as pd
        rfreq_array, fft_array = fft.calc_fft_core(acf_array, TIMESTEP)
        # VDOS:: time_data*TIMESTEP = Total MD time 
        fftvdos = 2*fft_array.real*(len(acf_array)*TIMESTEP) 
        
        df = pd.DataFrame()
        df["thz"] = rfreq_array
        df["freq_kayser"] = rfreq_array*33.3 # cm-1 = 33.3*THz
        df["vdos"] = fftvdos 
        return df
        
    @staticmethod
    def calc_fft(acf_array:np.ndarray, TIMESTEP:float)-> pd.DataFrame:
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
        rfreq_array, fft_array = fft.calc_fft_core(acf_array, TIMESTEP)
        # 誘電関数の計算
        # ffteps1の2項目の符号は反転させる必要があることに注意 !!
        # time_data*TIMESTEPは合計時間をかける意味
        acf_fourier_real = fft_array.real*rfreq_array*2*np.pi*(len(acf_array)*TIMESTEP)
        acf_fourier_imag = fft_array.imag*rfreq_array*2*np.pi*(len(acf_array)*TIMESTEP)
        # 
        return rfreq_array, acf_fourier_real, acf_fourier_imag

def calc_fft()->pd.DataFrame:
    return 0
    




def raw_calc_only_acffourier_type2(fft_data, TIMESTEP):
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
    TIMESTEP = TIMESTEP/1000 # fs to ps
    
    time_data=len(fft_data)
    freq=np.fft.fftfreq(time_data, d=TIMESTEP) # omega
    length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
    rfreq=freq[0:length]


    #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
    ans=np.fft.rfft(fft_data, norm="forward") #こっちが1/Nがかかる規格化．
    #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
    #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる
    #
    # 誘電関数の計算
    # ffteps1の2項目の符号は反転させる必要があることに注意 !!
    # time_data*TIMESTEPは合計時間をかける意味
    acf_fourier_real = ans.real*(time_data*TIMESTEP)
    acf_fourier_imag = ans.imag*(time_data*TIMESTEP)
    #
    return rfreq, acf_fourier_real, acf_fourier_imag