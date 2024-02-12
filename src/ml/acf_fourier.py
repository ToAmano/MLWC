"""Calculate auto-correlation&Fourier transform

"""
import numpy as np


class dielec:
    """calculate dielectric function
    2023/4/18: 双極子モーメントのデータから，acf計算，フーリエ変換を行うツール
    今までhard codeになっていた温度などの変数を入力にしたい．

    inputs
    -----------------
     TIMESTEP :: 単位はfsにする．従って，calc_fourierの部分で単位変換が入る．
    """
    def __init__(self,UNITCELL_VECTORS, TEMPERATURE, TIMESTEP):
        self.UNITCELL_VECTORS = UNITCELL_VECTORS # Angstrom
        self.TEMPERATURE      = TEMPERATURE # K 
        self.TIMESTEP         = TIMESTEP # fs

    def calc_acf(self, dipole_array, nlags="all",mode="norm"): 
        """return ACF from dipoles

        Args:
            dipole_array (_type_): _description_
            nlags (str, optional): _description_. Defaults to "all".
            mode (str, optional): _description_. Defaults to "norm".

        Returns:
            _type_: _description_
        """
        return raw_calc_acf(dipole_array, nlags, mode)
    def calc_eps0(self, dipole_array): 
        """return eps_0 from dipoles
        eps_0 is given as relative dielectric constant (unitless)
        Args:
            dipole_array (np.ndarray): time series of total dipole moment

        Returns:
            _type_: _description_
        """
        return raw_calc_eps0(dipole_array, self.UNITCELL_VECTORS, self.TEMPERATURE)
    def calc_fourier(self, dipole_array, eps_n2:float, window:str=None):
        """return dielectric function from dipoles

        Args:
            dipole_array (_type_): _description_
            eps_n2 (float): eps_inf, electronic contribution.
            window (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        acf_x, acf_y, acf_z = dielec.calc_acf(self,dipole_array)
        fft_data = (acf_x+acf_y+acf_z)/3 # mean
        eps_0=dielec.calc_eps0(self,dipole_array)
        return raw_calc_fourier_window(fft_data, eps_0, eps_n2, self.TIMESTEP, window) # fft_data::acfがinputになる．（デフォルトでは窓関数なし）
    def calc_fourier_only(self,fft_data,eps_0:float,eps_n2:float): # fft_data::acfを直接inputにする．これは窓関数をかけるときに便利
        return raw_calc_fourier(fft_data,eps_0, eps_n2, self.TIMESTEP)
    def calc_fourier_only_with_window(self,fft_data,eps_0:float,eps_n2:float,window="hann"): # デフォルトで窓関数hannをかける．
        return raw_calc_fourier_window(fft_data, eps_0, eps_n2, self.TIMESTEP, window)

def raw_calc_fourier_window(fft_data, eps_0:float, eps_n2:float, TIMESTEP:float, window:str="hann"):
    """wrapper function of raw_calc_fourier to apply window function
    
    As usual process to smooth spectrum, we apply window function to acf.
        ACF'(t) = ACF(t)*w(t)
    and cauclate FT of ACF' instead of ACF.
    
    NOTE !! :: Basically, we only use hann window.

    Args:
        fft_data (_type_): _description_
        eps_0 (float): _description_
        eps_n2 (float): _description_
        TIMESTEP (float): _description_
        window (str, optional): _description_. Defaults to "hann".

    Returns:
        _type_: _description_
    """
    from scipy import signal
    # https://dango-study.hatenablog.jp/entry/2021/06/22/201222
    fw1 = signal.hann(len(fft_data)*2)[len(fft_data):]      # ハニング窓
    fw2 = signal.hamming(len(fft_data)*2)[len(fft_data):]    # ハミング窓
    fw3 = signal.blackman(len(fft_data)*2)[len(fft_data):]   # ブラックマン窓
    fw4 = signal.gaussian(len(fft_data)*2,std=len(fft_data)/5)[len(fft_data):]   # ガウス窓
    if window == "hann":
        return raw_calc_fourier(fft_data*fw1,eps_0, eps_n2, TIMESTEP)
    elif window == "hamming":
        return raw_calc_fourier(fft_data*fw2,eps_0, eps_n2, TIMESTEP)        
    elif window == "blackman":
        return raw_calc_fourier(fft_data*fw3,eps_0, eps_n2, TIMESTEP)        
    elif window == "gaussian":
        return raw_calc_fourier(fft_data*fw4,eps_0, eps_n2, TIMESTEP)            
    elif window == None:
        return raw_calc_fourier(fft_data,eps_0, eps_n2, TIMESTEP)
    else:
        print("ERROR: window function is not defined")
        return 0
    
    

def raw_calc_acf(dipole_array: np.array, nlags: str = "all", mode="norm"):
    '''
    dipole_array : N*3次元配列
    '''
    import statsmodels.api as sm
    import numpy as np
    if nlags == "all":
        N_acf = len(dipole_array[:,0]) # すべて利用する場合
    elif nlags == "half":
        N_acf = int(len(dipole_array[:,0])/2) # 先頭半分利用する場合
    else:
        print("ERROR: nlags is not defined")
        return 0
    # dipole_arrayがN*3次元配列であることの確認
    if dipole_array.shape[1] != 3:
        print("ERROR: dipole_array is not 3D array")
        return 0

    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dmx=dipole_array[:,0]-np.mean(dipole_array[:,0])
    dmy=dipole_array[:,1]-np.mean(dipole_array[:,1])
    dmz=dipole_array[:,2]-np.mean(dipole_array[:,2])

    # ACFの計算
    # TODO :: hard code ここでACFを計算するnlagsを固定している．あまり良くないかも．
    # 2023/5/29 acfの計算法をfft=trueへ変更！
    # 2023/5/31 nlags=N_acfを削除！
    # N_acf = int(len(dmx)/2) # nlags=N_acf
    # N_acf = len(dmx) # すべて利用する場合
    acf_x = sm.tsa.stattools.acf(dmx,fft=True,nlags=N_acf)
    acf_y = sm.tsa.stattools.acf(dmy,fft=True,nlags=N_acf)
    acf_z = sm.tsa.stattools.acf(dmz,fft=True,nlags=N_acf)
    
    if not mode=="norm": #正規化しない場合，ACF(t=0)=<M(0)M(0)>=<M^2>を計算する．
        acf_x = acf_x*np.std(dmx)*np.std(dmx)
        acf_y = acf_y*np.std(dmy)*np.std(dmy)
        acf_z = acf_z*np.std(dmz)*np.std(dmz)
    return acf_x, acf_y, acf_z

def raw_calc_eps0(dipole_array, UNITCELL_VECTORS, TEMPERATURE:float=300 ):
    '''
    TEMPERATURE :: eps0を計算するのに利用する温度
    '''
    # >>>>>>>>>>>
    eps0 = 8.8541878128e-12
    debye = 3.33564e-30
    nm3 = 1.0e-27
    nm = 1.0e-9
    A3 = 1.0e-30
    kb = 1.38064852e-23
    # T =300

    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dMx=dipole_array[:,0] # -np.mean(dipole_array[:,0])
    dMy=dipole_array[:,1] # -np.mean(dipole_array[:,1])
    dMz=dipole_array[:,2] # -np.mean(dipole_array[:,2])

    V = np.abs(np.dot(np.cross(UNITCELL_VECTORS[:,0],UNITCELL_VECTORS[:,1]),UNITCELL_VECTORS[:,2])) * A3

    print("SUPERCELL VOLUME (m^3) :: ", V )
    # V=   11.1923*11.1923*11.1923 * A3
    kbT = kb * TEMPERATURE

    # 平均値計算
    mean_M2=(np.mean(dMx**2)+np.mean(dMy**2)+np.mean(dMz**2))
    mean_M=np.mean(dMx)**2+np.mean(dMy)**2+np.mean(dMz)**2

    # 比誘電率
    # !! 1.0とあるのはeps_inf=1.0とおいて計算しているため．
    # !! 後のfourier変換のところでeps_inf部分の修正を効かせるようになってる
    eps_0 = 1.0 + ((mean_M2-mean_M)*debye**2)/(3.0*V*kbT*eps0)

    # 比誘電率
    # eps_0 = 1.0 + ((np.mean(dMx_pred**2+dMy_pred**2+dMz_pred**2))*debye**2)/(3.0*V*kbT*eps0)
    print("EPS_0 {0}, mean_M {1}, mean_M2 {2}:: ".format(eps_0, mean_M, mean_M2))
    return eps_0

def calc_coeff(UNITCELL_VECTORS, TEMPERATURE:float=300):
    # >>>>>>>>>>>
    eps0 = 8.8541878128e-12
    debye = 3.33564e-30
    nm3 = 1.0e-27
    nm = 1.0e-9
    A3 = 1.0e-30
    kb = 1.38064852e-23
    # T =300

    V = np.abs(np.dot(np.cross(UNITCELL_VECTORS[:,0],UNITCELL_VECTORS[:,1]),UNITCELL_VECTORS[:,2])) * A3

    print("SUPERCELL VOLUME (m^3) :: ", V )
    # V=   11.1923*11.1923*11.1923 * A3
    kbT = kb * TEMPERATURE

    # 比誘電率
    # !! 1.0とあるのはeps_inf=1.0とおいて計算しているため．
    # !! 後のfourier変換のところでeps_inf部分の修正を効かせるようになってる
    eps_0 = (debye**2)/(3.0*V*kbT*eps0)

    return eps_0


def raw_calc_fourier(fft_data, eps_0:float, eps_n2:float, TIMESTEP:float):
    """_summary_

    Args:
        fft_data (_type_): ACFの平均値を入れる．この量がFFTされる．
        eps_0 (float): _description_
        eps_n2 (float): 高周波誘電定数：ナフタレン=1.5821**2 (屈折率の二乗．高周波誘電定数~屈折率^2とかけることから)
        TIMESTEP (float): データのtimestep[fs]. mdtrajからloadしたものを利用するのを推奨

    Returns:
        _type_: rfreq :: THz単位の周波数グリッド

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
    """
    
    # eps_inf to 1.0 (vaccume)
    # !! Fix it
    eps_inf = 1.0  
    TIMESTEP = TIMESTEP/1000 # fs to ps
    
    # 
    time_data=len(fft_data) # データの長さ
    freq=np.fft.fftfreq(time_data, d=TIMESTEP) # omega
    length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
    rfreq=freq[0:length] # THz
    
    #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
    ans=np.fft.rfft(fft_data, norm="forward" ) #こっちが1/Nがかかる規格化．
    #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
    #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる
    
    ans_real_denoise= ans.real-ans.real[-1] # 振幅が閾値未満はゼロにする（ノイズ除去）
    # print(ans.real)
    ans = ans_real_denoise + ans.imag*1j # 再度定義のし直しが必要
    
    # 2pi*f*L[ACF]
    ans_times_omega=ans*rfreq*2*np.pi
    
    # 誘電関数の計算
    # ffteps1の2項目の符号は反転させる必要があることに注意 !!
    # time_data*TIMESTEPは合計時間をかける意味
    ffteps1 = eps_0+(eps_0-eps_inf)*ans_times_omega.imag*(time_data*TIMESTEP) -1.0 + eps_n2
    ffteps2 = (eps_0-eps_inf)*ans_times_omega.real*(time_data*TIMESTEP)
    #
    return rfreq, ffteps1, ffteps2



def raw_calc_only_acffourier(fft_data, TIMESTEP):
    '''
    比例係数は無しで，単純に自己相関のfourier変換だけを計算する．
    input
    --------------------
     TIMESTEP :: データのtimestep[psec]. mdtrajからloadしたものを利用するのを推奨
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
    ans=np.fft.rfft(fft_data, norm="forward" ) #こっちが1/Nがかかる規格化．
    #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
    #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる
    ans_real_denoise= ans.real-ans.real[-1] # 振幅が閾値未満はゼロにする（ノイズ除去）
    # print(ans.real)
    ans = ans_real_denoise + ans.imag*1j # 再度定義のし直しが必要
    
    # 2pi*f*L[ACF]
    ans_times_omega=ans*rfreq*2*np.pi
    #
    # 誘電関数の計算
    # ffteps1の2項目の符号は反転させる必要があることに注意 !!
    # time_data*TIMESTEPは合計時間をかける意味
    acf_fourier_real = ans_times_omega.real*(time_data*TIMESTEP)
    acf_fourier_imag = ans_times_omega.imag*(time_data*TIMESTEP)
    #
    return rfreq, acf_fourier_real, acf_fourier_imag


def raw_calc_acffourier_with_amplitude(fft_data, TIMESTEP,eps_inf,UNITCELL_VECTORS, TEMPERATURE):
    '''
    比例計数も入れる．各要素の分解計算についてはすでにスライドにまとめてある．
    input
    --------------------
     TIMESTEP :: データのtimestep[psec]. mdtrajからloadしたものを利用するのを推奨
     fft_data :: ACFの平均値を入れる．（x,y,zについて平均化しておく）この量がFFTされる．
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
    print("fft_data :: ", fft_data.shape)
    
    # まずは，誘電定数に相当する部分を計算する．（最後1を引いてeps_inf部分を除去している．
    coeff = calc_coeff(UNITCELL_VECTORS, TEMPERATURE)
    
    # 窓関数
    from scipy import signal
    fw1 = signal.hann(len(fft_data)*2)[len(fft_data):]      # ハニング窓
        
    TIMESTEP = TIMESTEP/1000 # fs to ps
    
    time_data=len(fft_data)
    freq=np.fft.fftfreq(time_data, d=TIMESTEP) # omega
    length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
    rfreq=freq[0:length]


    #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
    ans=np.fft.rfft(fft_data*fw1, norm="forward" ) #こっちが1/Nがかかる規格化．
    #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
    #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる
    ans_real_denoise= ans.real-ans.real[-1] # 振幅が閾値未満はゼロにする（ノイズ除去）
    # print(ans.real)
    ans = ans_real_denoise + ans.imag*1j # 再度定義のし直しが必要
    
    # 2pi*f*L[ACF]
    ans_times_omega=ans*rfreq*2*np.pi
    #
    # 誘電関数の計算
    # ffteps1の2項目の符号は反転させる必要があることに注意 !!
    # time_data*TIMESTEPは合計時間をかける意味
    ffteps1 = eps_inf+coeff*(fft_data[0]+ans_times_omega.imag*(time_data*TIMESTEP) )
    ffteps2 = coeff*ans_times_omega.real*(time_data*TIMESTEP)
    #
    return rfreq, ffteps1, ffteps2


def calc_mol_acf(tmp_data,i,j,engine="scipy"):
    '''
    tmp_data :: total_dipole.txtから読み込んだ3次元データ．[frame,mol_id,3dvector]
    分子index iと分子index jの相互相関関数を計算する．
    注意：：engine == "scipy"の場合，ちゃんと平均を引く必要があるようだ．
    https://qiita.com/SHUsss/items/a357a33849d583093849
    '''
    # 誘電関数の計算まで
    import statsmodels.api as sm 
    import numpy as np
    # cell_dipoles_pred = np.load(filename)
    data_i = tmp_data[:,i,:]
    data_j = tmp_data[:,j,:]
    
    N=int(np.shape(data_i)[0]/2)
    # N=int(np.shape(cell_dipoles_pred)[0])
    # N=99001
    # print("nlag :: ", N)
    
    #自己相関関数を求める
    if engine == "tsa":
        #（元々nlag=N,fft=Falseだった．fft=Trueのほうが計算は早くなる．）
        acf_x_pred = sm.tsa.stattools.ccf(data_i[:,0],data_j[:,0],fft=True)*np.std(data_i[:,0]) * np.std(data_j[:,0])
        acf_y_pred = sm.tsa.stattools.ccf(data_i[:,1],data_j[:,1],fft=True)*np.std(data_i[:,1]) * np.std(data_j[:,1])
        acf_z_pred = sm.tsa.stattools.ccf(data_i[:,2],data_j[:,2],fft=True)*np.std(data_i[:,2]) * np.std(data_j[:,2])
        pred_data =(acf_x_pred+acf_y_pred+acf_z_pred)/3
        time=times[:len(acf_x_pred)]
    elif engine == "scipy":
        from scipy import signal
        acf_x_pred = signal.correlate(data_i[:,0]-np.mean(data_i[:,0]), data_j[:,0]-np.mean(data_j[:,0]), mode="same",method="fft")/len(data_i[:,0])
        acf_y_pred = signal.correlate(data_i[:,1]-np.mean(data_i[:,1]), data_j[:,1]-np.mean(data_j[:,1]), mode="same",method="fft")/len(data_i[:,1])
        acf_z_pred = signal.correlate(data_i[:,2]-np.mean(data_i[:,2]), data_j[:,2]-np.mean(data_j[:,2]), mode="same",method="fft")/len(data_i[:,2])
        pred_data =(acf_x_pred+acf_y_pred+acf_z_pred)/3
    else:
        print("ERROR :: engine is not defined.")
        return -1 
    return  pred_data


def calc_cross_acf(data_1,data_2,engine="scipy"):
    '''
    tmp_data :: total_dipole.txtから読み込んだ3次元データ．[frame,mol_id,3dvector]
    分子index iと分子index jの相互相関関数を計算する．
    注意：：engine == "scipy"の場合，ちゃんと平均を引く必要があるようだ．
    https://qiita.com/SHUsss/items/a357a33849d583093849
    '''
    # 誘電関数の計算まで
    import statsmodels.api as sm 
    import numpy as np
    # cell_dipoles_pred = np.load(filename)
    if len(data_1) != len(data_2):
        print("ERROR :: len(data_1) != len(data_2)")
    
    
    N=int(np.shape(data_1)[0]/2)
    # N=int(np.shape(cell_dipoles_pred)[0])
    # N=99001
    # print("nlag :: ", N)
    
    #自己相関関数を求める
    if engine == "tsa":
        #（元々nlag=N,fft=Falseだった．fft=Trueのほうが計算は早くなる．）
        acf_x_pred = sm.tsa.stattools.ccf(data_1[:,0],data_2[:,0],fft=True)*np.std(data_1[:,0]) * np.std(data_2[:,0])
        acf_y_pred = sm.tsa.stattools.ccf(data_1[:,1],data_2[:,1],fft=True)*np.std(data_1[:,1]) * np.std(data_2[:,1])
        acf_z_pred = sm.tsa.stattools.ccf(data_1[:,2],data_2[:,2],fft=True)*np.std(data_1[:,2]) * np.std(data_2[:,2])
        pred_data =(acf_x_pred+acf_y_pred+acf_z_pred)/3
    elif engine == "scipy":
        from scipy import signal
        acf_x_pred = signal.correlate(data_1[:,0]-np.mean(data_1[:,0]), data_2[:,0]-np.mean(data_2[:,0]), mode="same",method="fft")/len(data_1[:,0])
        acf_y_pred = signal.correlate(data_1[:,1]-np.mean(data_1[:,1]), data_2[:,1]-np.mean(data_2[:,1]), mode="same",method="fft")/len(data_1[:,1])
        acf_z_pred = signal.correlate(data_1[:,2]-np.mean(data_1[:,2]), data_2[:,2]-np.mean(data_2[:,2]), mode="same",method="fft")/len(data_1[:,2])
        pred_data =(acf_x_pred+acf_y_pred+acf_z_pred)/3
    else:
        print("ERROR :: engine is not defined.")
        return -1 
    return  acf_x_pred,acf_y_pred,acf_z_pred




def calc_mol_abs_acf(tmp_data,i,j,engine="scipy"):
    '''
    tmp_data :: total_dipole.txtから読み込んだ3次元データ．[frame,mol_id,3dvector]
    分子index iと分子index jの相互相関関数を計算する．
    '''
    # 誘電関数の計算まで
    import statsmodels.api as sm 
    import numpy as np
    # cell_dipoles_pred = np.load(filename)
    data_i = np.linalg.norm(tmp_data[:,i,:],axis=1)
    data_j = np.linalg.norm(tmp_data[:,j,:],axis=1)
    
    N=int(np.shape(data_i)[0]/2)
    # N=int(np.shape(cell_dipoles_pred)[0])
    # N=99001
    # print("nlag :: ", N)
    
    #自己相関関数を求める
    if engine == "tsa":
        #（元々nlag=N,fft=Falseだった．fft=Trueのほうが計算は早くなる．）
        # pred_data = sm.tsa.stattools.ccf(data_i[:,0],data_j[:,0],fft=True)*np.std(data_i[:,0]) * np.std(data_j[:,0])
        pred_data = sm.tsa.stattools.ccf(data_i,data_j,fft=True)
    elif engine == "scipy":
        from scipy import signal
        pred_data = signal.correlate(data_i,data_j,mode="same",method="fft")/len(data_i[:,0])
    else:
        print("ERROR :: engine is not defined.")
        return -1 
    return  pred_data

def plot_ACF(acf_x, acf_y, acf_z):
    import matplotlib.pyplot as plt
    ##=====================
    ## 自己相関関数の図示
    ##=====================
    ##
    plt.plot(acf_x,label="x")
    plt.plot(acf_y,label="y")
    plt.plot(acf_z,label="z")
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("ACF")
    plt.title("ACF vs timestep")
    plt.show()
    return 0

# 現状rfreqはTHz単位で入ることになっている．
def plot_diel(rfreq,ffteps1,ffteps2,FREQ_MAX=10 , ymax=None):
    import matplotlib.pyplot as plt
    import numpy as np
    #
    print("####  WARNING  #####")
    print(" rfreq should be in THz unit ")
    print("####################")

    # [0,FREQ_MAX]での最大最小を導出する．
    # まず，tmin,tmaxに応じたdatasのindexを取得
    max_indx=np.where(rfreq[:]<=FREQ_MAX)[0][-1] # 最大値に対応するindx
    # eps1用の最大最小値
    ymin1=np.min(ffteps1[:max_indx])*0.9  # みやすさのために1.1倍，0.9倍する．
    ymax1=np.max(ffteps1[:max_indx])*1.1


    # ymaxが与えられなかった場合はFREQ_MAXに応じて自動で計算
    if ymax==None:
        # eps2用の最大値
        ymax2=np.max(ffteps2[:max_indx])*1.1
    #
    else: # それ以外の場合はymax=ymax2とする
        ymax2=ymax


    ########################
    ## eps_1のプロット
    plt.plot(rfreq, ffteps1 , label="eps1_EMD")
    #plt.plot(exp_data[:,0],exp_data[:,2], label="experiment")
    #plt.scatter(nemd_freq, nemd_eps1-1.0+eps_n2, label="eps1_NEMD",color="red")
    plt.legend()
    plt.xlim(0,FREQ_MAX)
    plt.ylim(ymin1,ymax1)
    plt.xlabel("frequency [THz]")
    plt.ylabel("eps1")
    #plt.xscale('log')
    plt.show()

    ########################
    ## eps_2のプロット
    plt.plot(rfreq, ffteps2 , label="eps2_EMD")
    #plt.plot(exp_data[:,0],exp_data[:,1], label="experiment")
    #plt.scatter(nemd_freq, nemd_eps2, label="eps2_NEMD",color="red")
    plt.legend()
    plt.xlabel("frequency [THz]")
    plt.ylabel("eps2")
    plt.xlim(0,FREQ_MAX)
    plt.ylim(0,ymax2)
    #plt.xscale('log')
    plt.show()
    return 0


def calc_mol_acf(vector_data_1,vector_data_2,engine:str="scipy"):
    """分子双極子（3Dvector）を想定し，自己相関および相互相関を計算

    分子index iと分子index jの相互相関関数を計算する．
    Args:
        vector_data_1 (_type_): total_dipole.txtから読み込んだ3次元データ．[frame,3dvector]
        vector_data_2 (_type_): total_dipole.txtから読み込んだ3次元データ．[frame,3dvector]
        engine (str, optional): _description_. Defaults to "scipy".

    Returns:
        _type_: _description_
    """
    # 誘電関数の計算まで
    import statsmodels.api as sm 
    import numpy as np
    if np.shape(vector_data_1)[1] != 3:
        print(" ERROR vector_1 wrong shape")
        return 1
    if np.shape(vector_data_2)[1] != 3:
        print(" ERROR vector_1 wrong shape")
        return 1
    if np.shape(vector_data_1)[0] != np.shape(vector_data_2)[0]:
        print(" ERROR vector_1 not consistent with vector2")
        return 1
    
    # cell_dipoles_pred = np.load(filename)
    # データは，平均値を引かないといけない．
    data_i = vector_data_1-np.mean(vector_data_1,axis=0)
    data_j = vector_data_2-np.mean(vector_data_2,axis=0)
    
    N=int(np.shape(data_i)[0])
    # N=int(np.shape(cell_dipoles_pred)[0])
    # N=99001
    # print("nlag :: ", N)
    
    #自己相関関数を求める
    if engine == "tsa":
        #（元々nlag=N,fft=Falseだった．fft=Trueのほうが計算は早くなる．）
        # nlagsはccfの場合，デフォルトでlen(data)となる．
        acf_x_pred = sm.tsa.stattools.ccf(data_i[:,0],data_j[:,0],fft=True)*np.std(data_i[:,0]) * np.std(data_j[:,0])
        acf_y_pred = sm.tsa.stattools.ccf(data_i[:,1],data_j[:,1],fft=True)*np.std(data_i[:,1]) * np.std(data_j[:,1])
        acf_z_pred = sm.tsa.stattools.ccf(data_i[:,2],data_j[:,2],fft=True)*np.std(data_i[:,2]) * np.std(data_j[:,2])
        pred_data =(acf_x_pred+acf_y_pred+acf_z_pred)/3
        time=times[:len(acf_x_pred)]
    elif engine == "scipy":
        from scipy import signal
        acf_x_pred = signal.correlate(data_i[:,0],data_j[:,0],mode="same",method="fft")/len(data_i[:,0])
        acf_y_pred = signal.correlate(data_i[:,1],data_j[:,1],mode="same",method="fft")/len(data_i[:,1])
        acf_z_pred = signal.correlate(data_i[:,2],data_j[:,2],mode="same",method="fft")/len(data_i[:,2])
        pred_data =(acf_x_pred+acf_y_pred+acf_z_pred)/3
    else:
        print("ERROR :: engine is not defined.")
        return -1 
    return  pred_data


def calc_mol_abs_acf(tmp_data,i,j,engine="scipy"):
    '''
    tmp_data :: total_dipole.txtから読み込んだ3次元データ．[frame,mol_id,3dvector]
    分子index iと分子index jの相互相関関数を計算する．
    '''
    # 誘電関数の計算まで
    import statsmodels.api as sm 
    import numpy as np
    # cell_dipoles_pred = np.load(filename)
    data_i = np.linalg.norm(tmp_data[:,i,:],axis=1)
    data_j = np.linalg.norm(tmp_data[:,j,:],axis=1)
    
    N=int(np.shape(data_i)[0]/2)
    # N=int(np.shape(cell_dipoles_pred)[0])
    # N=99001
    # print("nlag :: ", N)
    
    #自己相関関数を求める
    if engine == "tsa":
        #（元々nlag=N,fft=Falseだった．fft=Trueのほうが計算は早くなる．）
        # pred_data = sm.tsa.stattools.ccf(data_i[:,0],data_j[:,0],fft=True)*np.std(data_i[:,0]) * np.std(data_j[:,0])
        pred_data = sm.tsa.stattools.ccf(data_i,data_j,fft=True)
    elif engine == "scipy":
        from scipy import signal
        pred_data = signal.correlate(data_i,data_j,mode="same",method="fft")/len(data_i[:,0])
    else:
        print("ERROR :: engine is not defined.")
        return -1 
    return  pred_data


#
# * 分子双極子の自己相関を計算
def mol_dipole_selfcorr(molecule_dipole, NUM_MOL:int):
    """molecule_dipole[frame,mol_id,3]から自己相関成分の和（平均ではない！！）を計算
    

    Args:
        molecule_dipole (_type_): _description_
        NUM_MOL (int): _description_
    """
    
    data_self = []
    for i in range(32): # 分子のループ
        data_self.append(calc_mol_acf(molecule_dipole[:,i,], molecule_dipole[:,i,], engine="tsa"))
    # 1つのtrajectoryの分子について和をとる．
    return np.sum(np.array(data_self),axis=0)


def mol_dipole_crosscorr(molecule_dipole, NUM_MOL:int):
    """molecule_dipole[frame,mol_id,3]から相互相関成分の和（平均ではない！！）を計算

    Args:
        molecule_dipole (_type_): _description_
        NUM_MOL (int): _description_

    Returns:
        _type_: _description_
    """
    #
    # * 次に違う分子間の相関関数Psi_interを計算する
    data_inter_tmp = []
    for i in range(NUM_MOL):
        for j in range(NUM_MOL):
            if i == j: # i=jはACFにになるので飛ばす．
                continue
            data_inter_tmp.append(calc_mol_acf(molecule_dipole[:,i,:],molecule_dipole[:,j,0],engine="tsa"))
    # nC2個のデータについては和をとる．
    return np.sum(np.array(data_inter_tmp),axis=0)

