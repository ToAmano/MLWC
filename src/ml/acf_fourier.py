import numpy as np


class dielec:
    '''
    2023/4/18: 双極子モーメントのデータから，acf計算，フーリエ変換を行うツール
    今までhard codeになっていた温度などの変数を入力にしたい．

    inputs
    -----------------
     TIMESTEP :: 単位はfsにする．従って，calc_fourierの部分で単位変換が入る．
    '''
    def __init__(self,UNITCELL_VECTORS, TEMPERATURE, TIMESTEP):
        self.UNITCELL_VECTORS = UNITCELL_VECTORS
        self.TEMPERATURE      = TEMPERATURE
        self.TIMESTEP         = TIMESTEP

    def calc_acf(self, dipole_array): # ACFを返す
        return raw_calc_acf(dipole_array)
    def calc_eps0(self, dipole_array):
        return raw_calc_eps0(dipole_array, self.UNITCELL_VECTORS, self.TEMPERATURE)
    def calc_fourier(self, dipole_array, eps_n2:float):
        acf_x, acf_y, acf_z = dielec.calc_acf(self,dipole_array)
        fft_data = (acf_x+acf_y+acf_z)/3 # 平均化
        eps_0=dielec.calc_eps0(self,dipole_array)
        return raw_calc_fourier(fft_data, eps_0, eps_n2, self.TIMESTEP) # fft_data::acfがinputになる．
    def calc_fourier_only(self,fft_data,eps_0:float,eps_n2:float):
        return raw_calc_fourier(fft_data,eps_0, eps_n2, self.TIMESTEP)

def raw_calc_acf(dipole_array):
    import statsmodels.api as sm
    import numpy as np

    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dmx=dipole_array[:,0]-np.mean(dipole_array[:,0])
    dmy=dipole_array[:,1]-np.mean(dipole_array[:,1])
    dmz=dipole_array[:,2]-np.mean(dipole_array[:,2])

    # ACFの計算
    # TODO :: hard code ここでACFを計算するnlagsを固定している．あまり良くないかも．
    # 2023/5/29 acfの計算法をfft=trueへ変更！
    # 2023/5/31 nlags=N_acfを削除！
    N_acf = int(len(dmx)/2) # nlags=N_acf
    acf_x = sm.tsa.stattools.acf(dmx,fft=True)
    acf_y = sm.tsa.stattools.acf(dmy,fft=True)
    acf_z = sm.tsa.stattools.acf(dmz,fft=True)

    return acf_x, acf_y, acf_z

def raw_calc_eps0(dipole_array, UNITCELL_VECTORS, TEMPERATURE:float=300, ):
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


def raw_calc_fourier(fft_data, eps_0, eps_n2, TIMESTEP):
    '''
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
    eps_inf = 1.0  #これは固定すべし．
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
    
    # 誘電関数の計算
    # ffteps1の2項目の符号は反転させる必要があることに注意 !!
    # time_data*TIMESTEPは合計時間をかける意味
    ffteps1 = eps_0+(eps_0-eps_inf)*ans_times_omega.imag*(time_data*TIMESTEP) -1.0 + eps_n2
    ffteps2 = (eps_0-eps_inf)*ans_times_omega.real*(time_data*TIMESTEP)
    #
    return rfreq, ffteps1, ffteps2




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
