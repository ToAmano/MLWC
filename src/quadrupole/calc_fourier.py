import numpy as np


def calc_fourier(fft_data, eps_0, eps_n2, TIMESTEP):
    '''
    
    ######### input ###############
     TIMESTEP :: データのtimestep[psec]. mdtrajからloadしたものを利用するのを推奨
     fft_data :: ACFの平均値を入れる．この量がFFTされる．
     eps_n2   :: ナフタレン=1.5821**2
     FREQ_MAX :: 図示する上限[THz]．（将来的にうまいことpltのwrapperにしたい．）

    eps_inf = 1.0  #これは固定すべし．
    #   

     ==========================
     フーリエ変換用のtimesteps (これが周波数・THz単位になるようにしたい．)
     !! 振動数ではないので注意 !!
     https://helve-blog.com/posts/python/numpy-fast-fourier-transform/

     1Hz=1/s．
     1THz=10^12 Hz
     1psec=10^(-12)s
     従って1THz=1/psecの関係にある． よってfourier変換の時間単位をpsec
     にしておけば返ってくる周波数はTHzということになる．

    # ==========================
    # dがサンプリング周期．単位をnsにすると横軸がちょうどTHzになる．
    # 例：1fsの時，1/1000
    # ==========================
    '''
    
    time_data=len(fft_data)
    freq=np.fft.fftfreq(time_data, d=TIMESTEP) # omega
    length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
    rfreq=freq[0:length]


    #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
    ans=np.fft.rfft(fft_data, norm="forward" ) #こっちが1/Nがかかる規格化．
    #ans=np.fft.rfft(fft_data, norm="backward") #その他の規格化1:何もかからない
    #ans=np.fft.rfft(fft_data, norm="ortho")　　#その他の規格化2:1/sqrt(N))がかかる

    # 2pi*f*L[ACF]
    ans_times_omega=ans*rfreq*2*np.pi


    # 誘電関数の計算
    # ffteps1の2項目の符号は反転させる必要があることに注意 !!
    # time_data*TIMESTEPは合計時間をかける意味
    ffteps1 = eps_0+(eps_0-eps_inf)*ans_times_omega.imag*(time_data*TIMESTEP) -1.0 + eps_n2 
    ffteps2 = (eps_0-eps_inf)*ans_times_omega.real*(time_data*TIMESTEP)
    # 

    return rfreq, ffteps1, ffteps2


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
    
