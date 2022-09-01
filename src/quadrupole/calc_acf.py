

##====================
## 3成分の自己相関関数を計算
##====================

def calculate_ACF(dipole_array, VOLUME):
    import statsmodels.api as sm
    import numpy as np

    # 各軸のdipoleを抽出．平均値を引く(eps_0のため．ACFは実装上影響なし)
    dmx=dipole_array[:,0]-np.mean(dipole_array[:,0])
    dmy=dipole_array[:,1]-np.mean(dipole_array[:,1])
    dmz=dipole_array[:,2]-np.mean(dipole_array[:,2])
    
    # ACFの計算
    # !!caution!! ここでACFを計算するnlagsを固定している．あまり良くないかも．
    N_acf = int(len(dmx)/2)
    acf_x = sm.tsa.stattools.acf(dmx,nlags=N_acf,fft=False)
    acf_y = sm.tsa.stattools.acf(dmy,nlags=N_acf,fft=False)
    acf_z = sm.tsa.stattools.acf(dmz,nlags=N_acf,fft=False)
    
    #==================
    ## SI単位への変更
    #==================
    ## !! caution !! 
    ## これを利用するときは，dmはデバイ，
    ## Vはnm^3
    ## でないといけない．

    #V=traj.unitcell_volumes[0]
    print("##==============")
    print("体積V[nm^3]")
    print(VOLUME)
    print("##==============")
    print("")
    
    print("##==============")
    print("双極子の大きさの平均[Debye]")
    print((np.sqrt(np.mean(dmx**2+dmy**2+dmz**2))))
    print("##==============")
    print("")
    eps0  = 8.8541878128e-12
    debye = 3.33564e-30               #[Cm]
    nm3   = 1.0e-27                     #[m^3] 
    nm    = 1.0e-9  # 体積がnm^3になってるのでmへ変換．
    kb    = 1.38064852e-23
    T =300                            #[K]

    ##=================
    ## 静的誘電率の計算
    eps_0 = 1+((np.mean(dmx**2+dmy**2+dmz**2))*(debye**2))/(3.0*VOLUME*nm3*kb*T*eps0)
    print("##==================")
    print("## 静的誘電定数")
    print(eps_0)
    print("##==================")
    
    return N_acf, acf_x, acf_y, acf_z, eps_0



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
