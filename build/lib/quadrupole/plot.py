# =================================
# 双極子モーメントの時間変化を図示する．
# =================================

#########  input  ########
## dipole_array: [:,3]次元の配列．各軸がD_x,y,z[D]の時間発展
##

def plot_dipole(dipole_array):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 全双極子のヒストグラムを出力
    plt.xlabel("Dipole_z[Debye]")
    plt.ylabel("num of molecule")
    plt.title(" total dipole_z[D] histgram")
    plt.hist(dipole_array[:,2], bins=100)
    #plt.xlim(-0.1,0.1)
    plt.legend()
    plt.show()

    # 各双極子成分の時間変化を出力
    plt.xlabel("timestep")
    plt.ylabel("Dipole[D]")
    plt.title(" total dipole[D] vs timestep")
    plt.plot(dipole_array[:,0], label="D_x")
    plt.plot(dipole_array[:,1], label="D_y")
    plt.plot(dipole_array[:,2], label="D_z")
    #plt.xlim(-0.1,0.1)
    plt.legend()
    plt.show()

    # 双極子の絶対値の時間変化を出力
    plt.xlabel("timestep")
    plt.ylabel("|Dipole| [Debye]")
    plt.title("total dipole[D] vs time")
    plt.plot(np.sqrt(dipole_array[:,0]**2+dipole_array[:,1]**2+dipole_array[:,2]**2))
    plt.legend()
    plt.show()
    #
    #
    return 0
    
