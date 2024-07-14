from __future__ import annotations # fugaku上のpython3.8で型指定をする方法（https://future-architect.github.io/articles/20201223/）


import argparse
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt


TEMPERATURE =293 # Kelvin
L = 16.2656  # change here (nm)
UNITCELL_VECTORS = np.array([
    [L, 0,0],
    [0, L,0],
    [0,0, L]
])
filename="traj/total_dipole_293K.txt"

def calc_eps0(cell_dipoles_pred, TEMPERATURE, UNITCELL_VECTORS):
    '''
    eps0だけ計算する．    
    '''
        # 誘電関数の計算まで
    import numpy as np
    # cell_dipoles_pred = np.load(filename)
    
    # N=int(np.shape(cell_dipoles_pred)[0]/2)
    N=int(np.shape(cell_dipoles_pred)[0])
    # N=99001
    # print("nlag :: ", N)

    # >>>>>>>>>>>
    eps0 = 8.8541878128e-12
    debye = 3.33564e-30
    nm3 = 1.0e-27
    nm = 1.0e-9
    A3 = 1.0e-30
    kb = 1.38064852e-23

    V = np.abs(np.dot(np.cross(UNITCELL_VECTORS[:,0],UNITCELL_VECTORS[:,1]),UNITCELL_VECTORS[:,2])) * A3
    ## V = np.abs(np.dot(np.cross(traj[0].UNITCELL_VECTOR[:,0],traj[0].UNITCELL_VECTOR[:,1]),traj[0].UNITCELL_VECTOR[:,2])) * A3
    # print("SUPERCELL VOLUME (m^3) :: ", V )

    # 予測値
    dMx_pred=cell_dipoles_pred[:,0]-np.mean(cell_dipoles_pred[:,0])
    dMy_pred=cell_dipoles_pred[:,1]-np.mean(cell_dipoles_pred[:,1])
    dMz_pred=cell_dipoles_pred[:,2]-np.mean(cell_dipoles_pred[:,2])
    
    # 平均値計算
    mean_M2=(np.mean(dMx_pred**2)+np.mean(dMy_pred**2)+np.mean(dMz_pred**2))
    mean_M=np.mean(dMx_pred)**2+np.mean(dMy_pred)**2+np.mean(dMz_pred)**2

    # 比誘電率
    eps_0 = 1.0 + ((mean_M2-mean_M)*debye**2)/(3.0*V*kb*TEMPERATURE*eps0)

    # 比誘電率
    # eps_0 = 1.0 + ((np.mean(dMx_pred**2+dMy_pred**2+dMz_pred**2))*debye**2)/(3.0*V*kbT*eps0)
    # print("EPS_0 {0}, mean_M {1}, mean_M2 {2}:: ".format(eps_0, mean_M, mean_M2))
    return [eps_0, mean_M2, mean_M]


#
# * eps_0の計算
# * acfの計算（絶対に必要なところ！！）
# TODO :: calc_acf関数を利用しているのでそれを置き換えたい

import numpy as np
eps_0 = 0
pred=[]
eps_list=[]

# length_list
length_list=[]

    
print(filename)
# cell_dipole_pred = np.load(filename)[:30000,] # result_dipole.npyから
cell_dipole_pred = np.loadtxt(filename)[:,1:] # totaldipole.txtから
print(len(cell_dipole_pred))
# eps_0_tmp, time, pred_data = calc_acf(cell_dipole_pred)
# length_list.append(len(pred_data))
# # eps_0_tmp, time, pred_data = calc_acf("../20230525_fugaku_20ps_dt05/wannier_dipole_"+str(i)+".npy")
# pred.append(pred_data) # TODO :: 長さの修正
# eps_list.append(eps_0_tmp)
[eps_0, mean_M2, mean_M] =  calc_eps0(cell_dipole_pred,TEMPERATURE, UNITCELL_VECTORS)

print(" =================== ")
print(f"eps_0 :: {eps_0}")
print(f"mean_M2 :: {mean_M2}")
print(f"mean_M :: {mean_M}")
print(" =================== ")    
# print(length_list)
