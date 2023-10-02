import numpy as np
import time

''' 
- ENERGIESとDIPOLEはそれぞれ1行のファイルで，これをnumpy.loadで読み込んでmergeする．
- WC_SPREADはそんなに重要ではないので，単純にcatするだけでいいように思う．従って，それは別途shell　scriptでやる．
- FTRAJECTORY：最重要．問題となるのはnp.loadの形の保証と，mergeの時に一行目に構造番号を追加すること．
- WANNIER_CENTER, WC_QUAD：形式がFTRAJECTORYと同じなのでFTRAJECTORYの方法が使える．問題となるのはnp.loadの形の保証．
'''

def save_file_1(data, filename):
    '''
    dataのshapeが(dim,7)の場合．
    '''
    print(data.shape)
    with open(filename, mode='w') as f:
        for i in range(np.len(data)):
            f.write("{0} {1} {2} {3} {4} {5} {6}".format(data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],data[i,5],data[i,6]))
    return 0


# TODO :: hard code
NUM_CONFIG=10001

# =======================
# energy
energy_list = [[] for i in range(NUM_CONFIG)]
# 時間計測開始
time_start = time.time()

for i in range(NUM_CONFIG):
    data = np.loadtxt("bulkjob/struc_"+str(i)+"/ENERGIES")
    # print(data.shape)
    # print(data.ndim)
    assert data.ndim == 1 # ENERGIESが0行，2行以上の場合にはエラー
    assert data.shape[0] == 8 # 要素が8個でない場合にはエラー
    data[0] = int(i)
    energy_list[i]=data
    # assert data.shape[0] = 8
# 時間計測終了
time_end = time.time()
# 経過時間（秒）
time_energy = time_end- time_start
print(time_energy, "[s]")
# save (ここを良い形で保存できるようにする)
np.savetxt("ENERGIES_merge_tmp.txt",np.array(energy_list))


# =======================-
# dipole
dipole_list = [[] for i in range(NUM_CONFIG)]
time_start = time.time()

for i in range(NUM_CONFIG):
    data = np.loadtxt("bulkjob/struc_"+str(i)+"/DIPOLE")
    # print(data.shape)
    # print(data.ndim)
    assert data.ndim == 1 # DIPOLEが0行，2行以上の場合にはエラー
    assert data.shape[0] == 7 # 要素が7個でない場合にはエラー
    data[0] = int(i)
    dipole_list[i]=data
    # assert data.shape[0] = 8
# 時間計測終了
time_end = time.time()
# 経過時間（秒）
time_dipole = time_end- time_start
print(time_dipole, "[s]")
# save (ここを良い形で保存できるようにする)
np.savetxt("DIPOLE_merge_tmp.txt",np.array(energy_list))


# =======================-
# ftrajectory
ftrajectory_list = [[] for i in range(NUM_CONFIG)]
time_start = time.time()
for i in range(NUM_CONFIG):
    ftraj_tmp = np.loadtxt("bulkjob/struc_"+str(i)+"/tmp/FTRAJECTORY")
    # print(ftraj_tmp)
    # print("")
    # ftraj_tmp = np.insert(ftraj_tmp, 0, i, axis=1)
    ftraj_tmp[:,0] = i
    ftrajectory_list[i] = ftraj_tmp
# >>>
time_dipole = time_end- time_start
print(time_dipole, "[s]")
np.savetxt("ftraj_merge.txt",np.array(ftrajectory_list))


# =======================
# wannier_center
wanniercenter_list = [[] for i in range(NUM_CONFIG)]
time_start = time.time()
for i in range(NUM_CONFIG):
    wanniercenter_tmp = np.loadtxt("bulkjob/struc_"+str(i)+"/tmp/WANNIER_CENTER")
    # print(ftraj_tmp)
    # print("")
    assert data.ndim == 2
    # assert data.shape[0] ==  # num_wannier
    assert data.shape[1] == 5
    # wanniercenter_tmp = np.insert(wanniercenter_tmp, 0, i, axis=1)
    wanniercenter_tmp[:,0] = i
    wanniercenter_list[i] = wanniercenter_tmp
# >>>
time_dipole = time_end- time_start
print(time_dipole, "[s]")
np.savetxt("WANNIER_CENTER_merge.txt",np.array(wanniercenter_list))


# =======================
# WC_QUAD
wcquad_list = [[] for i in range(NUM_CONFIG)]
time_start = time.time()
for i in range(NUM_CONFIG):
    wcquad_tmp = np.loadtxt("bulkjob/struc_"+str(i)+"/tmp/WC_QUAD")
    # print(ftraj_tmp)
    # print("")
    assert data.ndim == 2
    # assert data.shape[0] ==  # num_wannier
    assert data.shape[1] == 7
    # wanniercenter_tmp = np.insert(wanniercenter_tmp, 0, i, axis=1)
    wcquad_tmp[:,0] = i
    wcquad_list[i] = wcquad_tmp
# >>>
time_dipole = time_end- time_start
print(time_dipole, "[s]")
np.savetxt("WC_QUAD_merge.txt",np.array(wcquad_list))



