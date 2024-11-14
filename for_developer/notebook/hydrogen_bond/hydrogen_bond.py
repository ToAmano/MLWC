
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, WARNING, CRITICAL
# create logger
logger = getLogger('simple_example')
logger.setLevel(DEBUG)

# create console handler and set level to debug
ch = StreamHandler()
ch.setLevel(DEBUG)

# create formatter
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# from logging import getLogger, Formatter, StreamHandler, FileHandler, DEBUG, INFO

# def get_module_logger(module, verbose):
#     logger = getLogger(module)
#     logger = _set_handler(logger, StreamHandler(), False)
#     logger = _set_handler(logger, FileHandler(LOG_DIR), verbose)
#     logger.setLevel(DEBUG)
#     logger.propagate = False
#     return logger


# def _set_handler(logger, handler, verbose):
#     if verbose:
#         handler.setLevel(DEBUG)
#     else:
#         handler.setLevel(INFO)
#     handler.setFormatter(Formatter('%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s'))
#     logger.addHandler(handler)
#     return logger

import ase
import ase.io

# gasのaseデータを読み込み（32分子分）
# traj = ase.io.read("methanol_gas/dipole_20ps_5/mol_wan.xyz",index=":1280000")

# liquidのデータを読み込み（128分子分）
traj_liquid = ase.io.read("mol_wan.xyz",index=":10000")


def hydrogen_bond_trii(r):
    """水素結合の距離相関
    閾値以下なら(r-r_th)，r_th以上で0になる関数
    通常，水素結合距離は3.5Aくらい．

    Args:
        r (_type_): _description_

    Returns:
        _type_: _description_
    """
    import numpy as np
    r_th=6
    return np.where(r<r_th,(r-r_th)**2,0)


def hydrogen_bond_Pagliai(r):
    import numpy as np
    r_th=2
    sigma=0.5
    return np.where(r>r_th,np.exp(-(r_th-r)**2/(2*sigma*sigma)),1)

def hydrogen_bond_custom(r):
    """deepmpの使ってる記述子の形


    Args:
        r (_type_): _description_
    """
    import numpy as np
    r_c = 4
    r_th = 6
    s= np.where(r<r_c,1/r,np.where(r<r_th,(1/r)*(0.5*np.cos(np.pi*(r-r_c)/(r_th-r_c))+0.5),0))
    return s


# * itpデータの読み込み
# note :: itpファイルは記述子からデータを読み込む場合は不要なのでコメントアウトしておく
import ml.atomtype
itp_filename:str = "input_GMX.mol"
# 実際の読み込み
if itp_filename.endswith(".itp"):
    itp_data=ml.atomtype.read_itp(itp_filename)
elif itp_filename.endswith(".mol"):
    itp_data=ml.atomtype.read_mol(itp_filename)
else:
    print("ERROR :: itp_filename should end with .itp or .mol")
# bonds_list=itp_data.bonds_list
NUM_MOL_ATOMS=itp_data.num_atoms_per_mol


# NUM_ATOM_ALL = 17
# NUM_ATOM     = 6
NUM_MOL:int = 64 # !! number of molecules
#
# * 2024/1/24：：現状こっちを有力視している
# * 次に，H原子に注目して，Hから2番目に近いO原子までの距離を単純に計算する．
# * Hを基準にすれば，H原子が水素結合できる分子は一つだけなのでわかりやすいかもしれない．
import numpy as np
logger.setLevel(INFO)
ch.setLevel(INFO)
NUM_ATOM_ALL = 39
# 長さを格納するリスト
hydrogen_bond_list = np.zeros([len(traj_liquid),NUM_MOL*len(itp_data.h_oh)])
# OHボンドのH原子のリスト
hydrogen_list:list = [NUM_ATOM_ALL*mol_id+atom_id for mol_id in range(NUM_MOL) for atom_id in itp_data.h_oh ]
# OHボンドのO原子のリスト
oxygen_list:list   = [NUM_ATOM_ALL*mol_id+atom_id for mol_id in range(NUM_MOL) for atom_id in itp_data.o_oh ]
print(oxygen_list)


# 
from ase.geometry import get_distances
for counter,atoms in enumerate(traj_liquid): # frameに関するloop
    pos = atoms.get_positions()
    distances, distances_len = get_distances(pos[oxygen_list],pos[oxygen_list],cell=atoms.get_cell(),pbc=True)
    #distances = np.array(distances)
    # print(distances_len)
    #print(np.shape(distances_len))
    #print(distances)
    #print(np.shape(distances))
    hb_length = np.sort(distances_len,axis=1)[:,1] # 3番目に小さい近い原子を選択 (0: donor, 1: 多分同じ分子内のO原子)
    # print(np.shape(hb_length))
    hydrogen_bond_list[counter] = hb_length

import matplotlib.pyplot as plt
# plt.hist(hydrogen_bond_list.flatten(),bins=100)


len(hydrogen_bond_list)




import numpy as np
from scipy.signal import correlate

# N個の独立した時系列データを持つ2次元配列 (N行×T列)
N, T = 100, 1000  # N個の時系列、各時系列の長さT
data = np.random.randn(N, T)  # 例としてランダムなデータを使用


# 全ての時系列に対して自己相関を計算 (axis=1で各行に対して自己相関を計算)
# 'same' モードで時系列の長さを維持
# !! numpy correlate does not support FFT
correlations = np.apply_along_axis(lambda x: correlate(x, x, mode='full'), axis=0, arr=hydrogen_bond_list)
print(np.shape(correlations))

# 自己相関の平均化 (axis=0で全ての時系列に対する平均を取る)
mean_correlation = np.mean(correlations, axis=1)[len(hydrogen_bond_list)-1:]

# 結果をプロット
import matplotlib.pyplot as plt
plt.plot(mean_correlation)
plt.title('Average Autocorrelation of N Time Series')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


#
# * 計算したhydrogen_bond_listの自己相関を求める
#
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
#
import statsmodels.api as sm
import numpy as np

def autocorr(x):
    import numpy as np
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def autocorr_scipy(x):
    from scipy import signal
    result = signal.correlate(x, x, mode="same",method="fft")/len(x)
    return result[int(result.size/2):]

# * acfを計算する．
acf = 0 # np.zeros(len(traj_liquid))

for mol_id in range(NUM_MOL):
    acf += autocorr_scipy(hydrogen_bond_list[:,mol_id])




def calc_fourier(acf,TIMESTEP:float=2.5):
    # dt = 2* 10  # !! 0.5fs
    # au2ps = 2.4189e-17 /1.0e-15/1.0e3 
    # au2fs = 2.4189e-17 /1.0e-15
    # TIMESTEP = dt*au2fs
    TIMESTEP = TIMESTEP/1000 # fs to ps

    time_data=len(acf) # データの長さ
    freq=np.fft.fftfreq(time_data, d=TIMESTEP) # omega
    length=freq.shape[0]//2 + 1 # rfftでは，fftfreqのうちの半分しか使わない．
    rfreq=freq[0:length]

    #usage:: numpy.fft.fft(data, n=None, axis=-1, norm=None)
    ans=np.fft.rfft(acf, norm="forward" ) #こっちが1/Nがかかる規格化．
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
    ffteps1 = ans_times_omega.imag*(time_data*TIMESTEP)
    ffteps2 = ans_times_omega.real*(time_data*TIMESTEP)

    # 通常のfourier変換
    normal_fourier = ans.real*(time_data*TIMESTEP) # ここはrealが正しい？
    return rfreq, normal_fourier,ffteps2



vdos = calc_fourier(mean_correlation)
vdos2 = calc_fourier(acf)



#
# * unit_cellの計算結果の図示
import matplotlib as mpl
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,5),tight_layout=True) # figure, axesオブジェクトを作成

# ax.plot(diel_mean[0,:]*33.3, diel_mean[2,:]-0.0095*diel_mean[0,:],alpha=0.5, label="DNN", lw=3)  # 描画
# ax.plot(cpmd_diel_mean[0,:]*33.3, cpmd_diel_mean[2,:]-0.0095*cpmd_diel_mean[0,:], alpha=0.5, label="CPMD", lw=3)  # 描画

# ax.plot(diel_mean[0,:]*33.3, diel_mean[2,:],alpha=0.5, label="DNN", lw=3)  # 描画
# ax.plot(cpmd_diel_mean[0,:]*33.3, cpmd_diel_mean[2,:], alpha=0.5, label="CPMD", lw=3)  # 描画

ax.plot(vdos[0]*33.3, vdos[1], alpha=0.5, label="VDOS", lw=3)  # 描画
# ax.plot(vdos2[0]*33.3, vdos2[1], alpha=0.5, label="VDOS2", lw=3)  # 描画

# exp
# ax.scatter(expdata_eps1[:,0], expdata_eps1[:,1], label="Exp.", lw=3)  # 描画
# ax.scatter(expdata_eps2[:,0], expdata_eps2[:,1], label="Exp.", lw=3)  # 描画


# 各要素で設定したい文字列の取得
xticklabels = ax.get_xticklabels()
yticklabels = ax.get_yticklabels()
xlabel=r'frequency [$\mathrm{cm}^{-1}$]'
ylabel="hidrogen bond length (OH..O)"
# title="Dielectric function (liquid methanol)"


# 各要素の設定を行うsetコマンド
ax.set_xlabel(xlabel,fontsize=22)
ax.set_ylabel(ylabel,fontsize=22)

# ax.set_title(title,fontsize=22 )

XMAX=4000

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.1,XMAX)
ax.set_ylim(1e-6,1e3)
ax.grid()

# 2軸目（twinyを使い、y軸を共通にして同じグラフを書く）
ax2 = ax.twiny()
ax2.set_xlim(0,XMAX/33.3)
## ax2.plot(kayser/33.3, ffteps2_pred)
ax2.set_xlabel('frequency [THz]', fontsize=22)

ax.tick_params(axis='x', labelsize=20 )
ax.tick_params(axis='y', labelsize=20 )
ax2.tick_params(axis='x', labelsize=20 )

# ax.legend = ax.legend(*scatter.legend_elements(prop="colors"),loc="upper left", title="Ranking")

lgnd=ax.legend(loc="upper right",fontsize=20)
# lgnd.legendHandles[0]._sizes = [30]
# lgnd.legendHandles[0]._alpha = [1.0]


# pyplot.savefig("eps_real2.pdf",transparent=True)
# plt.show()
# fig.savefig("dielec_func_IR_0427_for_shorttalk2.pdf", transparent=True)
# ax.show()
# fig.delaxes(ax)

# plt.legend()
# plt.show()

# comp.plot(x="time",y=["mu_tot_x_pred","mu_tot_x_cpmd"])

